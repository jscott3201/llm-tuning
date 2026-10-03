"""Isolated SDK worker. The controller owns its lifetime and durable receipt."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.qat_preflight_io import collect_remote, evaluate_capture
from _common.qat_preflight_receipt import LIMITS, SOURCE_LIMIT, identity

PYTHON = "/usr/bin/python3.12"
BOOTSTRAP = """import hashlib, pathlib, subprocess, sys, tempfile
source = sys.stdin.buffer.read(65537)
if len(source) != int(sys.argv[2]) or hashlib.sha256(source).hexdigest() != sys.argv[1]:
    raise SystemExit(65)
with tempfile.TemporaryDirectory(prefix='qat-preflight-') as directory:
    path = pathlib.Path(directory) / 'qat_stack.py'
    path.write_bytes(source)
    result = subprocess.run([sys.executable, str(path)], check=False)
    raise SystemExit(result.returncode)
"""


def sandbox_options(image, app, tag):
    """The complete sandbox resource/security declaration, testable offline."""
    return {"app": app, "image": image, "tags": tag, "gpu": None,
            "cpu": (1, 1), "memory": (4096, 4096), "timeout": LIMITS["sandbox_seconds"],
            "block_network": True, "secrets": [], "volumes": {}, "network_file_systems": {},
            "encrypted_ports": [], "unencrypted_ports": [], "h2_ports": [],
            "include_oidc_identity_token": False, "pty": False,
            "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "CUDA_VISIBLE_DEVICES": ""}}


def emit(event, **fields):
    """Emit selected control events; never serialize SDK errors or configuration."""
    print("QAT_EVENT " + json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


async def run_preflight(config, sdk, acknowledge):
    """Create one owned App/Sandbox and execute only the source-bound validator."""
    source = base64.b64decode(config["source"], validate=True)
    if (not source or len(source) > SOURCE_LIMIT or len(source) != config["validator_bytes"]
            or hashlib.sha256(source).hexdigest() != config["validator_sha256"]):
        raise ValueError("validator_source_mismatch")
    image = await sdk.Image.from_id.aio(identity("image", config["image_id"]))
    app = sdk.App(config["description"])
    emit("allocating", kind="app")
    await acknowledge()
    async with app.run.aio(environment_name=config["environment"], detach=False):
        emit("allocated", kind="app", id=identity("app", app.app_id))
        emit("allocating", kind="sandbox")
        await acknowledge()
        sandbox = await sdk.Sandbox.create.aio(
            PYTHON, "-c", "import time; time.sleep(90)",
            **sandbox_options(image, app, config["tag"]))
        emit("allocated", kind="sandbox", id=identity("sandbox", sandbox.object_id))
        # Sandbox entrypoint logs are line-buffered by this SDK. Exec's raw
        # streams let us enforce byte bounds even without a newline.
        process = await sandbox.exec.aio(
            PYTHON, "-c", BOOTSTRAP, config["validator_sha256"], str(len(source)),
            text=False, bufsize=-1, pty=False, timeout=LIMITS["sandbox_seconds"], secrets=[])
        process.stdin.write(source)
        process.stdin.write_eof()
        await process.stdin.drain.aio()
        async def raw_stream(stream):
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose.aio()
        class RawProcess:
            stdout = raw_stream(process.stdout)
            stderr = raw_stream(process.stderr)
            async def wait(self):
                return await process.wait.aio()
        capture = await collect_remote(RawProcess(), asyncio.get_running_loop().time() + LIMITS["sandbox_seconds"])
        emit("result", result=evaluate_capture(capture))


async def active_sandboxes(sdk, app_id, tags=None):
    """Bound and deduplicate even an unexpectedly large exact-App inventory."""
    options = {"app_id": app_id}
    if tags is not None:
        options["tags"] = tags
    result = {}
    async for sandbox in sdk.Sandbox.list.aio(**options):
        target = identity("sandbox", sandbox.object_id)
        if len(result) >= 1024 or target in result:
            raise ValueError("invalid_sandbox_inventory")
        result[target] = sandbox
    return result


async def inventory(config, sdk):
    """Return only active identities under the exact owned App, without allocation."""
    app_id = identity("app", config["app_id"])
    all_ids = list(await active_sandboxes(sdk, app_id))
    tagged_ids = list(await active_sandboxes(sdk, app_id, config["tag"]))
    emit("inventory", app_id=app_id, sandbox_ids=all_ids, tagged_ids=tagged_ids)


async def terminate_owned(config, sdk, acknowledge):
    """Join termination only for acknowledged or uniquely tagged owned Sandboxes."""
    app_id = identity("app", config["app_id"])
    all_owned = await active_sandboxes(sdk, app_id)
    tagged = list(await active_sandboxes(sdk, app_id, config["tag"]))
    saved = config["sandbox_id"]
    if len(tagged) > 1 or not set(tagged) <= set(all_owned):
        raise ValueError("ambiguous_sandbox_inventory")
    if saved is not None:
        identity("sandbox", saved)
        if tagged and tagged != [saved]:
            raise ValueError("sandbox_identity_mismatch")
        targets = [saved] if saved in all_owned else []
    else:
        targets = tagged
    for target in targets:
        emit("sandbox_discovered", id=target)
        await acknowledge()
        try:
            await all_owned[target].terminate.aio(wait=True)
            emit("termination", status="acknowledged")
        except Exception:
            # Already-timed-out Sandboxes may raise from wait; final inventory
            # must independently confirm termination before cleanup can pass.
            emit("termination", status="unknown")


async def worker(config, mode):
    """Set the saved environment before importing the SDK or reading its config."""
    os.environ["MODAL_ENVIRONMENT"] = identity("environment", config["environment"])
    import modal

    if mode == "inventory":
        await inventory(config, modal)
        return
    reader = asyncio.StreamReader(limit=128)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    async def acknowledge():
        if await reader.readline() != b"ack\n":
            raise RuntimeError("receipt_ack_missing")
    try:
        if mode == "run":
            await run_preflight(config, modal, acknowledge)
        else:
            await terminate_owned(config, modal, acknowledge)
    finally:
        transport.close()


def main():
    """Private subprocess entrypoint; imports and help never contact Modal."""
    try:
        if len(sys.argv) != 2 or sys.argv[1] not in {"run", "inventory", "terminate"}:
            return 2
        # Only the bounded controller message is read here; no SDK is loaded yet.
        initial = sys.stdin.buffer.readline(131073)
        if len(initial) > 131072 or not initial.endswith(b"\n"):
            return 2
        config = json.loads(initial)
        asyncio.run(worker(config, sys.argv[1]))
        return 0
    except Exception:
        emit("failure", reason="sdk_operation_failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
