"""Receipt-driven CPU image preflight and allocation-free recovery."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import sys

from _common.qat_preflight_io import finish, run_local
from _common.qat_preflight_receipt import LIMITS, identity

WORKER = str(Path(__file__).with_name("qat_preflight_worker.py"))


class Backend:
    """Run native inventory commands and isolated SDK workers with fixed bounds."""
    def __init__(self, environment):
        self.environment = identity("environment", environment)
        self.child_environment = {**os.environ, "MODAL_ENVIRONMENT": self.environment}

    async def cli(self, arguments, deadline):
        result = await run_local([sys.executable, "-m", "modal", *arguments], deadline,
                                 environment=self.child_environment)
        if result["reason"] or result["returncode"] != 0:
            raise RuntimeError("modal_cli_failed")
        rows = json.loads(result["stdout"])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("invalid_inventory")
        return rows

    async def apps(self, deadline):
        return await self.cli(["app", "list", "--env", self.environment, "--json"], deadline)

    async def containers(self, app_id, deadline):
        return await self.cli(["container", "list", "--env", self.environment,
                               "--app-id", identity("app", app_id), "--json"], deadline)

    async def stop(self, app_id, deadline):
        # A nonzero result for an already-stopped App is resolved by live readback.
        await run_local([sys.executable, "-m", "modal", "app", "stop", "--env", self.environment,
                         "--yes", identity("app", app_id)], deadline, environment=self.child_environment)

    async def worker(self, mode, config, deadline, event):
        return await run_local([sys.executable, WORKER, mode], deadline, environment=self.child_environment,
                               initial=(json.dumps(config, separators=(",", ":")) + "\n").encode(),
                               event=event, stdout_limit=131072)

    async def terminate(self, receipt, deadline, *, writes=None):
        writes = writes if writes is not None else _ReceiptWrites()
        async def event(message):
            if message.get("event") == "sandbox_discovered":
                target = identity("sandbox", message.get("id"))
                data = receipt.data
                if data["allocation"]["sandbox"] == "not_requested" or data["sandbox_id"] not in {None, target}:
                    raise ValueError("unexpected_sandbox_identity")
                data["sandbox_id"] = target
                writes.save(receipt)
                return True
            if message.get("event") not in {"termination", "failure"}:
                raise ValueError("unexpected_termination_event")
            return False
        config = {key: receipt.data[key] for key in ("app_id", "sandbox_id", "tag", "environment")}
        outcome = await self.worker("terminate", config, deadline, event)
        return not outcome["reason"] and outcome["returncode"] == 0

    async def inventory(self, receipt, deadline):
        observed = None
        async def event(message):
            nonlocal observed
            if message.get("event") == "inventory" and observed is None:
                if message.get("app_id") != receipt["app_id"]:
                    raise ValueError("inventory_app_mismatch")
                for key in ("sandbox_ids", "tagged_ids"):
                    if not isinstance(message.get(key), list) or len(message[key]) > 1024:
                        raise ValueError("invalid_sandbox_inventory")
                    for value in message[key]:
                        identity("sandbox", value)
                observed = {key: message[key] for key in ("sandbox_ids", "tagged_ids")}
            elif message.get("event") != "failure":
                raise ValueError("unexpected_inventory_event")
            return False
        config = {key: receipt[key] for key in ("app_id", "tag", "environment")}
        result = await self.worker("inventory", config, deadline, event)
        if result["reason"] or result["returncode"] != 0 or observed is None:
            raise RuntimeError("sandbox_inventory_failed")
        return observed


def owned_app(rows, receipt):
    """Resolve one exact description and verify any previously acknowledged ID."""
    matches = [row for row in rows if row.get("Description") == receipt["description"]]
    if len(matches) > 1:
        raise ValueError("duplicate_app_description")
    saved = receipt["app_id"]
    if saved is not None:
        same_id = [row for row in rows if row.get("App ID") == saved]
        if len(same_id) != 1 or not matches or matches[0] is not same_id[0]:
            raise ValueError("app_identity_mismatch")
    if not matches:
        return None
    identity("app", matches[0].get("App ID"))
    if not isinstance(matches[0].get("State"), str):
        raise ValueError("invalid_app_state")
    return matches[0]


def sanitized_result(result):
    """Whitelist worker summary fields so receipts never retain raw diagnostics."""
    allowed = {"remote_exit_code", "streams_complete", "stream_bytes", "stream_sha256",
               "validator_passed", "reason", "checks"}
    if not isinstance(result, dict) or set(result) - allowed or type(result.get("validator_passed")) is not bool:
        raise ValueError("invalid_worker_result")
    if result.get("remote_exit_code") is not None and type(result["remote_exit_code"]) is not int:
        raise ValueError("invalid_worker_exit")
    reasons = {None, "incomplete_streams", "remote_nonzero", "invalid_validator_report",
               "remote_timeout", "output_limit", "nonbinary_stream", "remote_error"}
    if result.get("reason") not in reasons:
        raise ValueError("invalid_worker_reason")
    for field in ("streams_complete", "stream_bytes", "stream_sha256"):
        if not isinstance(result.get(field), dict) or set(result[field]) != {"stdout", "stderr"}:
            raise ValueError("invalid_worker_streams")
    for name, bound in (("stdout", LIMITS["stdout_bytes"]), ("stderr", LIMITS["stderr_bytes"])):
        if type(result["streams_complete"][name]) is not bool:
            raise ValueError("invalid_worker_eof")
        size = result["stream_bytes"][name]
        if type(size) is not int or not 0 <= size <= bound:
            raise ValueError("invalid_worker_output_size")
        if re.fullmatch(r"[0-9a-f]{64}", result["stream_sha256"][name]) is None:
            raise ValueError("invalid_worker_output_hash")
    checks = result.get("checks", {})
    if not isinstance(checks, dict) or set(checks) - {"metadata", "pip_check", "native", "vllm_cli"}:
        raise ValueError("invalid_worker_checks")
    for name, check in checks.items():
        if not isinstance(check, dict) or set(check) - {"status", "exit_code", "policy_status"}:
            raise ValueError("invalid_worker_check")
        if check.get("status") not in {"passed", "failed"} or type(check.get("exit_code")) is not int:
            raise ValueError("invalid_worker_check_status")
        if "policy_status" in check and check["policy_status"] != "passed":
            raise ValueError("invalid_worker_check_policy")
    if result["validator_passed"] and (result.get("reason") is not None or result.get("remote_exit_code") != 0
            or not all(result["streams_complete"].values()) or set(checks) != {"metadata", "pip_check", "native", "vllm_cli"}):
        raise ValueError("incomplete_worker_success")
    return result


@dataclass
class _ReceiptWrites:
    """Track write failures for one invocation without clearing durable history."""
    failed: bool = False

    def save(self, receipt):
        """Record every failed update and propagate it to withhold worker acknowledgments."""
        try:
            receipt.save()
        except Exception:
            self.failed = True
            receipt.data["receipt_write_failed"] = True
            raise


def save_cleanup(receipt, writes):
    """A failed receipt update must never prevent an owned resource stop."""
    try:
        writes.save(receipt)
    except Exception:
        pass


async def cleanup(receipt, backend, deadline, *, writes=None):
    """Stop only owned targets; absent snapshots cannot settle missing replies."""
    writes = writes if writes is not None else _ReceiptWrites()
    data = receipt.data
    data["cleanup"] = "unknown"
    save_cleanup(receipt, writes)
    if all(state == "not_requested" for state in data["allocation"].values()):
        data["cleanup"] = "verified"
        data["cleanup_evidence"] = {"no_allocation_requested": True}
        save_cleanup(receipt, writes)
        return
    try:
        async with asyncio.timeout_at(deadline):
            row = owned_app(await backend.apps(deadline), data)
            if row is None:
                data["cleanup_reason"] = "app_not_observed"
                return
            data["app_id"] = row["App ID"]
            save_cleanup(receipt, writes)
            try:
                await backend.terminate(receipt, min(deadline, asyncio.get_running_loop().time() + 10), writes=writes)
            except Exception:
                pass  # App stop and independent readback must still be attempted.
            await backend.stop(data["app_id"], deadline)
            while asyncio.get_running_loop().time() < deadline:
                row = owned_app(await backend.apps(deadline), data)
                if row is None:
                    raise ValueError("app_readback_missing")
                containers = await backend.containers(data["app_id"], deadline)
                sandboxes = await backend.inventory(data, deadline)
                evidence = {"app_state": row["State"] if row["State"] == "stopped" else "not_stopped",
                            "containers": len(containers), "sandboxes": len(sandboxes["sandbox_ids"]),
                            "tagged_sandboxes": len(sandboxes["tagged_ids"])}
                data["cleanup_evidence"] = evidence
                if row["State"] == "stopped" and not containers and not any(sandboxes.values()):
                    if "unresolved" in data["allocation"].values():
                        data["cleanup_reason"] = "allocation_reply_unresolved"
                    else:
                        data["cleanup"] = "verified"
                        data.pop("cleanup_reason", None)
                    return
                await asyncio.sleep(min(.5, max(0, deadline - asyncio.get_running_loop().time())))
            data["cleanup_reason"] = "cleanup_deadline"
    except asyncio.CancelledError:
        raise
    except Exception:
        data["cleanup_reason"] = "cleanup_readback_failed"
    finally:
        save_cleanup(receipt, writes)


async def preflight(receipt, backend, source, *, final_deadline=None):
    """Run one attempt, reserving final time for cleanup even after interruption."""
    writes = _ReceiptWrites()
    loop = asyncio.get_running_loop()
    final_deadline = final_deadline if final_deadline is not None else loop.time() + LIMITS["total_seconds"]
    work_deadline = final_deadline - LIMITS["cleanup_seconds"]
    data = receipt.data
    interrupted = False
    result_seen = False
    async def event(message):
        nonlocal result_seen
        kind = message.get("kind")
        if message.get("event") == "allocating" and kind in {"app", "sandbox"}:
            if data["allocation"][kind] != "not_requested":
                raise ValueError("duplicate_allocation_request")
            if kind == "sandbox" and data["allocation"]["app"] != "acknowledged":
                raise ValueError("sandbox_before_app_ack")
            data["allocation"][kind] = "unresolved"
            writes.save(receipt)
            return True
        if message.get("event") == "allocated" and kind in {"app", "sandbox"}:
            if data["allocation"][kind] != "unresolved":
                raise ValueError("unexpected_allocation_reply")
            data[kind + "_id"] = identity(kind, message.get("id"))
            data["allocation"][kind] = "acknowledged"
            writes.save(receipt)
            return False
        if message.get("event") == "result" and not result_seen:
            if data["allocation"]["sandbox"] != "acknowledged" or not isinstance(message.get("result"), dict):
                raise ValueError("result_before_sandbox_ack")
            result_seen = True
            data["result"] = sanitized_result(message["result"])
            writes.save(receipt)
            return False
        if message.get("event") == "failure":
            data["failure_reason"] = "sdk_operation_failed"
            writes.save(receipt)
            return False
        raise ValueError("unexpected_worker_event")
    try:
        async with asyncio.timeout_at(work_deadline):
            rows = await backend.apps(work_deadline)
            if any(row.get("Description") == data["description"] for row in rows):
                raise ValueError("baseline_description_present")
            data.update(baseline_absent=True, status="running")
            writes.save(receipt)
            config = {key: data[key] for key in ("image_id", "description", "tag", "environment",
                                                "validator_sha256", "validator_bytes")}
            config["source"] = base64.b64encode(source).decode("ascii")
            worker = await backend.worker("run", config, work_deadline, event)
            passed = not worker["reason"] and worker["returncode"] == 0 and result_seen and data["result"]["validator_passed"]
            data["status"] = "passed" if passed else "failed"
            if not passed:
                data["failure_reason"] = "preflight_failed"
    except asyncio.CancelledError:
        interrupted = True
        data.update(status="interrupted", failure_reason="interrupted")
        asyncio.current_task().uncancel()
    except Exception:
        data.update(status="failed", failure_reason="preflight_failed")
    finally:
        save_cleanup(receipt, writes)
        # The process helper reserves at most two seconds to kill/join its child.
        def cancelled_cleanup():
            nonlocal interrupted
            interrupted = True
            data.update(status="interrupted", failure_reason="interrupted")
        await finish(cleanup(receipt, backend, final_deadline - 2, writes=writes), on_cancel=cancelled_cleanup)
        save_cleanup(receipt, writes)
        if data["cleanup"] != "verified" or writes.failed:
            data["status"] = "interrupted" if interrupted else "failed"
            save_cleanup(receipt, writes)
    return 130 if interrupted else (0 if data["status"] == "passed" else 1)


async def recover(receipt, backend):
    """Reconcile the saved attempt without creating any remote resource."""
    writes = _ReceiptWrites()
    deadline = asyncio.get_running_loop().time() + LIMITS["cleanup_seconds"] - 2
    interrupted = False
    def cancelled_cleanup():
        nonlocal interrupted
        interrupted = True
    await finish(cleanup(receipt, backend, deadline, writes=writes), on_cancel=cancelled_cleanup)
    return 130 if interrupted else (0 if receipt.data["cleanup"] == "verified" and not writes.failed else 1)
