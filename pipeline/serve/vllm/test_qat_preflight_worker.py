"""Installed-SDK declarations and independent worker/CLI controls, offline only."""
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from _common import qat_preflight as controller
from _common import qat_preflight_worker as worker
from _common.qat_preflight_receipt import Receipt, validator_source
from test_qat_preflight_io import Stream, passing_report


def installed_byte_reader(chunks=(), *, delay=0, descriptor=1):
    """Exercise the SDK's translated public byte reader with a local router stub."""
    from modal._utils.async_utils import synchronizer
    from modal.io_streams import (
        _BytesStreamReaderThroughCommandRouter, _StreamReader,
        _StreamReaderThroughCommandRouterParams,
    )
    started, closed = threading.Event(), threading.Event()
    class Router:
        async def exec_stdio_read(self, task_id, object_id, file_descriptor, deadline):
            assert (task_id, object_id, file_descriptor, deadline) == ("ta-offline", "ex-offline", descriptor, None)
            started.set()
            try:
                for chunk in chunks:
                    yield SimpleNamespace(data=chunk)
                await asyncio.sleep(delay)
            finally:
                closed.set()
    # Avoid the SDK constructor's dedicated-loop assertion without replacing any
    # iteration/close implementation or creating a client.
    params = _StreamReaderThroughCommandRouterParams(descriptor, "ta-offline", "ex-offline", Router(), None)
    reader = _StreamReader.__new__(_StreamReader)
    reader._impl = _BytesStreamReaderThroughCommandRouter(params)
    reader._read_gen = None
    return synchronizer._translate_out(reader), reader, started, closed


class DeclarationTests(unittest.TestCase):
    def test_installed_sdk_accepts_all_declared_argument_names_without_hydration(self):
        import modal
        app = modal.App("offline-declaration-control")
        image = object()
        options = worker.sandbox_options(image, app, {"qat-preflight": "authored"})
        inspect.signature(modal.Sandbox.create).bind(worker.PYTHON, "-c", "pass", **options)
        inspect.signature(modal.Sandbox.exec).bind(
            object(), worker.PYTHON, "-c", worker.BOOTSTRAP, "0" * 64, "1",
            text=False, bufsize=-1, pty=False, timeout=90, secrets=[])
        inspect.signature(app.run).bind(app, environment_name="selected", detach=False)
        self.assertIsNone(app.app_id)
        self.assertEqual(options["cpu"], (1, 1))
        self.assertEqual(options["memory"], (4096, 4096))
        self.assertEqual(options["timeout"], 90)
        self.assertTrue(options["block_network"])
        for key in ("gpu", "pty", "include_oidc_identity_token", "secrets", "volumes",
                    "network_file_systems", "encrypted_ports", "unencrypted_ports", "h2_ports"):
            self.assertFalse(options[key])

    def test_cli_import_and_help_do_not_import_modal_or_allocate(self):
        script = PIPELINE / "serve/vllm/qat_preflight.py"
        code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'modal' not in sys.modules"
        imported = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, timeout=3)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        for args in (["--help"], ["run", "--help"], ["recover", "--help"]):
            result = subprocess.run([sys.executable, str(script), *args], capture_output=True, timeout=3)
            self.assertEqual(result.returncode, 0, result.stderr)
        missing = subprocess.run([sys.executable, str(script), "run"], capture_output=True, timeout=3)
        self.assertNotEqual(missing.returncode, 0)

    def test_bootstrap_enforces_source_hash_before_executing_and_removes_tempfile(self):
        source = b"from pathlib import Path; print(Path(__file__).parent)\n"
        digest = hashlib.sha256(source).hexdigest()
        result = subprocess.run([sys.executable, "-c", worker.BOOTSTRAP, digest, str(len(source))],
                                input=source, capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(Path(result.stdout.decode().strip()).exists())
        rejected = subprocess.run([sys.executable, "-c", worker.BOOTSTRAP, "0" * 64, str(len(source))],
                                  input=source, capture_output=True, timeout=3)
        self.assertEqual(rejected.returncode, 65)
        self.assertEqual(rejected.stdout, b"")


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        source = validator_source()
        return {"source": base64.b64encode(source).decode(), "validator_bytes": len(source),
                "validator_sha256": hashlib.sha256(source).hexdigest(), "image_id": "im-prebuilt",
                "description": "qat-preflight-authored", "tag": {"qat-preflight": "authored"},
                "environment": "selected"}

    async def test_exact_source_transfer_acknowledgments_and_raw_exec_are_ordered(self):
        events, acknowledged = [], []
        config = self.config()
        async def ack():
            acknowledged.append(events[-1])
        stdin = SimpleNamespace(write=Mock(), write_eof=Mock(), drain=SimpleNamespace(aio=AsyncMock()))
        process = SimpleNamespace(stdin=stdin, stdout=Stream([json.dumps(passing_report()).encode()]),
                                  stderr=Stream(), wait=SimpleNamespace(aio=AsyncMock(return_value=0)))
        process.stdout.aclose = AsyncMock()
        process.stderr.aclose = AsyncMock()
        sandbox = SimpleNamespace(object_id="sb-owned", exec=SimpleNamespace(aio=AsyncMock(return_value=process)))
        async def create(*args, **kwargs):
            self.assertEqual(acknowledged[-1], ("allocating", {"kind": "sandbox"}))
            return sandbox
        class App:
            app_id = "ap-owned"
            def __init__(self, name):
                self.name = name
                self.run = SimpleNamespace(aio=self.context)
            @asynccontextmanager
            async def context(self, **kwargs):
                assert kwargs == {"environment_name": "selected", "detach": False}
                assert acknowledged[-1] == ("allocating", {"kind": "app"})
                yield self
        image = object()
        sdk = SimpleNamespace(App=App, Image=SimpleNamespace(from_id=SimpleNamespace(aio=AsyncMock(return_value=image))),
                              Sandbox=SimpleNamespace(create=SimpleNamespace(aio=AsyncMock(side_effect=create))))
        with patch.object(worker, "emit", side_effect=lambda event, **fields: events.append((event, fields))):
            await worker.run_preflight(config, sdk, ack)
        sdk.Image.from_id.aio.assert_awaited_once_with("im-prebuilt")
        self.assertEqual([event for event, _ in events], ["allocating", "allocated", "allocating", "allocated", "result"])
        self.assertTrue(events[-1][1]["result"]["validator_passed"])
        stdin.write.assert_called_once_with(validator_source())
        stdin.write_eof.assert_called_once()
        stdin.drain.aio.assert_awaited_once()
        process.stdout.aclose.assert_awaited_once()
        process.stderr.aclose.assert_awaited_once()
        kwargs = sandbox.exec.aio.call_args.kwargs
        self.assertEqual(kwargs, {"text": False, "bufsize": -1, "pty": False, "timeout": 90, "secrets": []})

    async def run_installed_streams(self, stdout, stderr, *, code=0, wait_delay=0):
        async def wait():
            await asyncio.sleep(wait_delay)
            return code
        process = SimpleNamespace(
            stdin=SimpleNamespace(write=Mock(), write_eof=Mock(), drain=SimpleNamespace(aio=AsyncMock())),
            stdout=stdout, stderr=stderr, wait=SimpleNamespace(aio=wait))
        sandbox = SimpleNamespace(object_id="sb-owned", exec=SimpleNamespace(aio=AsyncMock(return_value=process)))
        class App:
            app_id = "ap-owned"
            def __init__(self, name):
                self.run = SimpleNamespace(aio=self.context)
            @asynccontextmanager
            async def context(self, **kwargs):
                yield self
        sdk = SimpleNamespace(App=App,
                              Image=SimpleNamespace(from_id=SimpleNamespace(aio=AsyncMock(return_value=object()))),
                              Sandbox=SimpleNamespace(create=SimpleNamespace(aio=AsyncMock(return_value=sandbox))))
        with patch.object(worker, "emit") as emit:
            await worker.run_preflight(self.config(), sdk, AsyncMock())
        return emit.call_args.kwargs["result"]

    async def test_installed_sdk_byte_stream_eof_closes_and_preserves_remote_exit(self):
        for code, payload, reason in ((0, json.dumps(passing_report()).encode(), None),
                                      (1, b'{"status":"failed"}', "remote_nonzero")):
            with self.subTest(code=code):
                stdout, stdout_internal, _, stdout_closed = installed_byte_reader([payload])
                stderr, stderr_internal, _, stderr_closed = installed_byte_reader(descriptor=2)
                self.assertTrue(inspect.iscoroutinefunction(stdout.aclose))
                self.assertFalse(hasattr(stdout.aclose, "aio"))
                result = await self.run_installed_streams(stdout, stderr, code=code)
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["remote_exit_code"], code)
                self.assertEqual(result["streams_complete"], {"stdout": True, "stderr": True})
                self.assertIsNone(stdout_internal._read_gen)
                self.assertIsNone(stderr_internal._read_gen)
                self.assertTrue(stdout_closed.is_set() and stderr_closed.is_set())

    async def test_installed_sdk_byte_stream_overflow_closes_both_readers(self):
        stdout, stdout_internal, _, stdout_closed = installed_byte_reader([b"x" * 40000], delay=30)
        stderr, stderr_internal, _, stderr_closed = installed_byte_reader(delay=30, descriptor=2)
        result = await self.run_installed_streams(stdout, stderr, wait_delay=30)
        self.assertEqual(result["reason"], "output_limit")
        self.assertEqual(result["stream_bytes"]["stdout"], 32768)
        self.assertIsNone(stdout_internal._read_gen)
        self.assertIsNone(stderr_internal._read_gen)
        self.assertTrue(stdout_closed.is_set() and stderr_closed.is_set())

    async def test_installed_sdk_byte_stream_cancellation_closes_both_readers(self):
        stdout, stdout_internal, stdout_started, stdout_closed = installed_byte_reader(delay=30)
        stderr, stderr_internal, stderr_started, stderr_closed = installed_byte_reader(delay=30, descriptor=2)
        task = asyncio.create_task(self.run_installed_streams(stdout, stderr, wait_delay=30))
        try:
            async with asyncio.timeout(2):
                while not stdout_started.is_set() or not stderr_started.is_set():
                    await asyncio.sleep(.005)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNone(stdout_internal._read_gen)
            self.assertIsNone(stderr_internal._read_gen)
            self.assertTrue(stdout_closed.is_set() and stderr_closed.is_set())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_source_mismatch_fails_before_image_lookup_or_allocation(self):
        sdk = Mock()
        config = {**self.config(), "validator_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "source_mismatch"):
            await worker.run_preflight(config, sdk, AsyncMock())
        self.assertEqual(sdk.mock_calls, [])

    async def test_termination_discovers_one_owned_tag_and_joins_exact_handle(self):
        owned = SimpleNamespace(object_id="sb-owned", terminate=SimpleNamespace(aio=AsyncMock()))
        unrelated = SimpleNamespace(object_id="sb-unrelated", terminate=SimpleNamespace(aio=AsyncMock()))
        calls = []
        async def listing(**kwargs):
            calls.append(kwargs)
            if kwargs.get("tags"):
                yield owned
            else:
                yield owned
                yield unrelated
        sdk = SimpleNamespace(Sandbox=SimpleNamespace(list=SimpleNamespace(aio=listing)))
        events = []
        config = {"app_id": "ap-owned", "sandbox_id": None, "tag": {"qat-preflight": "authored"}}
        with patch.object(worker, "emit", side_effect=lambda event, **fields: events.append((event, fields))):
            ack = AsyncMock()
            await worker.terminate_owned(config, sdk, ack)
        self.assertEqual(events[0], ("sandbox_discovered", {"id": "sb-owned"}))
        ack.assert_awaited_once()
        owned.terminate.aio.assert_awaited_once_with(wait=True)
        unrelated.terminate.aio.assert_not_awaited()
        self.assertEqual(calls, [{"app_id": "ap-owned"}, {"app_id": "ap-owned", "tags": config["tag"]}])

    async def test_duplicate_or_changed_sandbox_identity_is_not_terminated(self):
        handles = [SimpleNamespace(object_id="sb-one", terminate=SimpleNamespace(aio=AsyncMock())),
                   SimpleNamespace(object_id="sb-two", terminate=SimpleNamespace(aio=AsyncMock()))]
        async def listing(**kwargs):
            for handle in handles:
                yield handle
        sdk = SimpleNamespace(Sandbox=SimpleNamespace(list=SimpleNamespace(aio=listing)))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            await worker.terminate_owned({"app_id": "ap-owned", "sandbox_id": None, "tag": {}}, sdk, AsyncMock())
        for handle in handles:
            handle.terminate.aio.assert_not_awaited()

    async def test_timed_out_termination_emits_unknown_for_independent_readback(self):
        handle = SimpleNamespace(object_id="sb-owned", terminate=SimpleNamespace(aio=AsyncMock(side_effect=TimeoutError())))
        async def listing(**kwargs):
            yield handle
        sdk = SimpleNamespace(Sandbox=SimpleNamespace(list=SimpleNamespace(aio=listing)))
        with patch.object(worker, "emit") as emit:
            await worker.terminate_owned({"app_id": "ap-owned", "sandbox_id": "sb-owned", "tag": {}}, sdk, AsyncMock())
        emit.assert_any_call("termination", status="unknown")

    async def test_actual_recovery_worker_acknowledges_saved_id_before_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = Receipt(root / "receipt.json")
            try:
                receipt.create("selected", "im-prebuilt", validator_source())
                receipt.data.update(baseline_absent=True, app_id="ap-owned")
                receipt.data["allocation"] = {"app": "acknowledged", "sandbox": "unresolved"}
                receipt.save()
                (root / "modal.py").write_text(
                    "import json, os\n"
                    "from pathlib import Path\n"
                    "from types import SimpleNamespace\n"
                    "assert os.environ['MODAL_ENVIRONMENT'] == 'selected'\n"
                    "async def terminate(*, wait):\n"
                    "    assert wait is True\n"
                    "    receipt = json.loads(Path(os.environ['AUTHORED_RECEIPT']).read_text())\n"
                    "    assert receipt['sandbox_id'] == 'sb-owned'\n"
                    "    Path(os.environ['AUTHORED_MARKER']).touch()\n"
                    "async def listing(**kwargs):\n"
                    "    assert kwargs['app_id'] == 'ap-owned'\n"
                    "    yield SimpleNamespace(object_id='sb-owned', terminate=SimpleNamespace(aio=terminate))\n"
                    "Sandbox = SimpleNamespace(list=SimpleNamespace(aio=listing))\n")
                backend = controller.Backend("selected")
                marker = root / "terminated"
                backend.child_environment.update(PYTHONPATH=str(root), AUTHORED_RECEIPT=str(receipt.path),
                                                 AUTHORED_MARKER=str(marker))
                self.assertTrue(await backend.terminate(receipt, asyncio.get_running_loop().time() + 3))
                self.assertTrue(marker.exists())
                self.assertEqual(receipt.data["sandbox_id"], "sb-owned")
                self.assertEqual(receipt.data["allocation"]["sandbox"], "unresolved")
            finally:
                receipt.close()

    async def test_backend_binds_saved_environment_and_uses_exact_stop_id(self):
        with patch.dict(os.environ, {"MODAL_ENVIRONMENT": "wrong"}):
            backend = controller.Backend("selected")
        self.assertEqual(backend.child_environment["MODAL_ENVIRONMENT"], "selected")
        result = {"reason": None, "returncode": 0, "stdout": b"[]", "stderr": b""}
        with patch.object(controller, "run_local", AsyncMock(return_value=result)) as run:
            await backend.apps(100)
            await backend.containers("ap-owned", 100)
            await backend.stop("ap-owned", 100)
        self.assertEqual(run.call_args_list[0].args[0][-5:], ["app", "list", "--env", "selected", "--json"])
        stop = run.call_args_list[-1].args[0]
        self.assertEqual(stop[-4:], ["--env", "selected", "--yes", "ap-owned"])
        self.assertNotIn("qat-preflight-authored", stop)


if __name__ == "__main__":
    unittest.main()
