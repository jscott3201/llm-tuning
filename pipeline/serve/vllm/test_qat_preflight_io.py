"""Independent subprocess and cancellation controls for CPU preflight I/O."""
import asyncio
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from _common.qat_preflight_io import collect_remote, evaluate_capture, run_local


def passing_report():
    checks = {name: {"status": "passed", "exit_code": 0} for name in ("metadata", "pip_check", "native", "vllm_cli")}
    for name in ("metadata", "native"):
        checks[name]["observation"] = {"status": "passed"}
    checks["pip_check"].update(policy_status="passed", stdout="No broken requirements found.", stderr="")
    return {"schema": "qat-cpu-stack-v1", "status": "passed", "checks": checks}


class Stream:
    def __init__(self, chunks=(), delay=0):
        self.chunks, self.delay = chunks, delay
        self.joined = False

    async def __aiter__(self):
        try:
            for chunk in self.chunks:
                yield chunk
            await asyncio.sleep(self.delay)
        finally:
            self.joined = True


class Remote:
    def __init__(self, stdout=(), stderr=(), code=0, delay=0, eof_delay=0):
        self.stdout, self.stderr = Stream(stdout, eof_delay), Stream(stderr, eof_delay)
        self.code, self.delay = code, delay
        self.wait_joined = False

    async def wait(self):
        try:
            await asyncio.sleep(self.delay)
            return self.code
        finally:
            self.wait_joined = True


class RemoteTests(unittest.IsolatedAsyncioTestCase):
    async def capture(self, remote, timeout=.5, **kwargs):
        return await collect_remote(remote, asyncio.get_running_loop().time() + timeout, **kwargs)

    async def test_zero_exit_actual_process_with_failed_json_is_rejected(self):
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "print('{\"status\":\"failed\"}')",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        class Raw:
            async def wait(self):
                return await child.wait()
            @staticmethod
            async def chunks(reader):
                while chunk := await reader.read(17):
                    yield chunk
            stdout = None
            stderr = None
        raw = Raw()
        raw.stdout, raw.stderr = raw.chunks(child.stdout), raw.chunks(child.stderr)
        captured = await self.capture(raw)
        self.assertEqual(captured["returncode"], 0)
        self.assertFalse(evaluate_capture(captured)["validator_passed"])
        self.assertEqual(evaluate_capture(captured)["reason"], "invalid_validator_report")

    async def test_complete_valid_report_and_split_utf8_preserve_bytes(self):
        payload = json.dumps(passing_report()).encode()
        remote = Remote([payload[:30], payload[30:]], [b"\xe2", b"\x82\xac"], eof_delay=.03)
        started = time.monotonic()
        result = evaluate_capture(await self.capture(remote))
        self.assertTrue(result["validator_passed"])
        self.assertGreaterEqual(time.monotonic() - started, .025)
        self.assertEqual(result["stream_bytes"]["stderr"], 3)
        self.assertTrue(remote.stdout.joined and remote.stderr.joined and remote.wait_joined)

    async def test_nonzero_missing_malformed_failed_and_incomplete_reports_fail(self):
        for payload, code in ((b"", 0), (b"not json", 0), (b'{"status":"passed"}', 0),
                              (b'{"schema":"qat-cpu-stack-v1","status":"failed"}', 0),
                              (json.dumps(passing_report()).encode(), 7), (b"\xff", 0)):
            with self.subTest(payload=payload, code=code):
                self.assertFalse(evaluate_capture(await self.capture(Remote([payload], code=code)))["validator_passed"])
        report = passing_report()
        del report["checks"]["native"]
        self.assertFalse(evaluate_capture(await self.capture(Remote([json.dumps(report).encode()])))["validator_passed"])

    async def test_declared_nccl_exception_passes_but_unapproved_pip_conflict_fails(self):
        report = passing_report()
        report["checks"]["metadata"]["observation"]["declared_dependency_overrides"] = [{
            "package": "torch", "package_version": "2.13.0+cu129", "dependency": "nvidia-nccl-cu12",
            "required": "==2.29.7", "installed": "2.30.7"}]
        report["checks"]["pip_check"].update(
            status="failed", reason="nonzero_exit", exit_code=1,
            stdout="torch 2.13.0+cu129 has requirement nvidia-nccl-cu12==2.29.7, but you have nvidia-nccl-cu12 2.30.7.\n")
        self.assertTrue(evaluate_capture(await self.capture(Remote([json.dumps(report).encode()])))["validator_passed"])
        report["checks"]["pip_check"]["stdout"] += "unapproved conflict\n"
        self.assertFalse(evaluate_capture(await self.capture(Remote([json.dumps(report).encode()])))["validator_passed"])

    async def test_late_eof_cannot_be_replaced_by_remote_zero(self):
        remote = Remote([json.dumps(passing_report()).encode()], eof_delay=2)
        captured = await self.capture(remote, timeout=.03)
        self.assertEqual(captured["returncode"], 0)
        self.assertEqual(captured["reason"], "remote_timeout")
        self.assertFalse(evaluate_capture(captured)["validator_passed"])
        self.assertTrue(remote.stdout.joined and remote.stderr.joined and remote.wait_joined)

    async def test_overflow_on_either_raw_stream_is_bounded_and_tasks_join(self):
        for name in ("stdout", "stderr"):
            remote = Remote(**{name: [b"x" * 10000]}, delay=2)
            captured = await self.capture(remote, stdout_limit=64, stderr_limit=64)
            self.assertEqual(captured["reason"], "output_limit")
            self.assertEqual(len(captured[name]), 64)
            self.assertTrue(remote.stdout.joined and remote.stderr.joined and remote.wait_joined)

    async def test_cancellation_joins_all_remote_tasks(self):
        remote = Remote(delay=2, eof_delay=2)
        task = asyncio.create_task(self.capture(remote, timeout=3))
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(remote.stdout.joined and remote.stderr.joined and remote.wait_joined)


class LocalTests(unittest.IsolatedAsyncioTestCase):
    async def run_python(self, source, timeout=.4, **kwargs):
        return await run_local([sys.executable, "-c", source],
                               asyncio.get_running_loop().time() + timeout, **kwargs)

    async def test_native_pipe_bytes_and_nonzero_are_preserved_without_decoding(self):
        result = await self.run_python("import os; os.write(1,b'\\xff'); os.write(2,b'warning'); raise SystemExit(7)")
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["stdout"], b"\xff")
        self.assertEqual(result["stderr"], b"warning")

    async def test_real_overflow_and_timeout_kill_join_child_and_descendant(self):
        for source, expected in (("import os,time; os.write(1,b'x'*1000000); time.sleep(30)", "local_output_limit"),
                                 ("import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])", "local_timeout")):
            with self.subTest(expected=expected):
                started = time.monotonic()
                result = await self.run_python(source, timeout=.05, stdout_limit=64)
                self.assertEqual(result["reason"], expected)
                self.assertIsNotNone(result["returncode"])
                self.assertLessEqual(len(result["stdout"]), 64)
                self.assertLess(time.monotonic() - started, 1)

    async def test_group_kill_failure_still_joins_child_through_exact_process(self):
        with patch("os.killpg", side_effect=PermissionError("authored process-group failure")):
            result = await self.run_python("import time; time.sleep(30)", timeout=.03)
        self.assertEqual(result["reason"], "local_cleanup_error")
        self.assertIsNotNone(result["returncode"])

    async def test_cancellation_while_spawn_reply_is_pending_still_joins_child(self):
        original = asyncio.create_subprocess_exec
        spawned = []
        ready = asyncio.Event()
        async def delayed_reply(*args, **kwargs):
            child = await original(*args, **kwargs)
            spawned.append(child)
            ready.set()
            await asyncio.sleep(.05)
            return child
        try:
            with patch("asyncio.create_subprocess_exec", side_effect=delayed_reply):
                task = asyncio.create_task(self.run_python("import time; time.sleep(30)", timeout=2))
                await ready.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIsNotNone(spawned[0].returncode)
        finally:
            for child in spawned:
                if child.returncode is None:
                    child.kill()
                await child.communicate()

    async def test_repeated_cancellation_during_local_join_leaves_no_tasks(self):
        entered = asyncio.Event()
        joined = asyncio.Event()
        async def event(message):
            entered.set()
            try:
                await asyncio.sleep(.1)
            finally:
                joined.set()
        task = asyncio.create_task(self.run_python(
            "import time; print('QAT_EVENT {}', flush=True); time.sleep(30)", timeout=2, event=event))
        await entered.wait()
        task.cancel()
        asyncio.get_running_loop().call_later(.02, task.cancel)
        asyncio.get_running_loop().call_later(.04, task.cancel)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(joined.is_set())
        self.assertFalse([other for other in asyncio.all_tasks() if other is not asyncio.current_task() and not other.done()])


if __name__ == "__main__":
    unittest.main()
