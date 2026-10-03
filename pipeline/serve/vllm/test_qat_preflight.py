"""Receipt, authority, late-allocation and recovery controls; no Modal access."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from _common import qat_preflight as control
from _common.qat_preflight_receipt import Receipt, validator_source


class FakeBackend:
    def __init__(self, receipt, mode="success"):
        self.receipt, self.mode = receipt, mode
        self.rows = [{"App ID": "ap-unrelated", "Description": "unrelated", "State": "running"}]
        self.stops, self.creates = [], 0
        self.active = False
        self.entered = asyncio.Event()
        self.cleanup_entered = asyncio.Event()

    async def apps(self, deadline):
        return copy.deepcopy(self.rows)

    async def worker(self, mode, config, deadline, event):
        self.creates += 1
        self.assert_saved("not_requested", "not_requested")
        await event({"event": "allocating", "kind": "app"})
        self.assert_saved("unresolved", "not_requested")
        self.rows.append({"App ID": "ap-owned", "Description": config["description"], "State": "running"})
        if self.mode == "lost_app":
            return {"reason": "local_timeout", "returncode": -9}
        await event({"event": "allocated", "kind": "app", "id": "ap-owned"})
        await event({"event": "allocating", "kind": "sandbox"})
        self.assert_saved("acknowledged", "unresolved")
        self.active = True
        if self.mode == "lost_sandbox":
            return {"reason": "local_timeout", "returncode": -9}
        await event({"event": "allocated", "kind": "sandbox", "id": "sb-owned"})
        self.assert_saved("acknowledged", "acknowledged")
        self.entered.set()
        if self.mode == "cancel":
            await asyncio.sleep(30)
        if self.mode != "missing_result":
            await event({"event": "result", "result": {
                "validator_passed": self.mode != "failed_report", "remote_exit_code": 0,
                "streams_complete": {"stdout": True, "stderr": True},
                "stream_bytes": {"stdout": 100, "stderr": 0},
                "stream_sha256": {"stdout": "0" * 64, "stderr": "0" * 64},
                "reason": None if self.mode != "failed_report" else "invalid_validator_report",
                "checks": {name: {"status": "passed", "exit_code": 0} for name in
                           ("metadata", "pip_check", "native", "vllm_cli")}}})
        return {"reason": None, "returncode": 7 if self.mode == "worker_nonzero" else 0}

    def assert_saved(self, app, sandbox):
        saved = json.loads(self.receipt.path.read_text())
        assert saved["allocation"] == {"app": app, "sandbox": sandbox}

    async def terminate(self, receipt, deadline, *, writes=None):
        self.active = False
        return True

    async def stop(self, app_id, deadline):
        self.stops.append(app_id)
        self.cleanup_entered.set()
        if self.mode == "slow_cleanup":
            await asyncio.sleep(.08)
        if self.mode == "stop_failed":
            raise RuntimeError("authored stop failure")
        for row in self.rows:
            if row["App ID"] == app_id:
                row["State"] = "stopped"
        self.active = False

    async def containers(self, app_id, deadline):
        if self.mode == "readback_failed":
            raise RuntimeError("authored private inventory must not be saved")
        return [{"container": "owned"}] if self.active else []

    async def inventory(self, data, deadline):
        ids = ["sb-owned"] if self.active else []
        return {"sandbox_ids": ids, "tagged_ids": ids}


class ReceiptTests(unittest.TestCase):
    def test_exclusive_creation_atomic_recovery_and_environment_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            owner = Receipt(path)
            try:
                owner.create("selected", "im-prebuilt", validator_source())
                with self.assertRaisesRegex(ValueError, "receipt_in_use"):
                    Receipt(path)
                saved = owner.data.copy()
                with self.assertRaises(FileExistsError):
                    owner.create("selected", "im-prebuilt", validator_source())
                self.assertEqual(json.loads(path.read_text()), saved)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            finally:
                owner.close()
            recovery = Receipt(path)
            try:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    recovery.load("different")
                recovery.load("selected")
                self.assertEqual(recovery.data, saved)
                recovery.data["status"] = "failed"
                recovery.save()
                self.assertEqual(json.loads(path.read_text())["status"], "failed")
            finally:
                recovery.close()

    def test_tampered_ownership_and_limits_are_rejected_before_remote_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Receipt(Path(directory) / "receipt.json")
            try:
                receipt.create("selected", "im-prebuilt", validator_source())
                original = copy.deepcopy(receipt.data)
                for field, value in (("description", "unrelated"), ("limits", {}), ("image_id", "latest")):
                    receipt.path.write_text(json.dumps({**original, field: value}))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        receipt.load("selected")
            finally:
                receipt.close()


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.receipt = Receipt(Path(self.directory.name) / "receipt.json")
        self.source = validator_source()
        self.receipt.create("selected", "im-prebuilt", self.source)

    def tearDown(self):
        self.receipt.close()
        self.directory.cleanup()

    def acknowledged(self):
        self.receipt.data.update(baseline_absent=True, app_id="ap-owned", sandbox_id="sb-owned", status="failed")
        self.receipt.data["allocation"] = {"app": "acknowledged", "sandbox": "acknowledged"}
        self.receipt.save()
        backend = FakeBackend(self.receipt)
        backend.rows.append({"App ID": "ap-owned", "Description": self.receipt.data["description"], "State": "running"})
        backend.active = True
        return backend

    async def test_success_saves_before_allocating_and_preserves_unrelated_app(self):
        backend = FakeBackend(self.receipt)
        self.assertEqual(await control.preflight(self.receipt, backend, self.source), 0)
        self.assertEqual(self.receipt.data["cleanup"], "verified")
        self.assertEqual(backend.stops, ["ap-owned"])
        self.assertEqual(backend.rows[0]["State"], "running")
        self.assertEqual(self.receipt.data["sandbox_id"], "sb-owned")
        self.assertNotIn("unrelated", self.receipt.path.read_text())

    async def test_zero_worker_exit_without_valid_result_still_fails_and_cleans(self):
        for mode in ("failed_report", "missing_result", "worker_nonzero"):
            with self.subTest(mode=mode):
                self.receipt.data["allocation"] = {"app": "not_requested", "sandbox": "not_requested"}
                self.receipt.data.update(app_id=None, sandbox_id=None, baseline_absent=False)
                backend = FakeBackend(self.receipt, mode)
                self.assertEqual(await control.preflight(self.receipt, backend, self.source), 1)
                self.assertEqual(self.receipt.data["cleanup"], "verified")

    async def test_lost_app_or_sandbox_reply_cannot_be_cleared_by_empty_inventory(self):
        for mode in ("lost_app", "lost_sandbox"):
            with self.subTest(mode=mode):
                self.receipt.data["allocation"] = {"app": "not_requested", "sandbox": "not_requested"}
                self.receipt.data.update(app_id=None, sandbox_id=None, baseline_absent=False)
                backend = FakeBackend(self.receipt, mode)
                self.assertEqual(await control.preflight(self.receipt, backend, self.source), 1)
                self.assertEqual(self.receipt.data["cleanup"], "unknown")
                self.assertEqual(self.receipt.data["cleanup_reason"], "allocation_reply_unresolved")
                self.assertEqual(backend.stops, ["ap-owned"])
                self.assertEqual(await control.recover(self.receipt, backend), 1)
                self.assertEqual(backend.creates, 1)

    async def test_late_app_after_empty_snapshot_remains_unknown_and_is_stopped_on_recovery(self):
        self.receipt.data.update(baseline_absent=True)
        self.receipt.data["allocation"]["app"] = "unresolved"
        backend = FakeBackend(self.receipt)
        self.assertEqual(await control.recover(self.receipt, backend), 1)
        self.assertEqual(backend.stops, [])
        backend.rows.append({"App ID": "ap-late", "Description": self.receipt.data["description"], "State": "running"})
        self.assertEqual(await control.recover(self.receipt, backend), 1)
        self.assertEqual(backend.stops, ["ap-late"])
        self.assertEqual(backend.creates, 0)
        self.assertEqual(self.receipt.data["allocation"]["app"], "unresolved")

    async def test_baseline_presence_and_duplicate_discovery_never_stop_unrelated_resources(self):
        backend = FakeBackend(self.receipt)
        backend.rows[0]["Description"] = self.receipt.data["description"]
        self.assertEqual(await control.preflight(self.receipt, backend, self.source), 1)
        self.assertEqual(backend.creates, 0)
        self.assertEqual(backend.stops, [])
        backend = self.acknowledged()
        backend.rows[0]["Description"] = self.receipt.data["description"]
        self.assertEqual(await control.recover(self.receipt, backend), 1)
        self.assertEqual(backend.stops, [])
        backend.rows[0]["Description"] = "unrelated"
        backend.rows[1]["Description"] = "changed"
        self.assertEqual(await control.recover(self.receipt, backend), 1)
        self.assertEqual(backend.stops, [])

    async def test_already_stopped_is_verified_but_stop_or_readback_failure_is_unknown(self):
        backend = self.acknowledged()
        backend.rows[1]["State"] = "stopped"
        backend.active = False
        self.assertEqual(await control.recover(self.receipt, backend), 0)
        for mode in ("stop_failed", "readback_failed"):
            backend = self.acknowledged()
            backend.mode = mode
            self.assertEqual(await control.recover(self.receipt, backend), 1)
            self.assertEqual(self.receipt.data["cleanup"], "unknown")
            self.assertNotIn("authored private", self.receipt.path.read_text())

    async def test_repeated_cancellation_finishes_cleanup_and_returns_interrupted(self):
        backend = FakeBackend(self.receipt, "cancel")
        task = asyncio.create_task(control.preflight(self.receipt, backend, self.source))
        await backend.entered.wait()
        backend.mode = "slow_cleanup"
        task.cancel()
        await backend.cleanup_entered.wait()
        task.cancel()
        asyncio.get_running_loop().call_later(.02, task.cancel)
        self.assertEqual(await task, 130)
        self.assertEqual(self.receipt.data["cleanup"], "verified")
        self.assertEqual(self.receipt.data["status"], "interrupted")
        self.assertFalse([other for other in asyncio.all_tasks() if other is not asyncio.current_task() and not other.done()])

    async def test_recovery_reopens_durable_receipt_without_creating_resources(self):
        original = self.acknowledged()
        path = self.receipt.path
        self.receipt.close()
        self.receipt = Receipt(path)
        self.receipt.load("selected")
        backend = FakeBackend(self.receipt)
        backend.rows = original.rows
        backend.active = True
        self.assertEqual(await control.recover(self.receipt, backend), 0)
        self.assertEqual(backend.creates, 0)
        self.assertEqual(backend.stops, ["ap-owned"])

    async def test_empty_tag_subset_does_not_hide_active_exact_app_sandbox(self):
        backend = self.acknowledged()
        async def unexpected(data, deadline):
            return {"sandbox_ids": ["sb-extra"], "tagged_ids": []}
        backend.inventory = unexpected
        await control.cleanup(self.receipt, backend, asyncio.get_running_loop().time() + .03)
        self.assertEqual(self.receipt.data["cleanup"], "unknown")
        self.assertEqual(self.receipt.data["cleanup_evidence"]["sandboxes"], 1)

    async def test_recovery_cancellation_is_reported_after_joined_cleanup(self):
        backend = self.acknowledged()
        backend.mode = "slow_cleanup"
        task = asyncio.create_task(control.recover(self.receipt, backend))
        await backend.cleanup_entered.wait()
        task.cancel()
        asyncio.get_running_loop().call_later(.02, task.cancel)
        self.assertEqual(await task, 130)
        self.assertEqual(self.receipt.data["cleanup"], "verified")

    async def test_recovery_after_transient_run_receipt_failure_preserves_history(self):
        backend = FakeBackend(self.receipt)
        save = self.receipt.save
        failed_once = False
        def fail_once_after_success():
            nonlocal failed_once
            if not failed_once and self.receipt.data["status"] == "passed" and self.receipt.data["cleanup"] == "pending":
                failed_once = True
                raise OSError("authored transient disk failure")
            save()
        with patch.object(self.receipt, "save", side_effect=fail_once_after_success):
            self.assertEqual(await control.preflight(self.receipt, backend, self.source), 1)
        self.assertTrue(failed_once)
        self.assertEqual(self.receipt.data["status"], "failed")
        self.assertEqual(self.receipt.data["cleanup"], "verified")
        self.assertTrue(self.receipt.data["receipt_write_failed"])
        path = self.receipt.path
        self.receipt.close()
        self.receipt = Receipt(path)
        self.receipt.load("selected")
        recovered = FakeBackend(self.receipt)
        recovered.rows = backend.rows
        self.assertEqual(await control.recover(self.receipt, recovered), 0)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["cleanup"], "verified")
        self.assertTrue(saved["receipt_write_failed"])
        self.assertEqual(recovered.creates, 0)

    async def test_new_recovery_write_failure_fails_only_its_invocation(self):
        backend = self.acknowledged()
        self.receipt.data["receipt_write_failed"] = True
        self.receipt.save()
        save = self.receipt.save
        failed_once = False
        def fail_first_save():
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise OSError("authored new recovery disk failure")
            save()
        with patch.object(self.receipt, "save", side_effect=fail_first_save):
            self.assertEqual(await control.recover(self.receipt, backend), 1)
        self.assertTrue(failed_once)
        self.assertEqual(self.receipt.data["cleanup"], "verified")
        self.assertTrue(self.receipt.data["receipt_write_failed"])
        self.assertEqual(await control.recover(self.receipt, backend), 0)
        self.assertTrue(json.loads(self.receipt.path.read_text())["receipt_write_failed"])

    async def test_recovery_discovery_write_failure_withholds_ack_and_records_history(self):
        backend = self.acknowledged()
        actual = control.Backend("selected")
        backend.terminate = actual.terminate
        save = self.receipt.save
        in_callback = False
        failed_once = False
        acknowledgments = []
        async def worker(mode, config, deadline, event):
            nonlocal in_callback
            self.assertEqual(mode, "terminate")
            in_callback = True
            try:
                acknowledgments.append(await event({"event": "sandbox_discovered", "id": "sb-owned"}))
            except OSError:
                return {"reason": "local_process_error", "returncode": -9}
            finally:
                in_callback = False
            return {"reason": None, "returncode": 0}
        def fail_discovery_save_once():
            nonlocal failed_once
            if in_callback and not failed_once:
                failed_once = True
                raise OSError("authored discovery receipt failure")
            save()
        with patch.object(actual, "worker", side_effect=worker), patch.object(
                self.receipt, "save", side_effect=fail_discovery_save_once):
            self.assertEqual(await control.recover(self.receipt, backend), 1)
            self.assertTrue(failed_once)
            self.assertEqual(acknowledgments, [])
            self.assertEqual(backend.stops, ["ap-owned"])
            saved = json.loads(self.receipt.path.read_text())
            self.assertEqual(saved["cleanup"], "verified")
            self.assertTrue(saved["receipt_write_failed"])
            self.assertEqual(await control.recover(self.receipt, backend), 0)
            self.assertEqual(acknowledgments, [True])
        self.assertTrue(json.loads(self.receipt.path.read_text())["receipt_write_failed"])

    async def test_transient_allocation_ack_write_failure_records_history(self):
        backend = FakeBackend(self.receipt)
        save = self.receipt.save
        failed_once = False
        def fail_app_ack_once():
            nonlocal failed_once
            if not failed_once and self.receipt.data["allocation"]["app"] == "acknowledged":
                failed_once = True
                raise OSError("authored allocation receipt failure")
            save()
        with patch.object(self.receipt, "save", side_effect=fail_app_ack_once):
            self.assertEqual(await control.preflight(self.receipt, backend, self.source), 1)
        self.assertTrue(failed_once)
        self.assertEqual(backend.stops, ["ap-owned"])
        saved = json.loads(self.receipt.path.read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["cleanup"], "verified")
        self.assertTrue(saved["receipt_write_failed"])

    async def test_receipt_write_failure_after_allocation_cannot_skip_cleanup(self):
        backend = FakeBackend(self.receipt)
        save = self.receipt.save
        def fail_after_ack():
            if self.receipt.data["allocation"]["app"] == "acknowledged":
                raise OSError("authored disk failure")
            save()
        with patch.object(self.receipt, "save", side_effect=fail_after_ack):
            self.assertEqual(await control.preflight(self.receipt, backend, self.source), 1)
        self.assertEqual(backend.stops, ["ap-owned"])
        self.assertEqual(backend.rows[1]["State"], "stopped")

    async def test_cleanup_deadline_bounds_a_stalled_readback(self):
        backend = self.acknowledged()
        async def stalled(deadline):
            await asyncio.sleep(30)
        backend.apps = stalled
        await control.cleanup(self.receipt, backend, asyncio.get_running_loop().time() + .03)
        self.assertEqual(self.receipt.data["cleanup"], "unknown")


if __name__ == "__main__":
    unittest.main()
