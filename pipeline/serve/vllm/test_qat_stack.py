"""Authored CPU guard regressions; no serving packages or cloud resources needed."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from _common import qat_stack as guard


def distribution(name, version, requires=()):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=list(requires))


def good_distributions():
    return [distribution("torch", "2.13.0+cu129"),
            distribution("torchvision", "0.28.0+cu129", ["torch==2.13.0"]),
            distribution("vllm", "0.30.0+cu129", ["torch==2.13.0", "torchvision==0.28.0"]),
            distribution("transformers", "5.17.0")]


class DependencyTests(unittest.TestCase):
    def test_confirmed_torch_mismatch_fails_with_both_dependents(self):
        installed = good_distributions()
        installed[0] = distribution("torch", "2.14.0")
        with patch.object(guard.metadata, "distributions", return_value=installed):
            result = guard.metadata_check()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["version_mismatches"][0]["actual"], "2.14.0")
        self.assertEqual({row["package"] for row in result["dependency_issues"]}, {"torchvision", "vllm"})

    def test_cuda_build_and_missing_packages_fail(self):
        for installed in (good_distributions()[1:],
                          [distribution("torch", "2.13.0+cu130"), *good_distributions()[1:]]):
            with self.subTest(installed=installed), patch.object(guard.metadata, "distributions", return_value=installed):
                self.assertEqual(guard.metadata_check()["status"], "failed")

    def test_active_requirements_are_checked_but_unselected_extras_are_not(self):
        installed = [distribution("owner", "1", [
            'missing-extra; extra == "optional"', 'missing-platform; platform_system == "Windows"',
            'present>=2; platform_system == "Linux"']), distribution("present", "1")]
        issues, overrides = guard.dependency_report(installed, {"platform_system": "Linux"})
        self.assertEqual([row["dependency"] for row in issues], ["present"])
        self.assertEqual(overrides, [])

    def test_an_override_is_an_exact_conflict_not_a_package_wildcard(self):
        installed = [distribution("owner", "1", ["dependency==2"]), distribution("dependency", "3")]
        exact = frozenset({("owner", "1", "dependency", "==2", "3")})
        with patch.object(guard, "DEPENDENCY_OVERRIDES", exact):
            issues, overrides = guard.dependency_report(installed, {})
            self.assertEqual(issues, [])
            self.assertEqual(len(overrides), 1)
            installed[1] = distribution("dependency", "4")
            issues, overrides = guard.dependency_report(installed, {})
        self.assertEqual(len(issues), 1)
        self.assertEqual(overrides, [])

    def test_matching_metadata_passes_without_importing_torch(self):
        with patch.object(guard.metadata, "distributions", return_value=good_distributions()):
            self.assertEqual(guard.metadata_check()["status"], "passed")

    def test_requested_extras_propagate_through_dependencies(self):
        installed = [distribution("torch", "1", ["cuda-toolkit[cublas]==2"]),
                     distribution("cuda-toolkit", "2", ['nvidia-cublas-cu12==3; extra == "cublas"']),
                     distribution("nvidia-cublas-cu12", "4")]
        issues, overrides = guard.dependency_report(installed, {})
        self.assertEqual([row["dependency"] for row in issues], ["nvidia-cublas-cu12"])
        self.assertEqual(overrides, [])

    def test_only_shipped_nccl_exception_is_reported_as_override(self):
        installed = [distribution("torch", "2.13.0+cu129", ["nvidia-nccl-cu12==2.29.7"]),
                     distribution("nvidia-nccl-cu12", "2.30.7")]
        issues, overrides = guard.dependency_report(installed, {})
        self.assertEqual(issues, [])
        self.assertEqual(len(overrides), 1)
        installed[1].version = "2.30.8"
        issues, overrides = guard.dependency_report(installed, {})
        self.assertEqual(len(issues), 1)
        self.assertEqual(overrides, [])

    def test_duplicate_distribution_identity_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            guard.dependency_report([distribution("same-name", "1"), distribution("same_name", "2")], {})


class NativeTests(unittest.TestCase):
    def test_loader_error_and_missing_operator_fail_without_device_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            extension = Path(directory) / "_C.so"
            extension.touch()
            dist = SimpleNamespace(files=[PurePosixPath("torchvision/_C.so")], locate_file=lambda _: extension)
            load = Mock()
            registered = Mock(return_value=True)
            torch = SimpleNamespace(__version__="2.13.0+cu129", version=SimpleNamespace(cuda="12.9"),
                                    ops=SimpleNamespace(load_library=load),
                                    _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=registered))
            torch.tensor = Mock(side_effect=lambda data, **kw: data)
            nms = Mock(return_value=SimpleNamespace(tolist=lambda: [0, 2]))
            vision = SimpleNamespace(__version__="0.28.0+cu129", ops=SimpleNamespace(nms=nms))
            with patch.dict(sys.modules, {"torch": torch, "torchvision": vision}), patch.object(
                    guard.metadata, "distribution", return_value=dist), patch.object(guard, "nccl_runtime_version", return_value={}):
                self.assertTrue(guard.native_check()["cpu_nms_registered"])
                self.assertEqual(torch.tensor.call_args_list[0].kwargs, {"device": "cpu"})
                self.assertEqual(torch.tensor.call_args_list[1].kwargs, {"device": "cpu"})
                nms.assert_called_once_with([[0., 0., 2., 2.], [0., 0., 2., 2.], [4., 4., 6., 6.]], [.9, .8, .7], .5)
                load.assert_called_once_with(str(extension))
                registered.assert_called_once_with("torchvision::nms", "CPU")
                load.side_effect = OSError("authored incompatible native library")
                with self.assertRaisesRegex(OSError, "incompatible"):
                    guard.native_check()
                load.side_effect = None
                registered.return_value = False
                with self.assertRaisesRegex(RuntimeError, "not registered"):
                    guard.native_check()

    def test_nccl_runtime_query_requires_matching_record_and_actual_version(self):
        import base64
        import ctypes
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libnccl.so.2"
            library.write_bytes(b"authored library fixture")
            digest = base64.urlsafe_b64encode(hashlib.sha256(library.read_bytes()).digest()).rstrip(b"=").decode()
            record = guard.metadata.PackagePath("nvidia/nccl/lib/libnccl.so.2")
            record.hash = SimpleNamespace(mode="sha256", value=digest)
            dist = SimpleNamespace(version="2.30.7", files=[record], locate_file=lambda _: library)
            def query(pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 23007
                return 0
            function = Mock(side_effect=query)
            with patch.object(guard.metadata, "distribution", return_value=dist), patch.object(
                    ctypes, "CDLL", return_value=SimpleNamespace(ncclGetVersion=function)) as loader:
                self.assertEqual(guard.nccl_runtime_version()["runtime"], 23007)
                loader.assert_called_once_with(str(library))
                function.side_effect = None
                function.return_value = 1
                with self.assertRaisesRegex(RuntimeError, "unexpected NCCL runtime"):
                    guard.nccl_runtime_version()
                library.write_bytes(b"overwritten by a different CUDA package")
                with self.assertRaisesRegex(RuntimeError, "RECORD"):
                    guard.nccl_runtime_version()


class ProcessTests(unittest.TestCase):
    def run_python(self, code, **kwargs):
        return guard.run_child([sys.executable, "-c", code], **kwargs)

    def test_both_streams_and_split_utf8_are_collected_before_success(self):
        result = self.run_python("import os,time; os.write(1,b'\\xe2'); os.write(2,b'warning'); "
                                 "time.sleep(.02); os.write(1,b'\\x82\\xac')", timeout=2)
        self.assertEqual((result["status"], result["exit_code"]), ("passed", 0))
        self.assertEqual(result["stdout"], "€")
        self.assertEqual(result["stderr"], "warning")

    def test_nonzero_exit_does_not_pass_with_success_looking_output(self):
        result = self.run_python("print('{\"status\":\"passed\"}'); raise SystemExit(7)", timeout=2)
        self.assertEqual((result["status"], result["exit_code"]), ("failed", 7))

    def test_overflow_on_either_stream_is_bounded_and_child_is_joined(self):
        original = subprocess.Popen
        for fd in (1, 2):
            processes = []
            def spawn(*args, **kwargs):
                child = original(*args, **kwargs)
                processes.append(child)
                return child
            with self.subTest(fd=fd), patch.object(guard.subprocess, "Popen", side_effect=spawn):
                result = self.run_python(f"import os,time; os.write({fd},b'x'*100000); time.sleep(30)",
                                         timeout=2, stream_limit=128)
            self.assertEqual(result["reason"], "output_limit")
            self.assertLessEqual(len(result["stdout"].encode()), 128)
            self.assertLessEqual(len(result["stderr"].encode()), 128)
            self.assertIsNotNone(processes[0].returncode)

    def test_timeout_joins_child_even_when_it_closes_both_pipes(self):
        started = time.monotonic()
        result = self.run_python("import os,time; os.close(1); os.close(2); time.sleep(30)", timeout=.15)
        self.assertEqual(result["reason"], "timeout")
        self.assertIsNotNone(result["exit_code"])
        self.assertLess(time.monotonic() - started, 3)

    def test_exited_leader_with_descendant_holding_pipes_does_not_pass(self):
        result = self.run_python("import subprocess,sys; subprocess.Popen([sys.executable,'-c',"
                                 "'import time; time.sleep(30)'])", timeout=.15)
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual(result["exit_code"], 0)


class PolicyTests(unittest.TestCase):
    @staticmethod
    def child(text, **values):
        return {"status": "passed", "exit_code": 0, "stdout": text, "stderr": "", **values}

    def test_zero_exit_failed_or_malformed_json_is_rejected(self):
        for text in ('{"status":"failed","extension_load":{"status":"failed"}}', "not-json", "[]"):
            with self.subTest(text=text), patch.object(guard, "run_child", return_value=self.child(text)):
                self.assertEqual(guard.check_stack()["status"], "failed")

    def test_cpu_platform_is_scoped_to_cli_child_and_expected_help_is_required(self):
        for help_text, expected in (("usage: vllm [-h] {serve,chat}", "passed"), ("", "failed")):
            children = [self.child('{"status":"passed"}'), self.child("No broken requirements found."),
                        self.child('{"status":"passed"}'), self.child(help_text)]
            with patch.dict("os.environ", {}, clear=True), patch.object(guard, "run_child", side_effect=children) as run:
                self.assertEqual(guard.check_stack()["status"], expected)
            self.assertEqual(len(run.call_args_list), 4)
            for call in run.call_args_list[:3]:
                self.assertNotIn("VLLM_TARGET_DEVICE", call.kwargs["environment"])
            final = run.call_args_list[-1]
            self.assertEqual(final.kwargs["environment"]["VLLM_TARGET_DEVICE"], "cpu")
            self.assertEqual(final.args[0][1:], ["-m", "vllm.entrypoints.cli.main", "--help"])


    def test_pip_conflict_is_retained_and_only_exact_override_is_accepted(self):
        row = {"package": "torch", "package_version": "2.13.0+cu129", "dependency": "nvidia-nccl-cu12",
               "required": "==2.29.7", "installed": "2.30.7"}
        conflict = self.child("torch 2.13.0+cu129 has requirement nvidia-nccl-cu12==2.29.7, but you have nvidia-nccl-cu12 2.30.7.\n",
                              status="failed", exit_code=1, reason="nonzero_exit")
        self.assertTrue(guard.pip_policy(conflict, [row]))
        for change in ({"stdout": conflict["stdout"] + "unrelated conflict\n"}, {"reason": "timeout"},
                       {"reason": "output_limit"}, {"stderr": "No module named pip"}, {"exit_code": 2},
                       {"stdout": conflict["stdout"].replace("2.30.7.", "2.30.8.")}):
            with self.subTest(change=change):
                self.assertFalse(guard.pip_policy({**conflict, **change}, [row]))
        children = [self.child(json.dumps({"status": "passed", "declared_dependency_overrides": [row]})),
                    conflict.copy(), self.child('{"status":"passed"}'), self.child("usage: vllm {serve}")]
        with patch.object(guard, "run_child", side_effect=children):
            report = guard.check_stack()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checks"]["pip_check"]["status"], "failed")
        self.assertEqual(report["checks"]["pip_check"]["exit_code"], 1)
        self.assertEqual(report["checks"]["pip_check"]["stdout"], conflict["stdout"])
        self.assertEqual(report["checks"]["pip_check"]["policy_status"], "passed")

    def test_failed_policy_returns_nonzero_and_report_is_bounded(self):
        for report in ({"status": "failed", "error": "authored failure"},
                       {"status": "passed", "oversized": "x" * guard.REPORT_LIMIT}):
            output = io.StringIO()
            with patch.object(guard, "check_stack", return_value=report), redirect_stdout(output):
                self.assertEqual(guard.main([]), 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "failed")
            self.assertLessEqual(len(output.getvalue().encode()), guard.REPORT_LIMIT)


if __name__ == "__main__":
    unittest.main()
