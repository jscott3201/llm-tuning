"""Authored CPU guard regressions; no serving packages or cloud resources needed."""
from __future__ import annotations

from contextlib import redirect_stdout
from packaging.utils import canonicalize_name
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


def installed_map(distributions):
    return {canonicalize_name(dist.metadata["Name"]): dist for dist in distributions}


def good_distributions():
    return [distribution("torch", "2.13.0+cu129"),
            distribution("torchvision", "0.28.0+cu129", ["torch==2.13.0"]),
            distribution("vllm", "0.30.0+cu129", ["torch==2.13.0", "torchvision==0.28.0"]),
            distribution("transformers", "5.17.0")]


class DependencyTests(unittest.TestCase):
    def test_confirmed_torch_mismatch_fails_with_both_dependents(self):
        installed = good_distributions()
        installed[0] = distribution("torch", "2.14.0")
        with patch.object(guard, "active_distributions", return_value=installed_map(installed)):
            result = guard.metadata_check()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["version_mismatches"][0]["actual"], "2.14.0")
        self.assertEqual({row["package"] for row in result["dependency_issues"]}, {"torchvision", "vllm"})

    def test_cuda_build_and_missing_packages_fail(self):
        for installed in (good_distributions()[1:],
                          [distribution("torch", "2.13.0+cu130"), *good_distributions()[1:]]):
            with self.subTest(installed=installed), patch.object(guard, "active_distributions", return_value=installed_map(installed)):
                self.assertEqual(guard.metadata_check()["status"], "failed")

    def test_active_requirements_are_checked_but_unselected_extras_are_not(self):
        installed = [distribution("owner", "1", [
            'missing-extra; extra == "optional"', 'missing-platform; platform_system == "Windows"',
            'present>=2; platform_system == "Linux"']), distribution("present", "1")]
        issues, overrides = guard.dependency_report(installed_map(installed), {"platform_system": "Linux"})
        self.assertEqual([row["dependency"] for row in issues], ["present"])
        self.assertEqual(overrides, [])

    def test_an_override_is_an_exact_conflict_not_a_package_wildcard(self):
        installed = [distribution("owner", "1", ["dependency==2"]), distribution("dependency", "3")]
        exact = frozenset({("owner", "1", "dependency", "==2", "3")})
        with patch.object(guard, "DEPENDENCY_OVERRIDES", exact):
            issues, overrides = guard.dependency_report(installed_map(installed), {})
            self.assertEqual(issues, [])
            self.assertEqual(len(overrides), 1)
            installed[1] = distribution("dependency", "4")
            issues, overrides = guard.dependency_report(installed_map(installed), {})
        self.assertEqual(len(issues), 1)
        self.assertEqual(overrides, [])

    def test_matching_metadata_passes_without_importing_torch(self):
        with patch.object(guard, "active_distributions", return_value=installed_map(good_distributions())):
            self.assertEqual(guard.metadata_check()["status"], "passed")

    def test_requested_extras_propagate_through_dependencies(self):
        installed = [distribution("torch", "1", ["cuda-toolkit[cublas]==2"]),
                     distribution("cuda-toolkit", "2", ['nvidia-cublas-cu12==3; extra == "cublas"']),
                     distribution("nvidia-cublas-cu12", "4")]
        issues, overrides = guard.dependency_report(installed_map(installed), {})
        self.assertEqual([row["dependency"] for row in issues], ["nvidia-cublas-cu12"])
        self.assertEqual(overrides, [])

    def test_only_shipped_nccl_exception_is_reported_as_override(self):
        installed = [distribution("torch", "2.13.0+cu129", ["nvidia-nccl-cu12==2.29.7"]),
                     distribution("nvidia-nccl-cu12", "2.30.7")]
        issues, overrides = guard.dependency_report(installed_map(installed), {})
        self.assertEqual(issues, [])
        self.assertEqual(len(overrides), 1)
        installed[1].version = "2.30.8"
        issues, overrides = guard.dependency_report(installed_map(installed), {})
        self.assertEqual(len(issues), 1)
        self.assertEqual(overrides, [])


class ActiveDistributionTests(unittest.TestCase):
    """Use real filesystem metadata, including OS-style egg-info records."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.local = Path(directory.name) / "local"
        self.system = Path(directory.name) / "system"
        self.local.mkdir()
        self.system.mkdir()
        self.paths = [str(self.local), str(self.system)]
        for dist in good_distributions():
            self.write(self.local, dist.metadata["Name"], dist.version, dist.requires)

    def write(self, root, name, version, requires=(), *, egg=False, filename=None):
        stem = filename or f"{name.replace('-', '_')}-{version}"
        record = root / (stem + (".egg-info" if egg else ".dist-info"))
        record.mkdir()
        text = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        text += "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
        (record / ("PKG-INFO" if egg else "METADATA")).write_text(text)
        return record

    def check(self):
        return guard.metadata_check(paths=self.paths)

    def test_active_local_metadata_wins_over_shadowed_os_records(self):
        self.write(self.local, "Example_Package", "2", ["six==1.17"])
        self.write(self.local, "six", "1.17")
        self.write(self.system, "example-package", "1", ["absent==1"])
        self.write(self.system, "example-package", "1", ["absent==1"], egg=True)
        self.write(self.system, "six", "1.16", egg=True)
        self.assertEqual(self.check()["status"], "passed")

    def test_wrong_active_torch_is_not_hidden_by_correct_shadow(self):
        record = self.local / "torch-2.13.0+cu129.dist-info" / "METADATA"
        record.write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.14.0\n")
        self.write(self.system, "torch", "2.13.0+cu129")
        result = self.check()
        self.assertEqual(result["versions"]["torch"], "2.14.0")
        self.assertEqual(result["status"], "failed")
        self.assertEqual({row["package"] for row in result["dependency_issues"]}, {"torchvision", "vllm"})

    def test_wrong_active_nccl_cannot_use_shadowed_exception(self):
        record = self.local / "torch-2.13.0+cu129.dist-info" / "METADATA"
        record.write_text(record.read_text() + "Requires-Dist: nvidia-nccl-cu12==2.29.7\n")
        self.write(self.local, "nvidia-nccl-cu12", "2.30.8")
        self.write(self.system, "nvidia-nccl-cu12", "2.30.7")
        result = self.check()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["declared_dependency_overrides"], [])
        self.assertEqual(result["dependency_issues"][0]["installed"], "2.30.8")

    def test_multiple_winning_root_records_are_ambiguous_even_at_equal_versions(self):
        for name, version in (("same-version", "1"), ("different-version", "2")):
            with self.subTest(name=name):
                self.write(self.local, name, "1")
                self.write(self.local, name, version, egg=True)
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    self.check()
                for record in self.local.glob(name.replace("-", "_") + "-*"):
                    for file in record.iterdir():
                        file.unlink()
                    record.rmdir()

    def test_only_selected_owners_activate_requirements_and_extras(self):
        self.write(self.local, "owner", "1", ["toolkit[active]==1"])
        self.write(self.system, "owner", "9", ["toolkit[shadow]==9"])
        self.write(self.local, "toolkit", "1", ['component==2; extra == "active"',
                                                  'shadow-only==1; extra == "shadow"'])
        self.write(self.system, "toolkit", "9", ["shadow-only==1"])
        self.write(self.local, "component", "1")
        self.write(self.system, "component", "2")
        result = self.check()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["dependency_issues"], [{"package": "toolkit", "package_version": "1",
                         "dependency": "component", "required": "==2", "installed": "1"}])

    def test_repeated_and_symlink_roots_do_not_create_duplicate_records(self):
        alias = self.local.parent / "alias"
        alias.symlink_to(self.local, target_is_directory=True)
        self.paths = [str(self.local), str(alias), str(self.local), str(self.system)]
        self.assertEqual(self.check()["status"], "passed")

    def test_invalid_metadata_including_shadowed_records_fails(self):
        self.write(self.local, "bad", "2")
        for text in ("Name: invalid name\nVersion: 1\n", "Name: bad\nVersion: invalid\n",
                     "Name: bad\n", "Version: 1\n"):
            with self.subTest(text=text):
                record = self.system / "bad-1.dist-info"
                record.mkdir(exist_ok=True)
                (record / "METADATA").write_text(text)
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    self.check()

    def test_metadata_name_must_agree_with_name_based_lookup(self):
        self.write(self.local, "actual-name", "1", filename="different_name-1")
        with self.assertRaisesRegex(RuntimeError, "inconsistent active distribution lookup"):
            self.check()

    def test_foreign_provenance_and_discovery_errors_are_checker_failures(self):
        dist = next(guard.metadata.distributions(path=[str(self.local)]))
        with patch.object(guard.metadata, "distributions", return_value=[dist]):
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                guard.metadata_check(paths=[str(self.system)])
        with patch.object(guard.metadata, "distributions", side_effect=OSError("authored discovery error")):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(guard.main(["--metadata-check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "failed")
        self.assertIn("authored discovery error", result["error"])


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
        for help_text, expected in (("usage: main.py [-h] {serve,chat}", "passed"),
                                    ("usage: vllm [-h] {serve,chat}", "failed"),
                                    ("usage: main.py [-h] {chat}", "failed"), ("", "failed")):
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


    def test_cli_diagnostics_before_help_are_retained_and_embedded_headers_fail(self):
        diagnostic = "DEBUG platform: Explicitly selected CPU platform.\n"
        header = "usage: main.py [-h] {serve,chat}\n"
        for text, expected in ((diagnostic + header, "passed"),
                               (diagnostic + "DEBUG quoted " + header, "failed"),
                               (diagnostic + "usage: main.py.evil [-h] {serve}\n", "failed")):
            with self.subTest(text=text):
                children = [self.child('{"status":"passed"}'), self.child("No broken requirements found."),
                            self.child('{"status":"passed"}'), self.child(text)]
                with patch.object(guard, "run_child", side_effect=children):
                    result = guard.check_stack()
                self.assertEqual(result["status"], expected, result["checks"]["vllm_cli"])
                self.assertEqual(result["checks"]["vllm_cli"]["stdout"], text)

    def test_actual_module_help_is_accepted_with_python_argv0(self):
        # The pinned CLI constructs ArgumentParser without prog. Exercise that
        # Python module-launch contract independently, without importing vLLM.
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "parser_control"
            package.mkdir()
            (package / "__init__.py").touch()
            (package / "main.py").write_text(
                "import argparse\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_subparsers().add_parser('serve')\n"
                "parser.parse_args()\n")
            actual = subprocess.run([sys.executable, "-m", "parser_control.main", "--help"],
                                    cwd=directory, capture_output=True, text=True, timeout=3)
        self.assertEqual(actual.returncode, 0, actual.stderr)
        # Python 3.14 includes the interpreter and -m module in default prog.
        # The serving image stays pinned to 3.12; newer host output must not
        # weaken its guard contract or make this independent control fail.
        modern_prog = sys.version_info >= (3, 14)
        expected_prog = f"{Path(sys.executable).name} -m parser_control.main" if modern_prog else "main.py"
        self.assertTrue(actual.stdout.startswith(f"usage: {expected_prog} "), actual.stdout)
        children = [self.child('{"status":"passed"}'), self.child("No broken requirements found."),
                    self.child('{"status":"passed"}'), self.child(actual.stdout, stderr=actual.stderr)]
        with patch.object(guard, "run_child", side_effect=children):
            result = guard.check_stack()
        self.assertEqual(result["status"], "failed" if modern_prog else "passed", result["checks"]["vllm_cli"])

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
                    conflict.copy(), self.child('{"status":"passed"}'), self.child("usage: main.py {serve}")]
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
