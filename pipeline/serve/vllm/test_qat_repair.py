"""Offline repair tests: no installers, downloads or cloud clients are run."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from _common import qat_repair as repair, qat_stack as guard


def base_versions():
    return {"torch": "2.14.0", "cuda-toolkit": "13.0.3", **repair.RETAINED,
            "nvidia-nccl-cu13": "2.30.7", "nvidia-cublas": "13.1.1.3",
            "nvidia-nccl-cu12": "2.30.7", "cuda-bindings": "13.0.3", "cuda-python": "13.4.1",
            "cuda-pathfinder": "1.3.1", "optional-native": "7.1"}


class RepairTests(unittest.TestCase):
    def test_removals_precede_forced_restore_using_same_system_interpreter(self):
        remove, install = repair.repair_plan(base_versions(), "/override", "/constraints")
        self.assertEqual(remove, ["uv", "--no-config", "pip", "uninstall", "--python",
                                  "/usr/bin/python3.12", "nvidia-cublas", "nvidia-nccl-cu13"])
        self.assertEqual(install[:6], ["uv", "--no-config", "pip", "install", "--python", "/usr/bin/python3.12"])
        restored = {install[index+1] for index, word in enumerate(install) if word == "--reinstall-package"}
        self.assertEqual(restored, set(repair.RESTORE_NATIVE))
        self.assertEqual(len(restored), 15)
        self.assertIn("nvidia-nccl-cu12", restored)
        self.assertTrue(install[-1].endswith("#sha256=df28741fcd89e3da7cce2d48cbe5299d6732d510ac20f5d422d0b85edf18c327"))
        self.assertNotIn("--no-deps", install)
        self.assertNotIn("--upgrade", install)

    def test_unknown_base_or_owned_version_fails_before_any_install(self):
        for name, value in (("torch", "2.15.0"), ("cuda-toolkit", "13.1"),
                            ("vllm", "0.31.0"), ("nvidia-nccl-cu13", "2.31.0")):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                repair.repair_plan({**base_versions(), name: value}, "/override", "/constraints")

    def test_equivalent_toolkit_release_versions_preserve_plan_and_constraints(self):
        expected = repair.repair_plan(base_versions(), "/override", "/constraints")
        constraints = repair.constraints_text(base_versions())
        for value in ("13.0.3.0", "13.0.3.0.0"):
            with self.subTest(value=value):
                versions = {**base_versions(), "cuda-toolkit": value}
                self.assertEqual(repair.repair_plan(versions, "/override", "/constraints"), expected)
                self.assertEqual(repair.constraints_text(versions), constraints)

    def test_toolkit_other_releases_suffixes_invalid_and_missing_fail(self):
        for value in ("13.0.4", "13.0.3.post1", "13.0.3.dev1", "13.0.3rc1", "13.0.3+local", "invalid", "", None):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "unrecognized base"):
                repair.repair_plan({**base_versions(), "cuda-toolkit": value}, "/override", "/constraints")
        versions = {name: value for name, value in base_versions().items() if name != "cuda-toolkit"}
        with self.assertRaisesRegex(RuntimeError, "unrecognized base"):
            repair.repair_plan(versions, "/override", "/constraints")

    def test_audio_cuda_build_is_accepted_and_preserved(self):
        versions = {**base_versions(), "torchaudio": "2.11.0+cu129"}
        repair.repair_plan(versions, "/override", "/constraints")
        self.assertIn("torchaudio==2.11.0+cu129\n", repair.constraints_text(versions))
        with self.assertRaisesRegex(RuntimeError, "torchaudio"):
            repair.repair_plan({**versions, "torchaudio": "2.11.0+cu130"}, "/override", "/constraints")

    def test_absent_cuda13_components_are_not_uninstalled(self):
        versions = {name: value for name, value in base_versions().items() if name not in repair.OLD_NATIVE}
        plan = repair.repair_plan(versions, "/override", "/constraints")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][3], "install")

    def test_known_optional_lmcache_is_removed_before_restore_without_pruning_dependencies(self):
        versions = {**base_versions(), "lmcache": "0.5.5", "shared-cache-dependency": "7.1"}
        remove, install = repair.repair_plan(versions, "/override", "/constraints")
        self.assertEqual(remove[6:], ["lmcache", "nvidia-cublas", "nvidia-nccl-cu13"])
        self.assertEqual(install, repair.repair_plan(base_versions(), "/override", "/constraints")[-1])
        constraints = repair.constraints_text(versions)
        self.assertNotIn("lmcache==", constraints)
        self.assertIn("shared-cache-dependency==7.1\n", constraints)
        self.assertIn("optional-native==7.1\n", constraints)
        self.assertEqual(constraints, repair.constraints_text({name: version for name, version in versions.items()
                                                              if name != "lmcache"}))

    def test_unknown_optional_lmcache_versions_are_rejected(self):
        for version in ("0.5.4", "0.5.6", "0.5.5+local", None):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "unrecognized optional LMCache"):
                repair.repair_plan({**base_versions(), "lmcache": version}, "/override", "/constraints")

    def test_constraints_pin_owned_changes_and_preserve_unrelated_packages(self):
        constraints = repair.constraints_text(base_versions())
        self.assertIn("cuda-bindings==12.9.4\n", constraints)
        self.assertIn("cuda-toolkit==12.9.1\n", constraints)
        self.assertIn("triton==3.7.1\n", constraints)
        self.assertIn("nvidia-nccl-cu12==2.30.7\n", constraints)
        self.assertIn("optional-native==7.1\n", constraints)
        self.assertIn("cuda-pathfinder==1.3.1\n", constraints)
        self.assertNotIn("torch==", constraints)
        self.assertNotIn("nvidia-nccl-cu13", constraints)
        self.assertNotIn("nvidia-cublas==", constraints)
        self.assertEqual(constraints.splitlines(), sorted(constraints.splitlines()))

    def test_cuda_python_is_an_explicit_target_paired_with_its_bindings(self):
        install = repair.repair_plan(base_versions(), "/override", "/constraints")[-1]
        self.assertIn("cuda-python==12.9.4", install)
        constraints = repair.constraints_text(base_versions())
        self.assertIn("cuda-python==12.9.4\n", constraints)
        self.assertIn("cuda-bindings==12.9.4\n", constraints)
        self.assertNotIn("cuda-python==13.4.1\n", constraints)
        self.assertEqual(install[-1], repair.TORCH_URL)

    def test_cuda_python_dependency_closure_rejects_old_pair_and_accepts_restored_pair(self):
        installed = {
            "cuda-python": SimpleNamespace(version="13.4.1", requires=["cuda-bindings~=13.4.1"]),
            "cuda-bindings": SimpleNamespace(version="12.9.4", requires=[]),
        }
        issues, overrides = guard.dependency_report(installed, {})
        self.assertEqual([row["dependency"] for row in issues], ["cuda-bindings"])
        self.assertEqual(overrides, [])
        installed["cuda-python"] = SimpleNamespace(version="12.9.4", requires=["cuda-bindings~=12.9.4"])
        self.assertEqual(guard.dependency_report(installed, {}), ([], []))

    def test_ambient_resolver_settings_cannot_override_recipe(self):
        environment = {"UV_OVERRIDE": "/unexpected", "UV_INDEX_URL": "https://unexpected.invalid",
                       "PIP_INDEX_URL": "https://unexpected.invalid", "UV_PYTHON": "other-python",
                       "VIRTUAL_ENV": "/other", "CONDA_PREFIX": "/other", "PATH": "/usr/bin",
                       "SSL_CERT_FILE": "/certificates"}
        self.assertEqual(repair.resolver_environment(environment),
                         {"PATH": "/usr/bin", "SSL_CERT_FILE": "/certificates"})


class RepairEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local, self.system = self.root / "local", self.root / "system"
        self.local.mkdir()
        self.system.mkdir()
        for name, version in base_versions().items():
            self.write(self.local, name, version)
        self.write(self.local, "six", "1.17.0")
        self.write(self.system, "six", "1.16.0")
        self.write(self.system, "six", "1.16.0", egg=True)
        self.write(self.system, "torch", "2.13.0+cu129")

    def write(self, root, name, version, *, egg=False):
        directory = root / (f"{name.replace('-', '_')}-{version}" + (".egg-info" if egg else ".dist-info"))
        directory.mkdir()
        (directory / ("PKG-INFO" if egg else "METADATA")).write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")

    def call_main(self, run):
        runtime = SimpleNamespace(executable=repair.PYTHON, version_info=(3, 12), prefix="system", base_prefix="system")
        with patch.object(sys, "path", [str(self.local), str(self.system), *sys.path]), \
                patch.object(repair, "sys", runtime), patch.object(repair.subprocess, "run", side_effect=run):
            repair.main()

    def test_main_uses_active_versions_for_plan_and_constraints(self):
        self.write(self.local, "lmcache", "0.5.5")
        self.write(self.local, "shared-cache-dependency", "7.1")
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 900)
            if "--constraint" in argv:
                text = Path(argv[argv.index("--constraint") + 1]).read_text()
                self.assertIn("six==1.17.0\n", text)
                self.assertNotIn("six==1.16.0\n", text)
                self.assertIn("optional-native==7.1\n", text)
                self.assertIn("cuda-toolkit==12.9.1\n", text)
                self.assertIn("cuda-python==12.9.4\n", text)
                self.assertNotIn("cuda-python==13.4.1\n", text)
                self.assertNotIn("lmcache==", text)
                self.assertIn("shared-cache-dependency==7.1\n", text)
                self.assertIn("cuda-python==12.9.4", argv)
                self.assertEqual(argv, repair.repair_plan(base_versions(),
                    argv[argv.index("--override") + 1], argv[argv.index("--constraint") + 1])[1])
        self.call_main(run)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][6:], ["lmcache", "nvidia-cublas", "nvidia-nccl-cu13"])

    def test_unknown_lmcache_fails_before_first_installer_call(self):
        self.write(self.local, "lmcache", "0.5.6")
        calls = []
        with self.assertRaisesRegex(RuntimeError, "unrecognized optional LMCache"):
            self.call_main(lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])

    def test_winning_root_ambiguity_fails_before_subprocess_mutation(self):
        self.write(self.local, "six", "1.18.0")
        calls = []
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            self.call_main(lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])

    def test_unrecognized_toolkit_fails_before_subprocess_mutation(self):
        record = self.local / "cuda_toolkit-13.0.3.dist-info" / "METADATA"
        record.write_text("Metadata-Version: 2.1\nName: cuda-toolkit\nVersion: 13.0.3.post1\n")
        calls = []
        with self.assertRaisesRegex(RuntimeError, "unrecognized base"):
            self.call_main(lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])

    def test_standalone_script_imports_sibling_and_uses_same_active_constraints(self):
        # Copy the two files exactly as the image does; run the actual entrypoint.
        scripts = self.root / "opt-qat"
        scripts.mkdir()
        for name in ("qat_repair.py", "qat_stack.py"):
            shutil.copyfile(PIPELINE / "_common" / name, scripts / name)
        hooks = self.root / "hooks"
        hooks.mkdir()
        (hooks / "sitecustomize.py").write_text(
            "import sys, subprocess, json\nfrom pathlib import Path\n"
            "sys.executable = '/usr/bin/python3.12'\nsys.version_info = (3, 12)\n"
            "sys.prefix = sys.base_prefix\nsys.modules['torch'] = None\nsys.modules['modal'] = None\n"
            "def run(argv, **kwargs):\n"
            "    assert argv[:3] == ['uv', '--no-config', 'pip']\n"
            "    assert kwargs['check'] is True and kwargs['timeout'] == 900\n"
            "    text = Path(argv[argv.index('--constraint') + 1]).read_text() if '--constraint' in argv else None\n"
            "    print(json.dumps({'command': argv, 'constraints': text}))\n"
            "subprocess.run = run\n")
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (self.local, self.system, hooks)))}
        result = subprocess.run([sys.executable, str(scripts / "qat_repair.py")], env=environment,
                                cwd=self.root, text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["command"], repair.repair_plan(base_versions(), "/unused", "/unused")[0])
        self.assertIn("six==1.17.0\n", rows[1]["constraints"])
        self.assertNotIn("six==1.16.0\n", rows[1]["constraints"])
        self.assertEqual(rows[1]["command"][-1], repair.TORCH_URL)


if __name__ == "__main__":
    unittest.main()
