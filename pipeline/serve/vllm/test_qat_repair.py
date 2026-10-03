"""Plan-only repair tests: no installers, downloads or cloud clients are run."""
from pathlib import Path
import sys
import unittest

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))
from _common import qat_repair as repair


def base_versions():
    return {"torch": "2.14.0", "cuda-toolkit": "13.0.3", **repair.RETAINED,
            "nvidia-nccl-cu13": "2.30.7", "nvidia-cublas": "13.1.1.3",
            "nvidia-nccl-cu12": "2.30.7", "cuda-bindings": "13.0.3",
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

    def test_ambient_resolver_settings_cannot_override_recipe(self):
        environment = {"UV_OVERRIDE": "/unexpected", "UV_INDEX_URL": "https://unexpected.invalid",
                       "PIP_INDEX_URL": "https://unexpected.invalid", "UV_PYTHON": "other-python",
                       "VIRTUAL_ENV": "/other", "CONDA_PREFIX": "/other", "PATH": "/usr/bin",
                       "SSL_CERT_FILE": "/certificates"}
        self.assertEqual(repair.resolver_environment(environment),
                         {"PATH": "/usr/bin", "SSL_CERT_FILE": "/certificates"})


if __name__ == "__main__":
    unittest.main()
