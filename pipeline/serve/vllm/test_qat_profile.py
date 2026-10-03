"""Offline profile/command/lifecycle checks; no Modal client or image build."""
from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))

from _common.model_registry import get
from _common.vllm_common import build_serve_cmd, start_serve


def recorded_profile():
    """Record declarations independently of SDK private object layouts."""
    observed = {"image": [], "volumes": []}

    class Image:
        @classmethod
        def from_registry(cls, *args, **kwargs):
            observed["image"].append(("registry", args, kwargs))
            return cls()

        def entrypoint(self, value):
            observed["image"].append(("entrypoint", value))
            return self

        def add_local_python_source(self, value):
            observed["image"].append(("source", value))
            return self

    def decorator(name, config):
        observed[name] = config
        return lambda function: function

    class App:
        def __init__(self, name):
            observed["app"] = name

        def function(self, **kwargs):
            return decorator("function", kwargs)

    def volume(name, **kwargs):
        observed["volumes"].append((name, kwargs))
        return name

    sdk = SimpleNamespace(App=App, Image=Image,
                          Volume=SimpleNamespace(from_name=volume),
                          concurrent=lambda **kw: decorator("concurrent", kw),
                          web_server=lambda **kw: decorator("web", kw))
    with patch.dict(sys.modules, {"modal": sdk}):
        profile = runpy.run_path(str(Path(__file__).with_name("serve_31b_qat.py")))
    return profile, observed


class QatProfileTests(unittest.TestCase):
    def test_real_installed_sdk_accepts_unhydrated_profile(self):
        # Decorator and argument validation only. No app.run(), deploy or lookup.
        import modal
        from serve.vllm import serve_31b_qat
        self.assertIsInstance(serve_31b_qat.serve, modal.Function)
        self.assertFalse(serve_31b_qat.serve.is_hydrated)

    def test_image_is_immutable_without_install_or_python_injection(self):
        _, observed = recorded_profile()
        self.assertEqual(observed["image"], [
            ("registry", ("docker.io/vllm/vllm-openai:v0.30.0-cu129@sha256:"
                           "58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688",), {}),
            ("entrypoint", []), ("source", "_common")])

    def test_auth_and_resource_bounds_are_declared(self):
        _, observed = recorded_profile()
        self.assertEqual(observed["web"], {"port": 8000, "startup_timeout": 1200,
                                           "requires_proxy_auth": True})
        self.assertEqual(observed["concurrent"], {"max_inputs": 1})
        function = observed["function"]
        expected = {"gpu": "H100!", "cpu": 4, "memory": 65536, "min_containers": 0,
                    "max_containers": 1, "scaledown_window": 300, "timeout": 1800,
                    "enable_memory_snapshot": False}
        for name, value in expected.items():
            self.assertEqual(function[name], value)
        self.assertNotIn("max_inputs", function)
        self.assertNotIn("secrets", function)
        self.assertNotIn("experimental_options", function)
        self.assertEqual(observed["app"], "gemma4-31b-qat-pilot")
        self.assertEqual(observed["volumes"], [
            ("gemma4-31b-qat-pilot-hf-cache", {"create_if_missing": True}),
            ("gemma4-31b-qat-pilot-vllm-cache", {"create_if_missing": True})])

    def test_command_pins_model_tokenizer_and_text_route_without_overrides(self):
        profile, _ = recorded_profile()
        with patch.dict("os.environ", {"API_KEY": "AUTHORED_SECRET"}):
            command = profile["serve_command"]()
        self.assertEqual(command[:3], ["vllm", "serve", "google/gemma-4-31B-it-qat-w4a16-ct"])
        expected = {"--served-model-name": "gemma-4-31b-it-qat-w4a16-ct",
                    "--revision": "52f3f65bc7a02d555763bc923bd1d9094898219d",
                    "--tokenizer-revision": "52f3f65bc7a02d555763bc923bd1d9094898219d",
                    "--max-model-len": "32768", "--max-num-seqs": "1",
                    "--tensor-parallel-size": "1", "--tool-call-parser": "gemma4",
                    "--reasoning-parser": "gemma4", "--host": "0.0.0.0", "--port": "8000"}
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        for flag in ["--language-model-only", "--enable-auto-tool-choice"]:
            self.assertIn(flag, command)
        for flag in ["--api-key", "--attention-backend", "--quantization", "--dtype",
                     "--kv-cache-dtype", "--chat-template", "--speculative-config", "--enable-lora"]:
            self.assertNotIn(flag, command)
        self.assertNotIn("AUTHORED_SECRET", command)

    def test_serve_uses_owned_startup_with_declared_timeout(self):
        with patch("_common.vllm_common.start_serve") as start:
            profile, _ = recorded_profile()
            profile["serve"]()
        start.assert_called_once_with(profile["serve_command"](), timeout_s=1200,
                                      label="gemma4-31b-qat-pilot")

    def test_optional_revisions_do_not_change_legacy_command(self):
        with patch.dict("os.environ", {}, clear=True):
            command = build_serve_cmd("model", ["alias"])
        self.assertEqual(command, [
            "vllm", "serve", "model", "--served-model-name", "alias", "--host", "0.0.0.0",
            "--port", "8000", "--max-model-len", "16384", "--gpu-memory-utilization", "0.92",
            "--max-num-batched-tokens", "16384", "--reasoning-parser", "gemma4",
            "--enable-auto-tool-choice", "--tool-call-parser", "gemma4",
            "--enable-prefix-caching", "--async-scheduling"])
        self.assertIsNone(get("31b").revision)
        self.assertEqual(get("31b").gpu, "B200")
        self.assertIsNone(get("31b-qat").assistant_repo)
        self.assertFalse(get("31b-qat").requires_triton_attention)


class StartupTests(unittest.TestCase):
    def test_success_leaves_original_process_running(self):
        process = Mock()
        with patch("subprocess.Popen", return_value=process), patch("_common.vllm_common.wait_for_health") as health:
            self.assertIs(start_serve(["authored"], timeout_s=12, label="fixture"), process)
        health.assert_called_once_with(process, timeout_s=12, label="fixture")
        process.terminate.assert_not_called()
        process.wait.assert_not_called()

    def test_startup_failures_and_interruption_join_original_child(self):
        for failure in [TimeoutError("authored"), ValueError("authored"), KeyboardInterrupt()]:
            with self.subTest(failure=type(failure).__name__):
                process = Mock()
                process.poll.return_value = None
                with patch("subprocess.Popen", return_value=process), patch(
                        "_common.vllm_common.wait_for_health", side_effect=failure):
                    with self.assertRaises(type(failure)) as raised:
                        start_serve(["authored"])
                self.assertIs(raised.exception, failure)
                process.terminate.assert_called_once()
                process.wait.assert_called_once_with(timeout=10)

    def test_stubborn_child_is_killed_then_joined(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("authored", 10), -9]
        with patch("subprocess.Popen", return_value=process), patch(
                "_common.vllm_common.wait_for_health", side_effect=TimeoutError("authored")):
            with self.assertRaises(TimeoutError):
                start_serve(["authored"])
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_args_list[-1].args, ())
        self.assertEqual(process.wait.call_count, 2)

    def test_already_exited_child_is_reaped(self):
        process = Mock()
        process.poll.return_value = 7
        failure = subprocess.CalledProcessError(7, ["authored"])
        with patch("subprocess.Popen", return_value=process), patch(
                "_common.vllm_common.wait_for_health", side_effect=failure):
            with self.assertRaises(subprocess.CalledProcessError):
                start_serve(["authored"])
        process.terminate.assert_not_called()
        process.wait.assert_called_once_with(timeout=10)

    def test_real_local_child_is_joined_on_failed_readiness(self):
        original_popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            return child

        try:
            with patch("subprocess.Popen", side_effect=spawn), patch(
                    "_common.vllm_common.wait_for_health", side_effect=TimeoutError("authored")):
                with self.assertRaises(TimeoutError):
                    start_serve([sys.executable, "-c", "import time; time.sleep(60)"])
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].returncode)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    unittest.main()
