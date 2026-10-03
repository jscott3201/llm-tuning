"""Pinned, authenticated Gemma 4 31B QAT pilot; see QAT_PILOT.md.

Importing this module only declares Modal resources. Building and running it
downloads the image/model and uses a GPU; fit and runtime behavior require
separate qualification.
"""

from __future__ import annotations

import modal

from _common.model_registry import get
from _common.vllm_common import build_serve_cmd, start_serve

SPEC = get("31b-qat")
SERVED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
IMAGE = (
    "docker.io/vllm/vllm-openai:v0.30.0-cu129@"
    "sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688"
)
APP_NAME = "gemma4-31b-qat-pilot"
STARTUP_TIMEOUT = 20 * 60
INPUT_TIMEOUT = 30 * 60
IDLE_SECONDS = 300

# Modal discovers `python` on PATH. Reuse the image's system interpreter so
# its preinstalled serving packages stay in the same environment.
PYTHON_SETUP = [
    "RUN test -x /usr/bin/python3.12 && "
    "if [ ! -e /usr/local/bin/python ] && [ ! -L /usr/local/bin/python ]; then "
    "ln -s /usr/bin/python3.12 /usr/local/bin/python; fi && "
    'test "$(readlink -f /usr/local/bin/python)" = "$(readlink -f /usr/bin/python3.12)"',
    "RUN command -v pip && python -m pip --version && "
    "python -c \"import os, sys; from importlib.metadata import version; "
    "from importlib.util import find_spec; "
    "assert os.path.realpath(sys.executable) == '/usr/bin/python3.12'; "
    "assert sys.version_info[:2] == (3, 12); assert sys.prefix == sys.base_prefix; "
    "assert version('vllm').split('+', 1)[0] == '0.30.0'; "
    "print('python', sys.executable, sys.version, sys.prefix); "
    "print('vllm', version('vllm'), find_spec('vllm').origin); "
    "print('torch', version('torch'), find_spec('torch').origin)\"",
]

app = modal.App(APP_NAME)
vllm_image = (
    modal.Image.from_registry(IMAGE, setup_dockerfile_commands=PYTHON_SETUP)
    .entrypoint([])
    .add_local_python_source("_common")
)
hf_cache = modal.Volume.from_name("gemma4-31b-qat-pilot-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("gemma4-31b-qat-pilot-vllm-cache", create_if_missing=True)


def serve_command() -> list[str]:
    """Build the pinned text-only route without credentials in server argv."""
    return build_serve_cmd(
        model_path=SPEC.hf_repo,
        served_model_names=[SERVED_MODEL],
        revision=SPEC.revision,
        tokenizer_revision=SPEC.revision,
        max_model_len=SPEC.max_model_len,
        gpu_memory_utilization=SPEC.gpu_memory_utilization,
        max_num_batched_tokens=SPEC.max_num_batched_tokens,
        max_num_seqs=1,
        api_key_env=None,
        extra_args=["--tensor-parallel-size", "1", "--language-model-only"],
    )


@app.function(
    image=vllm_image,
    gpu=SPEC.gpu,
    cpu=4,
    memory=65_536,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=IDLE_SECONDS,
    timeout=INPUT_TIMEOUT,
    enable_memory_snapshot=False,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=8000, startup_timeout=STARTUP_TIMEOUT, requires_proxy_auth=True)
def serve() -> None:
    """Start the original vLLM process and wait for local readiness."""
    start_serve(serve_command(), timeout_s=STARTUP_TIMEOUT, label=APP_NAME)
