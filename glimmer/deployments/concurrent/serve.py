"""Muse Glimmer 30B NVFP4 on Modal — CONCURRENT deployment.

Shape this is tuned for
-----------------------
- MANY independent sessions sharing one GPU: a team's agents, an eval
  sweep, a batch of synthetic-data workers
- Goal: maximum aggregate tokens/sec per GPU-dollar while every session
  keeps the full 131K context and stays interactive
- Text + image input, tool calling, reasoning

Why one GPU can take this many sessions
---------------------------------------
Only 13 of Glimmer's 52 layers keep a full KV cache (the other 39 are
2048-token sliding windows), and they have just 2 KV heads. With FP8 KV
and SWA_FULL_TOKENS_RATIO=0.1 the RTX PRO 6000 holds ~2.9M tokens of KV
(B200: ~6.9M): ~22 sessions at a full 131K context, or 64 at a typical
~45K agent context. Compute sets the ceiling well before memory, so
MAX_RUNNING_REQUESTS comes from the benchmark curve in the README.

Decisions
---------
 1. Checkpoint:      nvidia/Muse-Glimmer-30B-NVFP4 (see solo/serve.py).
 2. Context:         131_072 (native) for every session.
 3. KV cache:        fp8_e4m3.
 4. Speculative:     DFlash block 8. Best aggregate throughput from 4 to 16
                     in flight and within ~11% of no-spec at 64 on the RTX
                     PRO 6000. Switch to DFLASH_OFF for batch jobs that keep
                     48+ requests in flight.
 5. Vision:          ON.
 6. Chat template:   custom fork, byte-identical unless reasoning_effort set.
 7. Snapshots:       ON (single GPU).
 8. Scaling:         one container by default. Raise MAX_CONTAINERS to let
                     Modal add GPUs when in-flight requests pass
                     TARGET_RUNNING_REQUESTS per container.

Access
------
Public ``*.modal.run`` URL, open by default; see
../../../docs/securing-endpoints.md.
"""

from __future__ import annotations

from pathlib import Path

import modal

from _common.health import (
    release_memory_occupation,
    resume_memory_occupation,
    send_warmup_request,
    wait_for_health,
)
from _common.model_registry import GPU_PROFILES, get
from _common.sglang_common import (
    DFLASH_8,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("nvfp4")

APP_NAME = "glimmer-concurrent"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "muse-glimmer"

# ── Tuned knobs ──────────────────────────────────────────────────────────
GPU = "RTX-PRO-6000"  # or "B200" — see the README's GPU table
SPECULATIVE = DFLASH_8  # DFLASH_OFF for saturated batch jobs (48+ in flight)
MAX_RUNNING_REQUESTS = 64
TARGET_RUNNING_REQUESTS = 48
MAX_CONTAINERS = 1
CONTEXT_LENGTH = 131_072
CHUNKED_PREFILL_SIZE = 16_384
SWA_FULL_TOKENS_RATIO = 0.1
CUDA_GRAPH_BS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
LANGUAGE_MODEL_ONLY = False  # True drops the vision tower (~4 GB)

# ── Chat template ────────────────────────────────────────────────────────
TEMPLATE_IN_IMAGE = "/opt/sglang/templates/custom_chat_template_muse_glimmer.jinja"
# Resolved from the repo only on the machine running `modal deploy`. Inside
# the container this script sits at /root/serve.py (no repo parents), and the
# template is already baked into the image at TEMPLATE_IN_IMAGE.
TEMPLATE_SRC = (
    Path(__file__).resolve().parents[2] / "chat_templates" / Path(TEMPLATE_IN_IMAGE).name
    if modal.is_local()
    else Path(TEMPLATE_IN_IMAGE)
)

assert CONTEXT_LENGTH <= SPEC.native_max_model_len


app = modal.App(APP_NAME)

# Build order: base SGLang -> baked template (copy=True) -> _common LAST.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(TEMPLATE_SRC, TEMPLATE_IN_IMAGE, copy=True)
    .add_local_python_source("_common")
)

hf_cache = modal.Volume.from_name("glimmer-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/modal-cache/huggingface": hf_cache},
    timeout=60 * 60 * 4,
    scaledown_window=60 * 15,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(
    max_inputs=MAX_RUNNING_REQUESTS,
    target_inputs=TARGET_RUNNING_REQUESTS,
)
class Serve:
    @modal.enter(snap=True)
    def startup(self) -> None:
        """Boot SGLang, warm it, then release GPU memory for the snapshot."""
        import subprocess

        cmd = build_serve_cmd(
            model_path=SPEC.hf_repo,
            served_model_name=SERVED_MODEL_NAME,
            revision=SPEC.hf_revision,
            context_length=CONTEXT_LENGTH,
            mem_fraction_static=GPU_PROFILES[GPU].mem_fraction_static,
            chunked_prefill_size=CHUNKED_PREFILL_SIZE,
            max_running_requests=MAX_RUNNING_REQUESTS,
            speculative_config=SPECULATIVE,
            draft_model_path=SPEC.draft.hf_repo,
            draft_revision=SPEC.draft.hf_revision,
            draft_quantization=SPEC.draft.quantization,
            kv_cache_dtype="fp8_e4m3",
            attention_backend=GPU_PROFILES[GPU].attention_backend,
            swa_full_tokens_ratio=SWA_FULL_TOKENS_RATIO,
            language_model_only=LANGUAGE_MODEL_ONLY,
            cuda_graph_bs=CUDA_GRAPH_BS,
            chat_template=TEMPLATE_IN_IMAGE,
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            port=SERVE_PORT,
        )
        self.process = subprocess.Popen(cmd)
        wait_for_health(self.process, port=SERVE_PORT, label=APP_NAME, timeout_s=1800)
        send_warmup_request(model=SERVED_MODEL_NAME, port=SERVE_PORT)
        release_memory_occupation(port=SERVE_PORT)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        """After restore (or the first boot): put weights back on the GPU."""
        resume_memory_occupation(port=SERVE_PORT)
        wait_for_health(port=SERVE_PORT, label=f"{APP_NAME}-resume", timeout_s=300)

    @modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
    def serve(self) -> None:
        """The subprocess started in startup() serves; this publishes the port."""

    @modal.exit()
    def stop(self) -> None:
        proc = getattr(self, "process", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
