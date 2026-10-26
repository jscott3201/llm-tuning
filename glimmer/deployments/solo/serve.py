"""Muse Glimmer 30B NVFP4 on Modal — SOLO deployment.

Shape this is tuned for
-----------------------
- ONE user driving an agent harness (coding agent, tool-using assistant)
- The harness fans out parallel tool calls and sub-agents, so a handful
  of requests can be in flight at once (MAX_RUNNING_REQUESTS)
- Goal: the fastest possible stream for each of those few requests
- Full native 131K context, text + image input, tool calling, reasoning

Decisions
---------
 1. Checkpoint:      nvidia/Muse-Glimmer-30B-NVFP4 (ModelOpt mixed
                     NVFP4/FP8/BF16, ~24.7 GB, vision tower included).
 2. GPU:             see GPU below and the README's GPU table.
 3. Context:         131_072 (native). The KV cache is tiny for this model
                     (13 of 52 layers are global, 2 KV heads), so full
                     context costs nothing worth saving.
 4. KV cache:        fp8_e4m3. Halves KV bytes; no measurable quality cost
                     in the SGLang cookbook's GSM8K A/B.
 5. Speculative:     DFlash, block 16 (the drafter's trained block). Draft
                     linears quantized to FP8 at load time.
 6. Vision:          ON. Image requests go through the perception encoder;
                     text-only clients pay ~4 GB of memory and nothing else.
 7. Chat template:   custom fork (chat_templates/), byte-identical to
                     upstream unless the request sets reasoning_effort.
 8. Snapshots:       ON. Single GPU, so Modal GPU memory snapshots skip
                     weight load, kernel autotuning, and CUDA-graph capture:
                     ~5 min first boot, ~8 s restores once Modal has a
                     snapshot for the worker type (2-3 boots after deploy).

Access
------
``modal deploy`` prints a public ``*.modal.run`` URL. It is open by
default; see ../../../docs/securing-endpoints.md for auth options. Setting
an ``API_KEY`` env var in the container (via a modal.Secret) turns on
SGLang's bearer-token check.
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
    DFLASH_16,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("nvfp4")

APP_NAME = "glimmer-solo"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "muse-glimmer"

# ── Tuned knobs ──────────────────────────────────────────────────────────
GPU = "RTX-PRO-6000"  # or "B200" — see the README's GPU table
SPECULATIVE = DFLASH_16
MAX_RUNNING_REQUESTS = 8
TARGET_RUNNING_REQUESTS = 6
CONTEXT_LENGTH = 131_072
CHUNKED_PREFILL_SIZE = 16_384
SWA_FULL_TOKENS_RATIO = 0.1
CUDA_GRAPH_BS = [1, 2, 3, 4, 5, 6, 7, 8]
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
    max_containers=1,
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
