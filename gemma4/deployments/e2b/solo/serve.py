"""Gemma 4 E2B-it on Modal 1xL4 - SOLO deployment (dev/test shape).

Shape this is tuned for
-----------------------
- Solo developer testing prompts before promoting to the concurrent shape
- Single user, a few in-flight requests to handle moderate fanout
- 64K context - generous for a dev loop, well under E2B's 128K ceiling
- Cheapest viable hardware: L4 (24 GiB). E2B is the smallest family member
  (5.1B total params, ~2B effective via PLE; ~10 GiB BF16 weights), so an
  L4 has ample headroom for weights + a 64K KV allocation for one stream.

Text-only despite E2B being multimodal
---------------------------------------
E2B ships vision + audio towers, but we never send image/audio inputs, so
those encoders load inert. Served text-only.

Chat template
-------------
E2B uses the upstream E4B-family template (``gemma4_e4b_upstream.jinja``),
NOT the 31B/26B/12B custom fork. The small members default to thinking OFF,
which is what we want for a fast dev/router-style endpoint. Baked into the
image with ``add_local_file(copy=True)``; referenced by a robust path
relative to this file so a fresh checkout deploys unmodified.

Memory snapshot lifecycle
-------------------------
Kept identical to the E4B shapes (warmup -> release pre-snapshot, resume
post-restore) via ``@modal.enter(snap=True/False)``. CAVEAT: L4 is NOT
explicitly named in Modal's Memory Snapshot docs (A10 and H100 are the
documented examples). L4 is Ada Lovelace, architecturally close to the
A10/L40S the E4B scripts snapshot, so it is likely to work - but if a
snapshot fails to create on your account, fall back to a plain
``@app.function`` by removing ``enable_memory_snapshot`` +
``experimental_options`` from ``@app.cls`` and dropping the
``--enable-memory-saver`` / ``--enable-weights-cpu-backup`` flags and the
release/resume calls. See https://modal.com/docs/guide/memory-snapshot .

Access
------
Deploy with ``modal deploy serve.py``; Modal publishes a public
``*.modal.run`` URL (printed by the deploy) that fronts the SGLang
OpenAI-compatible server on port 8000.
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
from _common.model_registry import get
from _common.sglang_common import (
    MTP_NEXTN_STANDARD,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

# Smallest family member: ~5.1B total / ~2B effective via PLE.
SPEC = get("e2b")
DRAFT = SPEC.draft

APP_NAME = "gemma4-e2b-solo"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "gemma-4-e2b-it"

# ── Tuned knobs ──────────────────────────────────────────────────────────
GPU = "L4"  # SPEC.default_gpu is already "L4"; pinned here for clarity.
MTP_PROFILE = MTP_NEXTN_STANDARD
MAX_RUNNING_REQUESTS = 4
TARGET_RUNNING_REQUESTS = 3
CONTEXT_LENGTH = 64_000
CHUNKED_PREFILL_SIZE = 4_096
MAX_PREFILL_TOKENS = 8_192
MEM_FRACTION_STATIC = 0.9
CUDA_GRAPH_BS = [1, 2, 3, 4]

# Robust path to the baked-in chat template. parents[3] of this file is the
# gemma4/ project root (.../gemma4/deployments/e2b/solo/serve.py).
TEMPLATE_SRC = (
    Path(__file__).resolve().parents[3] / "chat_templates" / "gemma4_e4b_upstream.jinja"
)
# Absolute path inside the container image where the template is baked.
E2B_TEMPLATE_PATH = "/opt/sglang/templates/gemma4_e4b_upstream.jinja"

assert CONTEXT_LENGTH <= SPEC.native_max_model_len, (
    f"CONTEXT_LENGTH {CONTEXT_LENGTH} exceeds E2B's native ceiling "
    f"{SPEC.native_max_model_len}"
)


app = modal.App(APP_NAME)

image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(str(TEMPLATE_SRC), E2B_TEMPLATE_PATH, copy=True)
    # add_local_python_source("_common") MUST be the last build step (see
    # sglang_common.make_sglang_image docstring).
    .add_local_python_source("_common")
)

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)

# Optional: these models are ungated, so an HF token is not required. If you
# hit HF rate limits or want authenticated pulls, create a Modal secret named
# "huggingface-secret" (user-created) and add `secrets=[HF_SECRET]` to
# @app.cls below. Left off by default since it is unnecessary for E2B.
# HF_SECRET = modal.Secret.from_name("huggingface-secret")


@app.cls(
    image=image,
    gpu=GPU,
    # Memory snapshot: see module docstring CAVEAT - L4 is undocumented for
    # snapshots. Remove these two lines (and the snap=True/False lifecycle)
    # to fall back to a plain function if snapshot creation fails on L4.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/modal-cache/huggingface": hf_cache},
    timeout=60 * 60 * 2,
    scaledown_window=60 * 20,
    max_containers=1,
)
@modal.concurrent(
    target_inputs=TARGET_RUNNING_REQUESTS,
    max_inputs=MAX_RUNNING_REQUESTS,
)
class Serve:
    @modal.enter(snap=True)
    def startup(self) -> None:
        import os
        import subprocess

        cmd = build_serve_cmd(
            model_path=SPEC.hf_repo,
            served_model_names=[SERVED_MODEL_NAME],
            max_model_len=CONTEXT_LENGTH,
            mem_fraction_static=MEM_FRACTION_STATIC,
            chunked_prefill_size=CHUNKED_PREFILL_SIZE,
            max_running_requests=MAX_RUNNING_REQUESTS,
            tp_size=1,
            revision=SPEC.hf_revision,
            speculative_config=MTP_PROFILE,
            draft_model_path=DRAFT.hf_repo,
            draft_revision=DRAFT.hf_revision,
            # FP8 E5M2 KV - halves per-seq KV at negligible quality cost.
            kv_cache_dtype="fp8_e5m2",
            # Triton is required for Gemma 4's 512-wide global head; passed
            # explicitly as defence against backend auto-detection drift.
            attention_backend="triton",
            cuda_graph_bs=CUDA_GRAPH_BS,
            max_prefill_tokens=MAX_PREFILL_TOKENS,
            chat_template=E2B_TEMPLATE_PATH,
            # Memory-snapshot co-operation (see docstring CAVEAT). Drop these
            # two if you remove enable_memory_snapshot on L4.
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            enable_metrics=True,
            enable_request_time_stats=True,
            log_requests=True,
            log_requests_level=1,
            skip_server_warmup=True,
            # Auth is the operator's choice: no --api-key baked in. To require
            # a key, pass api_key_env="SGLANG_API_KEY" here and inject the
            # value via a modal.Secret, or use Modal endpoint security
            # (proxy auth tokens): https://modal.com/docs/guide/webhook-proxy-auth
            # Bind 0.0.0.0 (NOT 127.0.0.1) so Modal's @modal.web_server can
            # reach the process across the container's external interface;
            # Modal then publishes the public *.modal.run ingress. The health
            # helpers poll /health over loopback (their own 127.0.0.1 default)
            # from inside the same container, so loopback access still works.
            host="0.0.0.0",
            port=SERVE_PORT,
        )

        # SGLANG_ENABLE_SPEC_V2=1: v0.5.12+ defaults to spec-decode V2;
        # set defensively against a future opt-in flip while MTP is on.
        env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}
        self.process = subprocess.Popen(cmd, env=env)

        wait_for_health(self.process, port=SERVE_PORT, label=APP_NAME, timeout_s=900)
        send_warmup_request(model=SERVED_MODEL_NAME, port=SERVE_PORT)
        release_memory_occupation(port=SERVE_PORT)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        resume_memory_occupation(port=SERVE_PORT)
        wait_for_health(port=SERVE_PORT, label=f"{APP_NAME}-resume", timeout_s=180)

    @modal.web_server(port=SERVE_PORT, startup_timeout=60 * 15)
    def serve(self) -> None:
        pass

    @modal.exit()
    def stop(self) -> None:
        proc = getattr(self, "process", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
