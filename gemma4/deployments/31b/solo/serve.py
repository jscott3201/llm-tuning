"""Gemma 4 31B-it on Modal B200/B300 — SOLO deployment.

Shape this is tuned for
-----------------------
- ONE user driving a coding/agentic harness (Claude Code, Cursor, Aider)
- The harness fans out up to 5 parallel tool calls per turn
  (max_running_requests=5 — see Decisions #5)
- Goal: maximum wall-clock-per-agent-turn throughput
- 192K context — the single session can afford a large window
- All KV / compute budget concentrated on this one user
- Text-only, OpenAI-compatible /v1/chat/completions with tool calling

Memory snapshot lifecycle
-------------------------
This deployment uses Modal's GPU memory snapshot feature
(`enable_memory_snapshot=True` + `experimental_options.enable_gpu_snapshot=True`)
to skip CUDA-graph capture + JIT compilation + initial model warmup on
subsequent cold starts. Expected savings: ~6-12 min cold start → ~30-60s.

Lifecycle (per Modal's official SGLang snapshot example,
https://modal.com/docs/examples/sglang_snapshot):

  1. **First boot (no snapshot exists yet)**:
       - `@modal.enter(snap=True)` runs: spawns SGLang subprocess with
         `--enable-memory-saver --enable-weights-cpu-backup`, waits for
         /health, sends one warmup request to trigger CUDA-graph capture
         and JIT compilation, then POSTs /release_memory_occupation to
         move weights/KV to CPU.
       - Modal captures the snapshot.
       - `@modal.enter(snap=False)` runs: POSTs /resume_memory_occupation
         to move weights back to GPU, waits for /health.
       - `@modal.web_server` detects port 8000 bound; container is live.

  2. **Subsequent cold starts**: Modal restores the snapshot directly
     into the running container (SGLang process intact, weights still
     on CPU). Then `@modal.enter(snap=False)` runs the resume as above.
     The expensive parts (weight load, CUDA graph capture, JIT compile)
     are SKIPPED.

  3. **Snapshot invalidation**: any image change re-runs the full
     first-boot path. Volume changes (e.g. HF cache contents) do NOT
     invalidate the snapshot because the snapshot doesn't touch the
     Volume contents after restore — weights are already in CPU memory.

Decisions
---------
 1. Weights:              BF16 (model native). No quantization.
 2. Context window:       196_608 (192K). Per-seq KV @ 192K with FP8 KV
                          is ~8.4 GiB. 5 streams ~= 42 GiB + ~63 GiB
                          weights + Triton workspace + CUDA-graph
                          capture ~= ~118 GiB on a 180 GiB B200 at 0.9
                          mem-fraction-static.
 3. MTP:                  ON. NEXTN_STANDARD with the 0.5B
                          `gemma-4-31B-it-assistant` drafter.
 4. attention_backend:    triton (Gemma 4 head_dim=512 mandate).
 5. max_running_requests: 5 — covers heavy fanout from coding harnesses.
 6. chunked_prefill:      32_768 — large chunks for single-stream TTFT.
 7. mem_fraction_static:  0.9 — cookbook command + SGLang default.
 8. CUDA graph bs:        [1,2,3,4,5].
 9. torch.compile:        OFF (SGLang server-args: "out of maintenance").
10. tp_size:              1.
11. parsers:              gemma4 / gemma4.
12. chat template:        custom 5-patch fork.
13. kv_cache_dtype:       fp8_e5m2.
14. modality:             text-only.
15. memory snapshot:      ON — see lifecycle block above.

Access / ingress
-----------------
Modal's `@modal.web_server` publishes a public `*.modal.run` URL — the URL
printed by `modal deploy`. The SGLang server binds 0.0.0.0 so Modal's web
endpoint can route to it. No API key is baked in; serving is open by
default. To restrict access, either set an api-key env hook (see
`build_serve_cmd(api_key_env=...)` in `_common/sglang_common.py`) or use
Modal's endpoint-security options, e.g. proxy auth tokens:
https://modal.com/docs/guide/webhook-proxy-auth
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

SPEC = get("31b")
DRAFT = SPEC.draft

APP_NAME = "gemma4-31b-solo"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "gemma-4-31b-it"

# ── Tuned knobs (see the Decisions block above) ──────────────────────────
MTP_PROFILE = MTP_NEXTN_STANDARD
MAX_RUNNING_REQUESTS = 5
TARGET_RUNNING_REQUESTS = 4  # leave 1-slot headroom for burst
CONTEXT_LENGTH = 196_608
CHUNKED_PREFILL_SIZE = 32_768
MAX_PREFILL_TOKENS = 32_768
MEM_FRACTION_STATIC = 0.9
CUDA_GRAPH_BS = [1, 2, 3, 4, 5]

# ── Chat templates ───────────────────────────────────────────────────────
# Templates live at <repo>/gemma4/chat_templates/. From this file
# (deployments/31b/solo/serve.py) the repo's gemma4/ dir is parents[3].
# Dense 31B uses the custom 5-patch fork; the verbatim upstream copy is
# baked alongside it for diff inspection. Both are copied into the image
# via add_local_file(copy=True) (a non-copy add would block the later
# add_local_python_source build step).
TEMPLATE_SRC_DIR = Path(__file__).resolve().parents[3] / "chat_templates"
CUSTOM_TEMPLATE_SRC = TEMPLATE_SRC_DIR / "custom_pub_chat_template_gemma4.jinja"
UPSTREAM_TEMPLATE_SRC = TEMPLATE_SRC_DIR / "gemma4_upstream.jinja"

# Destination paths inside the container image.
TEMPLATE_DIR_IN_IMAGE = "/opt/sglang/templates"
CUSTOM_TEMPLATE_PATH = f"{TEMPLATE_DIR_IN_IMAGE}/custom_pub_chat_template_gemma4.jinja"
UPSTREAM_TEMPLATE_PATH = f"{TEMPLATE_DIR_IN_IMAGE}/gemma4_upstream.jinja"

assert CONTEXT_LENGTH <= SPEC.native_max_model_len, (
    f"CONTEXT_LENGTH {CONTEXT_LENGTH} exceeds Gemma 4's native ceiling "
    f"{SPEC.native_max_model_len}"
)


app = modal.App(APP_NAME)

# Image build order: base SGLang → bake chat templates → local python
# source LAST. Modal forbids further build steps after a non-copy
# add_local_*, so the template files use copy=True and
# add_local_python_source("_common") is the final step.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(
        CUSTOM_TEMPLATE_SRC,
        CUSTOM_TEMPLATE_PATH,
        copy=True,
    )
    .add_local_file(
        UPSTREAM_TEMPLATE_SRC,
        UPSTREAM_TEMPLATE_PATH,
        copy=True,
    )
    .add_local_python_source("_common")
)

# Generic volume name so anyone can deploy on their own Modal account.
hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)

# These models are ungated, so an HF token is OPTIONAL. If you want one
# (e.g. higher download rate limits), create a Modal secret named
# "huggingface-secret" holding HF_TOKEN and add it to the @app.cls
# `secrets=[...]` list below. See https://modal.com/docs/guide/secrets .


@app.cls(
    image=image,
    gpu=SPEC.default_gpu,  # "B200+" — opts into B300 when available
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/modal-cache/huggingface": hf_cache},
    timeout=60 * 60 * 4,
    scaledown_window=60 * 20,
    max_containers=1,
)
@modal.concurrent(
    target_inputs=TARGET_RUNNING_REQUESTS,
    max_inputs=MAX_RUNNING_REQUESTS,
)
class Serve:
    """SGLang server lifecycle wrapped for Modal Memory Snapshot.

    The @app.cls / @modal.enter(snap=True/False) pattern is what lets
    us skip ~6-12 min of cold-start work (CUDA graph capture, JIT,
    first-request warmup) on every cold start after the first.
    """

    @modal.enter(snap=True)
    def startup(self) -> None:
        """Pre-snapshot phase: bring SGLang up and warm it, then release GPU."""
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
            kv_cache_dtype="fp8_e5m2",
            attention_backend="triton",
            cuda_graph_bs=CUDA_GRAPH_BS,
            max_prefill_tokens=MAX_PREFILL_TOKENS,
            chat_template=CUSTOM_TEMPLATE_PATH,
            # Snapshot co-operation flags — see _common/health.py.
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            enable_metrics=True,
            enable_request_time_stats=True,
            log_requests=True,
            log_requests_level=1,
            skip_server_warmup=True,
            # host is intentionally left at the build_serve_cmd default of
            # 0.0.0.0 so SGLang binds the container's external interface and
            # Modal's @modal.web_server ingress can route to it (published
            # as a public *.modal.run URL). The health/warmup helpers below
            # connect over loopback (127.0.0.1) from inside the container,
            # which is their own default — no override needed.
            port=SERVE_PORT,
        )

        env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}
        self.process = subprocess.Popen(cmd, env=env)

        # Wait for SGLang to bind /health, then trigger CUDA graph
        # capture + JIT via one warmup request.
        wait_for_health(self.process, port=SERVE_PORT, label=APP_NAME, timeout_s=1800)
        send_warmup_request(model=SERVED_MODEL_NAME, port=SERVE_PORT)

        # Move weights/KV off the GPU so the snapshot captures a leaner
        # state. After restore, @modal.enter(snap=False) will resume.
        release_memory_occupation(port=SERVE_PORT)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        """Post-snapshot phase: resume GPU memory before traffic arrives."""
        # Move weights/KV back to the GPU.
        resume_memory_occupation(port=SERVE_PORT)
        # Confirm SGLang is fully back to ready before exposing to traffic.
        wait_for_health(port=SERVE_PORT, label=f"{APP_NAME}-resume", timeout_s=300)

    @modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
    def serve(self) -> None:
        """No-op — the SGLang subprocess started in `startup()` is what
        serves traffic. This decorator just tells Modal which port is
        the externally-visible endpoint (published as a *.modal.run URL).
        The SGLang server binds 0.0.0.0 so this ingress can reach it."""
        pass

    @modal.exit()
    def stop(self) -> None:
        """Best-effort cleanup of the SGLang subprocess on container exit."""
        proc = getattr(self, "process", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
