"""Gemma 4 31B-it on Modal B200/B300 — CONCURRENT deployment.

Shape this is tuned for
-----------------------
- 4-5 simultaneous agentic sessions on ONE B200/B300
- 96K context per session
- Heavy prefix sharing — RadixAttention hit rate is the dominant cost lever
- Text-only, OpenAI-compatible /v1/chat/completions with tool calling

Memory snapshot lifecycle
-------------------------
Same Memory Snapshot pattern as the solo shape — see
`deployments/31b/solo/serve.py` module docstring for the full lifecycle
explanation (@modal.enter(snap=True) → warmup → /release_memory_occupation
→ snapshot → @modal.enter(snap=False) → /resume_memory_occupation).
Expected cold-start: ~6-12 min first deploy → ~30-60s on subsequent cold
starts.

Knobs vs the SOLO deployment
----------------------------
| Knob                  | concurrent (here)        | solo                  |
|-----------------------|--------------------------|-----------------------|
| max_running_requests  | 5                        | 5                     |
| context_length        | 98_304 (96K)             | 196_608 (192K)        |
| chunked_prefill_size  | 8_192 (interleave)       | 32_768                |
| max_prefill_tokens    | 16_384                   | 32_768                |

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

APP_NAME = "gemma4-31b-concurrent"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "gemma-4-31b-it"

# ── Tuned knobs ──────────────────────────────────────────────────────────
MTP_PROFILE = MTP_NEXTN_STANDARD
MAX_RUNNING_REQUESTS = 5
TARGET_RUNNING_REQUESTS = 4
CONTEXT_LENGTH = 98_304
CHUNKED_PREFILL_SIZE = 8_192  # multi-tenant interleave
MAX_PREFILL_TOKENS = 16_384  # SGLang default
MEM_FRACTION_STATIC = 0.9
CUDA_GRAPH_BS = [1, 2, 3, 4, 5]

# ── Chat templates ───────────────────────────────────────────────────────
# Templates live at <repo>/gemma4/chat_templates/. From this file
# (deployments/31b/concurrent/serve.py) the repo's gemma4/ dir is
# parents[3]. Dense 31B uses the custom 5-patch fork; the verbatim
# upstream copy is baked alongside it for diff inspection. Both are copied
# into the image via add_local_file(copy=True) (a non-copy add would block
# the later add_local_python_source build step).
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
            kv_cache_dtype="fp8_e5m2",
            attention_backend="triton",
            cuda_graph_bs=CUDA_GRAPH_BS,
            max_prefill_tokens=MAX_PREFILL_TOKENS,
            chat_template=CUSTOM_TEMPLATE_PATH,
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

        wait_for_health(self.process, port=SERVE_PORT, label=APP_NAME, timeout_s=1800)
        send_warmup_request(model=SERVED_MODEL_NAME, port=SERVE_PORT)
        release_memory_occupation(port=SERVE_PORT)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        resume_memory_occupation(port=SERVE_PORT)
        wait_for_health(port=SERVE_PORT, label=f"{APP_NAME}-resume", timeout_s=300)

    @modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
    def serve(self) -> None:
        """No-op — the SGLang subprocess started in `startup()` serves
        traffic. This decorator tells Modal which port is the externally-
        visible endpoint (published as a *.modal.run URL). The SGLang
        server binds 0.0.0.0 so this ingress can reach it."""
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
