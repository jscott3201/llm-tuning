"""Gemma 4 12B-it on Modal — CONCURRENT deployment.

Shape this is tuned for
-----------------------
- Several simultaneous agentic sessions on ONE GPU
- 96K context per session
- Heavy prefix sharing — RadixAttention hit rate is the dominant cost lever
- Text-only, OpenAI-compatible /v1/chat/completions with tool calling

Model notes (Gemma 4 12B dense)
-------------------------------
- ~11.95B dense, model_type=gemma4_unified, 48 layers, hidden_size=3840,
  16 attention heads / 8 KV heads, vocab 262144, bf16 (~24 GiB weights).
- Fixed head_dim=256, hybrid 5:1 sliding-1024:global attention — the same
  hybrid scheme as the rest of Gemma 4, so the Triton attention backend is
  required.
- Because only 1/6 layers are full-attention/global, long-context KV
  growth is modest, but many concurrent 256K streams still pressure HBM.
  This shape defaults SPEC.default_gpu to H200 (141 GiB) for higher
  concurrency headroom; an H100 80 GiB handles moderate concurrency, and
  'H100:2' is an alternative for higher concurrency.

MTP / speculative decoding
--------------------------
OFF. Gemma 4 12B ships NO published MTP/speculative drafter checkpoint
(model_registry.get("12b").draft is None), so MTP_PROFILE is MTP_OFF and
no drafter args are passed. See the solo serve script for detail.

Memory snapshot lifecycle
-------------------------
Same Memory Snapshot pattern as the solo shape — see
`deployments/12b/solo/serve.py` module docstring for the full lifecycle
explanation (@modal.enter(snap=True) -> warmup -> /release_memory_occupation
-> snapshot -> @modal.enter(snap=False) -> /resume_memory_occupation).
Expected cold-start: several minutes first deploy -> ~30-60s on
subsequent cold starts.

Knobs vs the SOLO deployment
----------------------------
| Knob                  | concurrent (here)        | solo                  |
|-----------------------|--------------------------|-----------------------|
| gpu                   | H200 (141 GiB)           | H100 (80 GiB)         |
| max_running_requests  | 5                        | 5                     |
| context_length        | 98_304 (96K)             | 196_608 (192K)        |
| chunked_prefill_size  | 8_192 (interleave)       | 32_768                |
| max_prefill_tokens    | 16_384                   | 32_768                |

Ingress
-------
Under Modal, `@modal.web_server(port=8000, ...)` publishes a PUBLIC
*.modal.run URL (the URL printed by `modal deploy`). The SGLang server
binds 0.0.0.0 so Modal's web endpoint can route to it. Endpoints are
public by default; to restrict access use Modal's endpoint-security
options (proxy auth tokens) — see
https://modal.com/docs/guide/webhook-proxy-auth — or set the optional
API-key env hook below. No auth is baked in by default.
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
    MTP_OFF,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("12b")
DRAFT = SPEC.draft  # None for 12B — no published MTP drafter.

APP_NAME = "gemma4-12b-concurrent"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "gemma-4-12b-it"

# GPU override for the concurrent shape: SPEC.default_gpu is "H100" (the
# solo class). The concurrent shape wants more HBM for many simultaneous
# long-context streams, so it bumps to H200 (141 GiB). Use "H100:2" as an
# alternative for higher concurrency.
GPU = "H200"

# ── Tuned knobs ──────────────────────────────────────────────────────────
# MTP_OFF (None): 12B has no published drafter, so it runs without
# speculative decoding. No draft_model_path / draft_revision are passed.
MTP_PROFILE = MTP_OFF
MAX_RUNNING_REQUESTS = 5
TARGET_RUNNING_REQUESTS = 4
CONTEXT_LENGTH = 98_304
CHUNKED_PREFILL_SIZE = 8_192  # multi-tenant interleave
MAX_PREFILL_TOKENS = 16_384  # SGLang default
MEM_FRACTION_STATIC = 0.9
CUDA_GRAPH_BS = [1, 2, 3, 4, 5]

# ── Chat template ────────────────────────────────────────────────────────
# 12B uses the custom P1-P5 fork: recon confirms 12B's upstream
# chat_template.jinja is BYTE-FOR-BYTE IDENTICAL to the on-disk 31B
# upstream (gemma4_upstream.jinja, 17466 bytes, same SHA-256), so the
# custom fork (custom_pub_chat_template_gemma4.jinja) applies cleanly to
# 12B with no per-size adjustment. Resolved relative to the repo's
# chat_templates/ dir via parents[3]
# (deployments/12b/concurrent/serve.py -> gemma4/), then baked into the
# image with add_local_file(copy=True).
CHAT_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "chat_templates"
LOCAL_CUSTOM_TEMPLATE = CHAT_TEMPLATES_DIR / "custom_pub_chat_template_gemma4.jinja"
LOCAL_UPSTREAM_TEMPLATE = CHAT_TEMPLATES_DIR / "gemma4_upstream.jinja"

# Absolute paths the templates are baked to INSIDE the container image.
TEMPLATE_DIR_IN_IMAGE = "/opt/sglang/templates"
CUSTOM_TEMPLATE_PATH = f"{TEMPLATE_DIR_IN_IMAGE}/custom_pub_chat_template_gemma4.jinja"
UPSTREAM_TEMPLATE_PATH = f"{TEMPLATE_DIR_IN_IMAGE}/gemma4_upstream.jinja"

assert CONTEXT_LENGTH <= SPEC.native_max_model_len, (
    f"CONTEXT_LENGTH {CONTEXT_LENGTH} exceeds Gemma 4's native ceiling "
    f"{SPEC.native_max_model_len}"
)


app = modal.App(APP_NAME)

# Image build order: base SGLang -> bake chat templates -> local python
# source LAST. Modal forbids further build steps after a non-copy
# add_local_*; the templates use copy=True and add_local_python_source is
# the final step.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(
        LOCAL_CUSTOM_TEMPLATE,
        CUSTOM_TEMPLATE_PATH,
        copy=True,
    )
    .add_local_file(
        LOCAL_UPSTREAM_TEMPLATE,
        UPSTREAM_TEMPLATE_PATH,
        copy=True,
    )
    .add_local_python_source("_common")
)

# Generic HF weight cache Volume. These weights are ungated/public, so no
# HF token is required. If you deploy a gated model, create a Modal Secret
# named "huggingface-secret" (HF_TOKEN=...) and add `secrets=[...]` below.
hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU,  # "H200" (141 GiB) — concurrency headroom over the solo H100
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
            # MTP OFF for 12B (no published drafter) — speculative_config is
            # None, so build_serve_cmd emits no --speculative-* flags.
            speculative_config=MTP_PROFILE,
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
            # Optional API-key gate: pass api_key_env="<ENV_VAR>" here and
            # inject the value via a modal.Secret to require an API key.
            # Left off by default — auth is the operator's choice (see the
            # module docstring's Ingress note and Modal's proxy-auth docs).
            #
            # The SGLang process binds 0.0.0.0 so Modal's @modal.web_server
            # can bridge the public *.modal.run URL to it; the health/warmup
            # helpers below connect over 127.0.0.1.
            host="0.0.0.0",
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
        traffic. Modal publishes a public *.modal.run URL for this port
        (the URL printed by `modal deploy`)."""
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
