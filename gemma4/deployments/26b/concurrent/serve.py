"""Gemma 4 26B-A4B-it (MoE) on Modal 2xB200 — CONCURRENT deployment.

Shape this is tuned for
-----------------------
- 4-5 simultaneous agentic sessions on 2xB200 (TP=2)
- 96K context per session
- Same multi-tenant interactive pattern as the 31B concurrent shape
- MoE all-to-all dispatch benefits from the second GPU's bandwidth
- Text-only, OpenAI-compatible /v1/chat/completions with tool calling

Why TP=2
--------
SGLang Gemma 4 cookbook recipe
(``docs.sglang.io/cookbook/autoregressive/Google/Gemma4``): the published
Gemma 4 26B-A4B command runs at ``--tp 2`` on B200. The model fits on one
B200 by VRAM (~50 GiB weights + KV well under 150 GiB at this shape) but
MoE all-to-all expert dispatch favors expert parallelism across two GPUs.
Single-B200 is unproven; benchmark before deviating from the cookbook
recipe.

Reuses the custom chat-template fork: the 31B, 26B-A4B, and 12B upstream
chat_template.jinja files are byte-identical, so the same fork applies
cleanly with no per-size adjustment (model_registry docstring +
chat_templates/README.md).

Decisions
---------
 1. Weights:              BF16 (model native). MoE 128 experts x 8 active
                          per token.
 2. Context window:       98_304 (96K). Per-seq KV at 96K with FP8 KV
                          across the 1 global layer (sliding-1024 layers
                          are window-capped, ~0.4 GiB) is ~1 GiB per
                          stream (much smaller than 31B because there is
                          only 1 global layer, vs 31B's 10). 5 streams
                          x 96K ~= 5 GiB KV.
 3. tp_size:              **2** — cookbook recipe.
 4. attention_backend:    triton (Gemma 4 family head_dim=512 mandate).
 5. max_running_requests: 5 — matches the 31B concurrent shape.
 6. chunked_prefill:      8_192 — fair multi-tenant interleave.
 7. mem_fraction_static:  0.9 — cookbook command + SGLang default.
 8. CUDA graph bs:        [1, 2, 3, 4, 5].
 9. parsers:              gemma4 / gemma4.
10. chat template:        the custom fork (same as 31B).
11. kv_cache_dtype:       fp8_e5m2 — production-validated on the family.
12. MTP:                  NEXTN_STANDARD (steps=5, draft=6, topk=1) with
                          the 26B-A4B-it-assistant drafter.

NO memory snapshot: 26B-A4B is multi-GPU (2xB200 TP=2), which Modal lists
as incompatible with GPU Memory Snapshots. So this stays on
``@app.function`` (not ``@app.cls`` with ``@modal.enter(snap=...)``) and
pays a full cold start each time. If snapshots ever support multi-GPU,
this becomes an ``@app.cls`` refactor matching the dense 31B pattern.

Deploy with ``modal deploy``; Modal then publishes a public ``*.modal.run``
URL for the web endpoint (see the URL printed by ``modal deploy``).
"""

from __future__ import annotations

from pathlib import Path

import modal

from _common.health import wait_for_health
from _common.model_registry import get
from _common.sglang_common import (
    MTP_NEXTN_STANDARD,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("26b")
DRAFT = SPEC.draft

APP_NAME = "gemma4-26b-concurrent"
SERVE_PORT = 8000

# ── Tuned knobs ──────────────────────────────────────────────────────────
MTP_PROFILE = MTP_NEXTN_STANDARD
TP_SIZE = 2  # cookbook
MAX_RUNNING_REQUESTS = 5
TARGET_RUNNING_REQUESTS = 4
CONTEXT_LENGTH = 98_304  # 96K
CHUNKED_PREFILL_SIZE = 8_192
MAX_PREFILL_TOKENS = 16_384
MEM_FRACTION_STATIC = 0.9
CUDA_GRAPH_BS = [1, 2, 3, 4, 5]

# ── Chat template ─────────────────────────────────────────────────────────
# 26B-A4B (MoE) uses the same byte-identical upstream as 31B + the same
# custom fork (model_registry docstring + chat_templates/README.md).
#
# Robust source path: from this file, parents[3] is the gemma4/ package root
# (concurrent -> 26b -> deployments -> gemma4), so the template resolves
# regardless of the working directory at deploy time.
TEMPLATE_SRC = (
    Path(__file__).resolve().parents[3]
    / "chat_templates"
    / "custom_pub_chat_template_gemma4.jinja"
)
TEMPLATE_DIR_IN_IMAGE = "/opt/sglang/templates"
CUSTOM_TEMPLATE_PATH = (
    f"{TEMPLATE_DIR_IN_IMAGE}/custom_pub_chat_template_gemma4.jinja"
)

assert CONTEXT_LENGTH <= SPEC.native_max_model_len, (
    f"CONTEXT_LENGTH {CONTEXT_LENGTH} exceeds Gemma 4 26B-A4B's native "
    f"ceiling {SPEC.native_max_model_len}"
)


app = modal.App(APP_NAME)

# Image build order: base SGLang image -> bake the chat template
# (add_local_file copy=True) -> add_local_python_source("_common") LAST.
# Modal forbids further build steps after a non-copy local-file addition, so
# add_local_python_source must come last.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(TEMPLATE_SRC, CUSTOM_TEMPLATE_PATH, copy=True)
    .add_local_python_source("_common")
)

# HF weight cache (shared with the 26B solo app — same weights). These
# models are ungated, so no token is required to pull them. Volume name is
# generic so a stranger can deploy on their own Modal account.
hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=SPEC.default_gpu,  # "B200:2" — 2-GPU spec via Modal's `:N` syntax
    volumes={"/modal-cache/huggingface": hf_cache},
    # HF token is OPTIONAL: the Gemma 4 weights are ungated. If you mirror
    # them to a gated repo, create a Modal Secret named "huggingface-secret"
    # (HF_TOKEN=...) yourself and uncomment the next line.
    # secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 2,
    scaledown_window=60 * 20,
    max_containers=1,
)
@modal.concurrent(
    target_inputs=TARGET_RUNNING_REQUESTS,
    max_inputs=MAX_RUNNING_REQUESTS,
)
@modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
def serve() -> None:
    """Spawn SGLang and block until /health is ready.

    Multi-GPU (2xB200 TP=2), so Modal Memory Snapshots do not apply — this
    function pays a full cold start (~6-12 min) each time. If snapshots ever
    support multi-GPU, this becomes an @app.cls refactor matching the dense
    31B pattern.
    """
    import os
    import subprocess

    cmd = build_serve_cmd(
        model_path=SPEC.hf_repo,
        served_model_names=["gemma-4-26b-a4b-it"],
        max_model_len=CONTEXT_LENGTH,
        mem_fraction_static=MEM_FRACTION_STATIC,
        chunked_prefill_size=CHUNKED_PREFILL_SIZE,
        max_running_requests=MAX_RUNNING_REQUESTS,
        tp_size=TP_SIZE,
        revision=SPEC.hf_revision,
        speculative_config=MTP_PROFILE,
        draft_model_path=DRAFT.hf_repo,
        draft_revision=DRAFT.hf_revision,
        kv_cache_dtype="fp8_e5m2",
        attention_backend="triton",
        cuda_graph_bs=CUDA_GRAPH_BS,
        max_prefill_tokens=MAX_PREFILL_TOKENS,
        chat_template=CUSTOM_TEMPLATE_PATH,
        enable_metrics=True,
        enable_request_time_stats=True,
        log_requests=True,
        log_requests_level=1,
        skip_server_warmup=True,
        # Bind 0.0.0.0 (NOT 127.0.0.1) so Modal's @modal.web_server ingress
        # can reach the process; Modal publishes a public *.modal.run URL.
        host="0.0.0.0",
        port=SERVE_PORT,
        # Auth is the operator's choice — no API key is baked in. To require
        # one, set api_key_env to an env-var name and inject it via a Modal
        # Secret, or use Modal's endpoint security (proxy auth tokens):
        # https://modal.com/docs/guide/webhook-proxy-auth
    )

    # v0.5.12+ defaults to spec-decode V2; set explicitly as defence against
    # a future opt-in flip when MTP is enabled.
    env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}
    proc = subprocess.Popen(cmd, env=env)

    # The server binds 0.0.0.0; this in-container health poll reaches it over
    # loopback (health.py defaults host to 127.0.0.1). Waiting for /health
    # before returning lets @modal.web_server see a working server right away.
    wait_for_health(
        proc, port=SERVE_PORT, label="gemma4-26b-concurrent", timeout_s=1800
    )
