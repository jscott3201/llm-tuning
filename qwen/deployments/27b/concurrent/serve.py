"""Qwen3.6-27B on Modal B200/B300 — CONCURRENT deployment.

Shape this is tuned for
-----------------------
- 4-5 simultaneous agentic-coding sessions on ONE B200/B300.
- Full 262K native context per session (no YaRN extension here).
- Latency-sensitive interactive loops (not high-throughput batch).
- Heavy prefix sharing (agents share system prompts; prefix-cache hit
  rate is the dominant cost lever).

Knobs vs the SOLO deployment
----------------------------
| Knob                 | concurrent (here)        | solo                          |
|----------------------|--------------------------|-------------------------------|
| max_running_requests | 5                        | 3                             |
| chunked_prefill_size | 16K (interleaves prefill)| 32K (one user, less contention)|
| mem_fraction_static  | 0.90                     | 0.83                          |
| MTP profile          | LATENCY (linear, topk=1) | AGGRESSIVE_LINEAR (linear)    |
| torch.compile        | off                      | on (+ torchinductor volume)   |
| chat template        | qwen36_upstream.jinja    | custom_pub fork               |

Deploying this yourself
-----------------------
- App / volume / template names are generic; nothing is account-specific.
- These weights are ungated, so no Hugging Face token is required. If you
  prefer authenticated pulls (higher rate limits), create a Modal secret
  named ``huggingface-secret`` holding ``HF_TOKEN`` and add it to the
  ``secrets=`` list on ``@app.function`` (left empty here on purpose).
- Run ``modal deploy deployments/27b/concurrent/serve.py``; Modal prints
  the public ``*.modal.run`` URL that forwards to the container's port
  8000. Query it like any OpenAI-compatible endpoint:

      curl <the URL printed by modal deploy>/v1/models

- Auth is your choice: this server ships OPEN. The optional ``--api-key``
  hook in ``build_serve_cmd`` stays off unless you set the ``API_KEY`` env
  var in the container. Prefer Modal's own endpoint protection instead —
  see https://modal.com/docs/guide/webhooks#security
"""

from __future__ import annotations

from pathlib import Path

import modal

from _common.health import wait_for_health
from _common.model_registry import HF_REVISION, SPEC
from _common.sglang_common import (
    LATENCY,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

# ─────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────

# App name is descriptive only — rename freely for your own account.
APP_NAME = "qwen36-27b-concurrent"
SERVE_PORT = 8000

# MTP profile. LATENCY is the cookbook linear chain (EAGLE 3/1/4,
# eagle_topk=1). Tree-verify (eagle_topk>1) is invalid on this stack
# (trtllm_mha + page_size 64 + SPEC_V2) — it raises a hard ValueError at
# startup — so a linear-chain profile is the only runnable option here.
# AGGRESSIVE_LINEAR (EAGLE 5/1/5) is the alternative to benchmark if a
# longer chain helps aggregate throughput at c=4-5.
MTP_PROFILE = LATENCY

# Concurrency cap = max in-flight inference requests, enforced by SGLang's
# --max-running-requests so its KV-cache budget can't be over-committed.
# Mirrored onto Modal's per-container input fan-in below.
MAX_RUNNING_REQUESTS = 5

# Chunked prefill size. Smaller than max_model_len so one long prompt
# can't monopolise the GPU for 5-10s while other agents wait — the
# scheduler interleaves decode tokens between prefill chunks. The 16K
# value is the LMSYS cookbook recommendation for high-context workloads.
CHUNKED_PREFILL_SIZE = 16_384

# Parser pairing — required for OpenAI-compatible tool/reasoning output.
TOOL_CALL_PARSER = "qwen3_coder"
REASONING_PARSER = "qwen3"

# Chat template. The CONCURRENT shape uses the upstream template
# (byte-identical to Qwen/Qwen3.6-27B's built-in), baked into the image
# via copy=True. parents[3] walks concurrent -> 27b -> deployments ->
# qwen, landing on the shared chat_templates/ directory.
TEMPLATE_FILE = "qwen36_upstream.jinja"
TEMPLATE_LOCAL = str(
    Path(__file__).resolve().parents[3] / "chat_templates" / TEMPLATE_FILE
)
TEMPLATE_IN_IMAGE = f"/etc/sglang/{TEMPLATE_FILE}"


app = modal.App(APP_NAME)

# Image build order: base sglang -> chat template baked in via copy=True
# (a regular build layer) -> local python source LAST. Modal's layering
# rule requires non-copy add_local_* to be the final build step; copy=True
# converts the file addition into a normal layer, so it must come BEFORE
# add_local_python_source.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(TEMPLATE_LOCAL, TEMPLATE_IN_IMAGE, copy=True)
    .add_local_python_source("_common")
)

# Persistent volume for model weights (~54 GiB BF16) so they survive
# container restarts. Generic name, shared with the solo app for reuse.
# No torchinductor volume here: this shape does not enable torch.compile.
hf_cache = modal.Volume.from_name("qwen36-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=SPEC.gpu,  # "B200+" — picks B300 when available, bills as B200.
    volumes={
        "/modal-cache/huggingface": hf_cache,
    },
    # No secrets required: the weights are ungated. To use authenticated
    # HF pulls, create a Modal secret named "huggingface-secret" (holding
    # HF_TOKEN) and add it here, e.g. secrets=[modal.Secret.from_name(
    # "huggingface-secret")].
    timeout=60 * 60 * 2,  # 2h: long enough for cold start + long sessions.
    scaledown_window=60 * 30,  # 30m: agents are bursty, cold start is 6-12 min.
    max_containers=1,  # Single B200 — hard cap prevents runaway parallel spend.
    # No min_containers — set to 1 in production if cold-start cost is
    # worse than always-warm idle cost.
)
# Mirrors SGLang's --max-running-requests so Modal's per-container input
# fan-in matches the server's real concurrency ceiling.
@modal.concurrent(max_inputs=MAX_RUNNING_REQUESTS)
@modal.web_server(port=SERVE_PORT, startup_timeout=60 * 15)
def serve() -> None:
    """Launch SGLang and block on it.

    @modal.web_server detects the port bind (waits up to startup_timeout)
    and publishes the public *.modal.run URL that forwards to port 8000.
    We additionally poll /health over loopback before blocking on
    proc.wait() so a misconfigured launch (which crashes before binding
    the port) surfaces a clear error instead of a silent startup timeout.
    """
    import subprocess

    cmd = build_serve_cmd(
        model_path=SPEC.hf_repo,
        served_model_names=["qwen3.6-27b"],
        max_model_len=SPEC.max_model_len,
        mem_fraction_static=0.90,
        chunked_prefill_size=CHUNKED_PREFILL_SIZE,
        max_running_requests=MAX_RUNNING_REQUESTS,
        speculative_config=MTP_PROFILE,
        # BF16 KV cache (model native). FP8 KV would halve footprint but
        # SGLang warns about uncalibrated scaling factors; BF16 +
        # paged-cache eviction is the safer pick here.
        kv_cache_dtype=None,
        # B200 / B300: TRT-LLM MHA kernel beats the FlashInfer default.
        attention_backend="trtllm_mha",
        # V2 Mamba scheduler with overlap (selected purely by this
        # strategy string). page_size 64 is forced independently by
        # trtllm_mha's paged MHA (the builder default), not by this
        # scheduler.
        mamba_scheduler_strategy="extra_buffer",
        # ---- Parser & template wiring ----
        tool_call_parser=TOOL_CALL_PARSER,
        reasoning_parser=REASONING_PARSER,
        chat_template=TEMPLATE_IN_IMAGE,
        # Bind 0.0.0.0 (the builder default) so Modal's web endpoint can
        # route public traffic to the container's port.
        host="0.0.0.0",
        port=SERVE_PORT,
        revision=HF_REVISION,
    )

    proc = subprocess.Popen(cmd)
    # The probe runs inside the container, so it dials loopback even though
    # the server binds 0.0.0.0 for public ingress.
    wait_for_health(proc, timeout_s=60 * 15, port=SERVE_PORT, label=APP_NAME)
    # Block on the inference server. If it exits, the container exits and
    # Modal restarts on the next request (within scaledown_window).
    proc.wait()
