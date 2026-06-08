"""Qwen3.6-27B on Modal B200/B300 — SOLO deployment.

Shape this is tuned for
-----------------------
- ONE user driving a coding agent.
- The agent fans out up to 3 parallel tool calls per turn
  (``max_running_requests=3`` — see Decisions below).
- Goal: maximum wall-clock-per-agent-turn throughput, not strict
  single-stream throughput.
- Full precision: BF16 weights (Qwen's native) + BF16 KV cache.
- All KV / compute / speculative budget concentrated on this one session.
- OpenAI-compatible ``/v1/chat/completions`` for an MCP-aware agent harness.

Deploying this yourself
-----------------------
- App / volume / template names are generic; nothing is account-specific.
- These weights are ungated, so no Hugging Face token is required. If you
  prefer authenticated pulls (higher rate limits), create a Modal secret
  named ``huggingface-secret`` holding ``HF_TOKEN`` and add it to the
  ``secrets=`` list on ``@app.function`` (left empty here on purpose).
- Run ``modal deploy deployments/27b/solo/serve.py``; Modal prints the
  public ``*.modal.run`` URL that forwards to the container's port 8000.
- Auth is your choice: this server ships OPEN. The optional ``--api-key``
  hook in ``build_serve_cmd`` stays off unless you set the ``API_KEY`` env
  var in the container. Prefer Modal's own endpoint protection instead —
  see https://modal.com/docs/guide/webhooks#security

Decisions
---------
 1. Weights:               BF16 (model's native; no quantization).
 2. Context window:        262_144 native (no YaRN extension; BF16 KV at
                           ~1M doesn't fit on a single B200).
 3. MTP:                   AGGRESSIVE_LINEAR — EAGLE, num_steps=5, topk=1,
                           num_draft_tokens=5 (linear chain). A longer
                           linear chain pays off at bs=1-3 because target
                           verification is essentially free on Blackwell
                           at small bs. The upstream cookbook LATENCY
                           profile is EAGLE 3/1/4; benchmark at
                           concurrency=1,2,3 to confirm the longer chain
                           actually wins before treating it as final.
 4. torch.compile:         on, ``--torch-compile-max-bs 2``. Fuses kernels
                           for the decode shapes we actually use. Adds
                           1-3 min to the first cold start; cached after
                           (see the torchinductor volume below).
 5. max_running_requests:  3 — supports parallel tool-call fanout inside
                           a single agent turn.
 6. Chunked prefill:       32_768 tokens / chunk. Balanced TTFT for
                           parallel tool calls vs scheduler overhead.
 7. mem_fraction_static:   0.83 — this is (model weights + KV-cache pool)
                           / TOTAL GPU memory. The remaining (1 - 0.83) is
                           headroom for activations + CUDA-graph buffers +
                           torch.compile JIT scratch. It is NOT a
                           "KV-pool only" knob. Tune from the boot-log
                           Allocated/Reserved memory, not a heuristic.
 8. CUDA graph bs:         [1, 2, 3] explicit. Matches max_running.
 9. continuous-decode:     2 steps per scheduler tick.
10. mixed-chunk:           left False. SGLang auto-disables it when EAGLE
                           MTP is on (sglang #5886), so setting it True is
                           a no-op; left False to avoid log noise.
11. max_prefill_tokens:    32_768 — aligned with chunked_prefill_size.
12. attention_backend:     trtllm_mha (Blackwell default at topk=1).
                           NEVER set ``--linear-attn-decode-backend
                           cutedsl`` on B200 with Qwen3.5/3.6 — broken per
                           sglang #22472 (gibberish output, no error
                           surface).
13. mamba scheduler:       extra_buffer + page_size=64. V1 vs V2 is
                           selected purely by this strategy string
                           (extra_buffer => V2); there is NO silent
                           fallback to V1. page_size 64 is forced
                           independently by trtllm_mha's paged MHA, NOT by
                           extra_buffer. FLA only needs page_size and
                           chunk-size to be mutually divisible (1/16/32/64
                           all valid); a non-divisor page size hard-errors.
14. tool-call parser:      qwen3_coder. XML wire format with schema-driven
                           type coercion. Maps to OpenAI
                           tool_calls[].function.arguments as a JSON string.
15. reasoning parser:      qwen3. Surfaces <think>...</think> as
                           delta.reasoning_content (an SGLang extension,
                           NOT OpenAI-standard).
16. chat template:         The custom public fork
                           ``custom_pub_chat_template_qwen36.jinja``, tuned
                           for open-source agentic coding harnesses. Baked
                           into the image via add_local_file(copy=True).
                           The template defaults to enable_thinking=true,
                           preserve_thinking=true; clients override
                           per-request via
                           extra_body.chat_template_kwargs.
17. SGLANG_ENABLE_SPEC_V2: env var set to 1. Spec V2 is already the
                           default since v0.5.11, so this is a defensive
                           set (NOT required); recent cookbook examples set
                           it explicitly too.

Knobs explicitly NOT set
------------------------
- ``--default-chat-template-kwargs``: a vLLM CLI flag, NOT SGLang. SGLang
  has no server-wide default for chat-template kwargs. The template's own
  Jinja defaults are the fallback; per-request overrides go through
  extra_body.chat_template_kwargs.
- ``--enable-single-batch-overlap`` / ``--enable-two-batch-overlap``: tp>1
  features for overlapping all-reduce with compute. Single B200, no-op.
- ``--linear-attn-decode-backend cutedsl``: gibberish on B200 + Qwen3.5/3.6
  (sglang #22472). Default Triton is correct.
- ``--kv-cache-dtype fp8_e4m3``: BF16 KV cache fits comfortably; keep the
  model's native precision end-to-end.
- ``--enable-fp32-lm-head``: kept OFF. No demonstrated quality need, and it
  saves the ~2.5 GB FP32 LM-head weight cache.
- ``--enable-mixed-chunk``: auto-disabled with EAGLE MTP, so passing True
  is silently-ignored noise.

Streaming + multi-tool guardrails
---------------------------------
sglang #18102, #9654, and #18001 document that EAGLE/NEXTN + streaming +
multi-tool fanout + tool_choice="auto" can produce only-first-tool or
tool-calls-in-content. #24293 (filed against Qwen3.6-27B + qwen3_coder)
confirms the parser emits an inter-tool "\n" content delta between
consecutive tool_call deltas in streaming mode, which crashes some
downstream adapters. Mitigate client-side by either (a) preferring
non-streaming for multi-tool fanout, or (b) filtering whitespace-only
content deltas while in tool-call state.
"""

from __future__ import annotations

from pathlib import Path

import modal

from _common.model_registry import HF_REVISION, SPEC
from _common.sglang_common import (
    AGGRESSIVE_LINEAR,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

# App name is descriptive only — rename freely for your own account.
APP_NAME = "qwen36-27b-solo"
SERVE_PORT = 8000

# ----------------------------------------------------------------------
# Tuned knobs (see the module docstring for the rationale behind each).
# ----------------------------------------------------------------------
# AGGRESSIVE_LINEAR is EAGLE 5/1/5 (linear, eagle_topk=1). 5/1/5 vs the
# cookbook LATENCY 3/1/4 has not been validated for every workload — run
# the bench at concurrency=1,2,3 to confirm the longer chain wins before
# treating it as final.
MTP_PROFILE = AGGRESSIVE_LINEAR
MAX_RUNNING_REQUESTS = 3
CHUNKED_PREFILL_SIZE = 32_768
MAX_PREFILL_TOKENS = 32_768
MEM_FRACTION_STATIC = 0.83
CUDA_GRAPH_BS = [1, 2, 3]
NUM_CONTINUOUS_DECODE_STEPS = 2
TORCH_COMPILE_MAX_BS = 2

# page_size 64 is forced by trtllm_mha's paged MHA (not by the Mamba
# extra_buffer strategy). FLA only requires page_size and chunk-size to be
# mutually divisible (1/16/32/64 all valid); a non-divisor page size
# hard-errors. FLA_CHUNK_SIZE is 64, so 64 satisfies both.
MAMBA_PAGE_SIZE = 64

# Parser pairing — required for OpenAI-compatible tool/reasoning output.
# Verified against SGLang server_arguments.md (--tool-call-parser and
# --reasoning-parser choice lists).
TOOL_CALL_PARSER = "qwen3_coder"
REASONING_PARSER = "qwen3"

# Chat template. The SOLO shape uses the custom public fork tuned for
# agentic coding harnesses; it is baked into the image via copy=True so
# add_local_python_source can come AFTER it as another build step.
# parents[3] walks solo -> 27b -> deployments -> qwen, landing on the
# shared chat_templates/ directory.
TEMPLATE_FILE = "custom_pub_chat_template_qwen36.jinja"
TEMPLATE_LOCAL = str(
    Path(__file__).resolve().parents[3] / "chat_templates" / TEMPLATE_FILE
)
TEMPLATE_IN_IMAGE = f"/etc/sglang/{TEMPLATE_FILE}"


app = modal.App(APP_NAME)

# Image build order: base sglang -> chat-template fork baked in via
# copy=True (a regular build layer) -> local python source LAST.
# Modal's layering rule requires non-copy add_local_* to be the final
# build step; copy=True converts the file addition into a normal layer,
# so it must come BEFORE add_local_python_source.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(TEMPLATE_LOCAL, TEMPLATE_IN_IMAGE, copy=True)
    .add_local_python_source("_common")
)

# Shared volumes — generic names; same names as the concurrent app so the
# weights and compiled artifacts are reused across both shapes.
#   - qwen36-hf-cache: model weights survive container restarts (~54 GiB BF16).
#   - torchinductor-cache: torch.compile artifacts persist across cold boots.
hf_cache = modal.Volume.from_name("qwen36-hf-cache", create_if_missing=True)
torchinductor_cache = modal.Volume.from_name(
    "torchinductor-cache", create_if_missing=True
)


@app.function(
    image=image,
    gpu=SPEC.gpu,  # "B200+" — picks B300 when available, bills as B200.
    volumes={
        "/modal-cache/huggingface": hf_cache,
        "/modal-cache/torchinductor": torchinductor_cache,
    },
    # No secrets required: the weights are ungated. To use authenticated
    # HF pulls, create a Modal secret named "huggingface-secret" (holding
    # HF_TOKEN) and add it here, e.g. secrets=[modal.Secret.from_name(
    # "huggingface-secret")].
    timeout=60 * 60 * 4,  # 4h — solo sessions can be very long.
    scaledown_window=60 * 20,  # 20m — solo workloads are more intentional.
    max_containers=1,  # Single B200 — hard cap prevents runaway parallel spend.
)
# Mirrors SGLang's --max-running-requests so Modal's per-container input
# fan-in matches the server's real concurrency ceiling.
@modal.concurrent(max_inputs=MAX_RUNNING_REQUESTS)
@modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
def serve() -> None:
    """Launch SGLang as a background subprocess.

    @modal.web_server handles container lifecycle and port-bind detection
    (waits up to startup_timeout for the named port to accept
    connections), then publishes the public *.modal.run URL that forwards
    to the container's port 8000. We just spawn the subprocess and return;
    Modal keeps the container alive for the function's lifetime. See
    https://modal.com/docs/guide/webhooks#non-asgi-web-servers
    """
    import os
    import subprocess

    cmd = build_serve_cmd(
        model_path=SPEC.hf_repo,
        served_model_names=["qwen3.6-27b"],
        max_model_len=SPEC.max_model_len,
        mem_fraction_static=MEM_FRACTION_STATIC,
        chunked_prefill_size=CHUNKED_PREFILL_SIZE,
        max_running_requests=MAX_RUNNING_REQUESTS,
        speculative_config=MTP_PROFILE,
        # BF16 KV (model native). Plenty of headroom at concurrency=3.
        kv_cache_dtype=None,
        # Blackwell default at topk=1. Do NOT set
        # --linear-attn-decode-backend cutedsl (sglang #22472).
        attention_backend="trtllm_mha",
        # V2 selected by this strategy string; page_size 64 is forced by
        # trtllm_mha's paged MHA (not by extra_buffer).
        mamba_scheduler_strategy="extra_buffer",
        page_size=MAMBA_PAGE_SIZE,
        # Solo-only perf bundle.
        enable_fp32_lm_head=False,
        enable_torch_compile=True,
        torch_compile_max_bs=TORCH_COMPILE_MAX_BS,
        cuda_graph_bs=CUDA_GRAPH_BS,
        num_continuous_decode_steps=NUM_CONTINUOUS_DECODE_STEPS,
        # NOTE: enable_mixed_chunk is auto-disabled by SGLang when EAGLE
        # MTP is active (sglang #5886). Setting True is a no-op.
        enable_mixed_chunk=False,
        max_prefill_tokens=MAX_PREFILL_TOKENS,
        # ---- Parser & template wiring ----
        tool_call_parser=TOOL_CALL_PARSER,
        reasoning_parser=REASONING_PARSER,
        chat_template=TEMPLATE_IN_IMAGE,
        # ---- Observability ----
        enable_metrics=True,
        enable_request_time_stats=True,
        log_requests=True,
        log_requests_level=1,  # metadata + sampling params (no payload).
        # SGLang's default warmup probes the vision path even for our
        # text-only workload. The vision tower loads regardless (these are
        # multimodal VLMs and SGLang has no single-flag vision skip), but
        # skipping the redundant boot probe saves a wasted boot step.
        skip_server_warmup=True,
        # Bind 0.0.0.0 (the builder default) so Modal's web endpoint can
        # route public traffic to the container's port.
        host="0.0.0.0",
        port=SERVE_PORT,
        revision=HF_REVISION,
    )

    # Defensive: explicitly opt into Speculative Decoding V2 even though
    # it's the default in v0.5.11+. Recent cookbook examples still set
    # this on the command line.
    env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}

    subprocess.Popen(cmd, env=env)
