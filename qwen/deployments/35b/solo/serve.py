"""Qwen3.6-35B-A3B (MoE) on Modal B200/B300 — SOLO deployment.

Shape this is tuned for
-----------------------
- ONE user driving a coding agent
- Agent fans out up to 3 parallel tool calls per turn
  (max_running_requests=3 — see Decisions below)
- Goal: maximum wall-clock-per-agent-turn throughput, not strict
  single-stream throughput
- Full-precision: BF16 weights (Qwen's native), BF16 KV
- All KV / compute / speculative budget concentrated on this one user
- OpenAI-compatible /v1/chat/completions for an MCP-aware agent harness

Deploy it with ``modal deploy``; Modal prints the public ``*.modal.run``
URL that fronts the in-container SGLang server (see the ingress note
below).

Single GPU, TP=1 (the MoE serving decision)
-------------------------------------------
~71.9 GB BF16 weights (66.97 GiB on disk, 26 shards) fit one B200
(~180-183 GB usable; B300 = 288 GB) at TP=1 — the deliberate divergence
from the HF model card's `--tp-size 8`. NO `--ep-size` / `--enable-ep-moe`
/ DeepEP / `--enable-dp-attention`. There are NO expert-parallel flags at
TP=1. BF16 is the DEFAULT; FP8 weights / FP8 e4m3 KV are the long-context
/ OOM contingency only. TP>1 is a KV-headroom contingency only (and
forfeits single-GPU snapshot eligibility) — leave it at 1 unless a
benchmark shows KV pressure at the target context.

Decisions (conservative starting points; this MoE shape is UNVALIDATED)
-----------------------------------------------------------------------
 1. Weights:               BF16 (model's native; no quantization). FP8 is
                           the OOM/long-context contingency only.
 2. Context window:        262_144 native (no YaRN extension; BF16 KV
                           at 1M doesn't fit on a single B200).
 3. MTP:                   AGGRESSIVE_LINEAR — num_steps=5, topk=1,
                           num_draft_tokens=5 (linear chain). Longer
                           linear chain pays off at bs=1-3 because the
                           MoE at low batch is memory-bandwidth-bound, so
                           verifying a longer chain is nearly free.
                           topk MUST be 1: tree-verify (eagle_topk>1)
                           raises a hard ValueError on this stack
                           (trtllm_mha + SPEC_V2 + page_size 64); the
                           builder also guards it.
                           TODO(benchmark): 5/1/5 vs LATENCY 3/1/4 vs
                           4/1/5 is UNPROVEN for this MoE — A/B via
                           sglang.bench_speculative at bs=1-3 before
                           treating it as final. Watch open bugs #24863
                           (trtllm_mha + MTP CUDA IMA) and #23330
                           (adaptive step-size recapture on hybrid GDN).
 4. torch.compile:         OFF for this greenfield MoE. SGLang documents
                           it as out-of-maintenance, it is risky with
                           spec decode on the hybrid GDN backbone, and it
                           can break snapshot creation.
                           Enable only after a benchmark pass.
 5. max_running_requests:  3 — supports parallel tool-call fanout
                           inside a single agent turn.
 6. Chunked prefill:       32_768 tokens / chunk. Balanced TTFT for
                           parallel tool calls vs scheduler overhead.
                           TODO(benchmark): UNVALIDATED for this MoE;
                           feeds mem_fraction_static headroom — tune
                           together from the boot memory lines.
 7. mem_fraction_static:   0.8 — this is (model weights + KV-cache pool)
                           / TOTAL GPU memory. The remaining (1 - 0.8)
                           is headroom for activations + CUDA-graph
                           buffers + spec-decode draft buffers + the
                           loaded vision tower (SGLang auto-applies
                           adjust_mem_fraction_for_vlm). It is NOT a
                           "KV-pool only" knob, and there is NO "+10%
                           real utilization" rule. 0.8 is the SGLang
                           cookbook MoE default starting point.
                           TODO(benchmark): tune from the boot
                           Allocated/Reserved memory lines + an OOM probe
                           at 262K; the ~72 GB MoE weights are heavier
                           than the dense 27B's ~54 GB.
 8. CUDA graph bs:         [1, 2, 3] explicit. Matches max_running.
                           TODO(benchmark): with spec decode the effective
                           verify batch is bs × draft_tokens; confirm the
                           capture set covers the real range.
 9. continuous-decode:     2 steps per scheduler tick.
10. mixed-chunk:           SGLang auto-disables it when EAGLE MTP is on
                           (sglang #5886). Setting it is a no-op; left
                           False to avoid log noise.
11. max_prefill_tokens:    32_768 — aligned with chunked_prefill_size.
                           TODO(benchmark): UNVALIDATED for this MoE.
12. attention_backend:     trtllm_mha (Blackwell default at topk=1).
                           NEVER set --linear-attn-decode-backend
                           cutedsl on B200 with qwen3_5/qwen3_5_moe —
                           broken per sglang #22472 (silent gibberish).
                           Under MTP the GDN decode backend correctly
                           falls back to the Triton default; no flag.
13. mm backend:            NOT set. This is a real VLM (vision_config
                           present) so the tower loads regardless, but
                           vision is never driven by our text-only
                           clients and --mm-attention-backend fa4 is
                           unverified on the pinned image — so we omit it
                           rather than risk an unrecognized-arg failure.
14. mamba scheduler:       extra_buffer + page_size=64. V1 vs V2 is
                           selected purely by this strategy string
                           (extra_buffer => V2); there is NO silent
                           fallback to V1. page_size 64 is forced
                           independently by trtllm_mha's paged MHA, NOT
                           by extra_buffer. FLA only needs page_size and
                           chunk-size to be mutually divisible.
15. tool-call parser:      qwen3_coder. XML wire format with
                           schema-driven type coercion. Maps to OpenAI
                           tool_calls[].function.arguments as a JSON
                           string.
16. reasoning parser:      qwen3. Surfaces <think>...</think> as
                           delta.reasoning_content (SGLang extension,
                           NOT OpenAI-standard).
17. chat template:         the custom fork
                           custom_pub_chat_template_qwen36.jinja, baked
                           into the image and passed via --chat-template.
18. SGLANG_ENABLE_SPEC_V2: env var set to 1. Spec V2 is already the
                           default since v0.5.11, so this is a defensive
                           set (NOT required).

Knobs explicitly NOT set
------------------------
- --ep-size / --enable-ep-moe / DeepEP / --enable-dp-attention: there are
  NO expert-parallel flags at TP=1.
- --moe-runner-backend deep_gemm: the DeepGEMM MoE runner is OFF (default
  Triton). See the disabled block in serve(). Only enable behind a
  benchmark win.
- --enable-fp32-lm-head: kept OFF. Real decode cost is small
  (~0.3 ms / ~3-5%, memory-bandwidth-bound), so the justification is
  "no demonstrated quality need + ~2.5 GB FP32 LM-head weight cache
  saved", not a throughput cliff.
- --linear-attn-decode-backend cutedsl: gibberish on B200 + qwen3_5/
  qwen3_5_moe (sglang #22472). Default Triton is correct.
- --kv-cache-dtype fp8_e4m3: BF16 KV fits comfortably; FP8 is a
  long-context/OOM contingency only.

Streaming + multi-tool guardrails
---------------------------------
sglang #24293 (Anthropic /v1/messages emits text_delta on an open
tool_use block) crashes some downstream adapters in streaming mode. The
harness handles this by either: (a) preferring non-streaming for
multi-tool fanout, or (b) filtering whitespace-only content deltas while
in tool-call state.

Inherited-but-unvalidated dependencies
--------------------------------------
The image / argv defaults are inherited from the shared _common library
(SGLANG_TAG digest, FlashInfer / sgl-kernel versions, the `distro` shim,
env knobs). Smoke-test the exact argv on the pinned image before trusting
a clean deploy — wrong SGLang knobs fail as silent broken output, not as
an error.

Ingress
-------
The SGLang server binds 0.0.0.0:8000 inside the container so Modal's web
endpoint can reach it. @modal.web_server(port=8000, ...) publishes a
public *.modal.run URL (printed by `modal deploy`) that forwards to the
container's port 8000; that public URL is the intended ingress.
"""

from __future__ import annotations

from pathlib import Path

import modal

from _common.model_registry import HF_REVISION, get
from _common.sglang_common import (
    AGGRESSIVE_LINEAR,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("35b")

APP_NAME = "qwen36-35b-solo"  # descriptive only
SERVED_MODEL_NAME = "qwen3.6-35b-a3b"
SERVE_PORT = 8000

# ----------------------------------------------------------------------
# Tuned knobs (the decisions from the module docstring). These are
# CONSERVATIVE STARTING POINTS — this MoE shape has not been validated;
# keep the TODO(benchmark) markers until A/B'd on the real checkpoint.
# ----------------------------------------------------------------------
# AGGRESSIVE_LINEAR is EAGLE 5/1/5 (linear, eagle_topk=1). NOTE: 5/1/5
# vs the cookbook LATENCY 3/1/4 has NOT yet been validated for this MoE —
# A/B via sglang.bench_speculative at bs=1-3 to confirm the longer chain
# actually wins before treating this as final.
MTP_PROFILE = AGGRESSIVE_LINEAR
MAX_RUNNING_REQUESTS = 3
# TODO(benchmark): CHUNKED_PREFILL_SIZE / MAX_PREFILL_TOKENS are mirrored
# from the dense 27B and are UNVALIDATED for the MoE. They feed the
# mem_fraction_static headroom — tune together from the boot memory lines.
CHUNKED_PREFILL_SIZE = 32_768
MAX_PREFILL_TOKENS = 32_768
# 0.8 = SGLang cookbook MoE default starting point (weights + KV pool /
# total HBM). TODO(benchmark): tune from the boot Allocated/Reserved
# memory lines + an OOM probe at 262K context — there is no "+10%" rule.
MEM_FRACTION_STATIC = 0.8
# TODO(benchmark): CUDA_GRAPH_BS spans the running-requests range; with
# spec decode the effective verify batch is bs × draft_tokens.
CUDA_GRAPH_BS = [1, 2, 3]
NUM_CONTINUOUS_DECODE_STEPS = 2

# page_size 64 is forced by trtllm_mha's paged MHA (not by the Mamba
# extra_buffer strategy). FLA only requires page_size and chunk-size to
# be mutually divisible (1/16/32/64 all valid); a non-divisor page size
# hard-errors. FLA_CHUNK_SIZE is currently 64, so 64 satisfies both.
MAMBA_PAGE_SIZE = 64

# Chat template: the SOLO shape uses the custom fork. It lives in the repo
# at qwen/chat_templates/ and is baked into the image with copy=True so it
# survives as a regular build layer. parents[3] of this file resolves to
# the qwen/ project root (…/qwen/deployments/35b/solo/serve.py -> …/qwen).
FORK_FILENAME = "custom_pub_chat_template_qwen36.jinja"
FORK_LOCAL = Path(__file__).resolve().parents[3] / "chat_templates" / FORK_FILENAME
FORK_IN_IMAGE = f"/etc/sglang/{FORK_FILENAME}"


app = modal.App(APP_NAME)
# Image build order: base sglang → chat-template fork baked in via
# copy=True → local python source LAST.
# Modal forbids further build steps after a non-copy add_local_*, so
# add_local_python_source("_common") must be the final layer. copy=True
# converts the fork addition into a regular build layer, so the fork bake
# must come BEFORE add_local_python_source.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(FORK_LOCAL.as_posix(), FORK_IN_IMAGE, copy=True)
    .add_local_python_source("_common")
)

# HF cache Volume — generic name, shared with the concurrent app for
# cold-boot weight reuse (~71.9 GB BF16, 26 shards). No torchinductor
# volume (torch.compile is off) and no DeepGEMM volume (the DeepGEMM MoE
# runner is off — see the disabled block in serve()).
hf_cache = modal.Volume.from_name("qwen36-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=SPEC.gpu,  # "B200+" single GPU; TP=1
    volumes={
        "/modal-cache/huggingface": hf_cache,
    },
    # HF token is optional here — these weights are ungated. If you want to
    # use one (e.g. for higher rate limits), create a Modal Secret named
    # "huggingface-secret" yourself and uncomment the line below.
    # secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 4,  # 4h — solo sessions can be very long
    scaledown_window=60 * 20,  # 20m — solo workloads are more intentional
    max_containers=1,
    # NO Modal memory snapshots: TP=1 makes this MoE snapshot-ELIGIBLE,
    # but v1 keeps them OFF — cold start is weight-load-dominated
    # (snapshots don't help that) and torch.compile is off (little JIT to
    # snapshot). torch.compile, if ever enabled (decision 4), can itself
    # fail snapshot creation.
)
# Mirrors SGLang's --max-running-requests so Modal's autoscaler view of
# in-flight inputs matches the server's real concurrency cap. Capacity is
# static (max_containers=1); the hard cap is SGLang's flag.
@modal.concurrent(max_inputs=MAX_RUNNING_REQUESTS)
@modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
def serve() -> None:
    """Launch SGLang as a background subprocess.

    Modal's @modal.web_server handles container lifecycle and port-bind
    detection on its own (waits up to startup_timeout for the port to
    accept connections), then publishes the public *.modal.run URL that
    forwards to port 8000. We just spawn the subprocess and return —
    Modal keeps the container alive for the function's lifetime.
    See https://modal.com/docs/guide/webhooks#non-asgi-web-servers
    and the official Modal vLLM example for this exact pattern.
    """
    import os
    import subprocess

    cmd = build_serve_cmd(
        model_path=SPEC.hf_repo,
        served_model_names=[SERVED_MODEL_NAME],
        max_model_len=SPEC.max_model_len,
        mem_fraction_static=MEM_FRACTION_STATIC,
        chunked_prefill_size=CHUNKED_PREFILL_SIZE,
        max_running_requests=MAX_RUNNING_REQUESTS,
        speculative_config=MTP_PROFILE,
        # BF16 KV (model native). Plenty of headroom at concurrency=3;
        # only 10 of 40 layers are full-attention. FP8 e4m3 KV is the
        # long-context/OOM contingency only.
        kv_cache_dtype=None,
        # Blackwell default at topk=1. Do NOT set
        # --linear-attn-decode-backend cutedsl (sglang #22472).
        attention_backend="trtllm_mha",
        # Do NOT hardcode --mm-attention-backend fa4 (omitted): vision is
        # never driven by text-only clients and fa4 is unverified on the
        # pinned image.
        # V2 selected by this strategy string; page_size 64 is forced by
        # trtllm_mha's paged MHA (not by extra_buffer).
        mamba_scheduler_strategy="extra_buffer",
        page_size=MAMBA_PAGE_SIZE,
        # Solo-only perf bundle. torch.compile is OFF for this greenfield
        # MoE (decision 4) — it is out-of-maintenance, risky with spec
        # decode on hybrid GDN, and can fail snapshot creation.
        enable_fp32_lm_head=False,
        enable_torch_compile=False,
        cuda_graph_bs=CUDA_GRAPH_BS,
        num_continuous_decode_steps=NUM_CONTINUOUS_DECODE_STEPS,
        # NOTE: enable_mixed_chunk is auto-disabled by SGLang when EAGLE
        # MTP is active (sglang #5886). Setting True would be a no-op.
        enable_mixed_chunk=False,
        max_prefill_tokens=MAX_PREFILL_TOKENS,
        # ---- Parser & template wiring ----
        # Parsers default on in build_serve_cmd (--reasoning-parser qwen3,
        # --tool-call-parser qwen3_coder). The SOLO shape passes the baked
        # custom fork via --chat-template.
        chat_template=FORK_IN_IMAGE,
        # ---- Observability ----
        enable_metrics=True,
        enable_request_time_stats=True,
        log_requests=True,
        log_requests_level=1,  # metadata + sampling params (no payload)
        # SGLang's default warmup probes the vision path even for our
        # text-only workload. The vision tower loads regardless (this is
        # a multimodal VLM and SGLang has no single-flag vision skip),
        # but skipping the redundant boot probe saves a wasted boot step.
        skip_server_warmup=True,
        # Bind 0.0.0.0 so Modal's web endpoint (the public *.modal.run URL)
        # can reach the server; the in-container /health probe still works
        # over loopback because 0.0.0.0 accepts on every interface.
        host="0.0.0.0",
        port=SERVE_PORT,
        revision=HF_REVISION,
    )

    # Defensive: explicitly opt into Speculative Decoding V2 even though
    # it's the default in v0.5.11+. Recent cookbook examples still set
    # this on the command line.
    env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}

    # ── DeepGEMM MoE runner — DISABLED (default OFF) ──────────────────
    # DeepGEMM is NOT on the BF16 execution path unless the MoE is opted
    # in here; the default MoE runner is Triton. Only worth enabling if a
    # benchmark shows a win for this 256-expert / 3B-active MoE. To try
    # it, uncomment the flag + env vars and mount a persistent
    # SGLANG_DG_CACHE_DIR volume (JIT cache):
    #   cmd += ["--moe-runner-backend", "deep_gemm"]
    #   env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "1"
    #   env["SGLANG_DG_CACHE_DIR"] = "/modal-cache/deep_gemm"  # + Volume mount

    subprocess.Popen(cmd, env=env)
