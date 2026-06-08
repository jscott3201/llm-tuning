"""Qwen3.6-35B-A3B (MoE) on Modal B200/B300 — CONCURRENT deployment.

Shape this is tuned for
-----------------------
- 4-5 simultaneous agentic-coding sessions on ONE B200/B300, TP=1
- Full 262K native context per session (no YaRN extension here)
- Latency-sensitive interactive loops (not high-throughput batch)
- Heavy prefix sharing (agents share system prompts; prefix-cache hit
  rate is the dominant cost lever)

Deploy it with ``modal deploy``; Modal prints the public ``*.modal.run``
URL that fronts the in-container SGLang server. Query the live model with::

    $ curl https://<the URL printed by modal deploy>/v1/models

Single GPU, TP=1 (the MoE serving decision)
-------------------------------------------
~71.9 GB BF16 weights (66.97 GiB on disk, 26 shards) fit one B200
(~180-183 GB usable; B300 = 288 GB) at TP=1 — the deliberate divergence
from the HF model card's `--tp-size 8` (a generic large-cluster
illustration). NO `--ep-size` / `--enable-ep-moe` / DeepEP /
`--enable-dp-attention`: there are NO expert-parallel flags at TP=1. BF16
is the DEFAULT; FP8 weights / FP8 e4m3 KV are the long-context / OOM
contingency only, A/B'd before adoption. TP>1 is a KV-headroom
contingency only (and forfeits single-GPU snapshot eligibility) — leave it
at 1 unless a benchmark shows KV pressure.

Knobs vs the SOLO deployment
----------------------------
| Knob                       | concurrent (here)        | solo                  |
|----------------------------|--------------------------|-----------------------|
| max_running_requests       | 5                        | 3                     |
| chunked_prefill_size       | 16K (interleaves prefill)| 32K                   |
| mem_fraction_static        | 0.8                      | 0.8                   |
| MTP profile                | LATENCY (linear, topk=1) | AGGRESSIVE_LINEAR (linear, topk=1) |
| chat template              | qwen36_upstream.jinja    | custom fork           |

All of these are CONSERVATIVE STARTING POINTS — this MoE shape has not
been validated; keep the TODO(benchmark) markers until A/B'd on the real
checkpoint.

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

from _common.health import wait_for_health
from _common.model_registry import HF_REVISION, get
from _common.sglang_common import (
    LATENCY,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("35b")

# ─────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────

APP_NAME = "qwen36-35b-concurrent"  # descriptive only
SERVED_MODEL_NAME = "qwen3.6-35b-a3b"
SERVE_PORT = 8000

# MTP profile. LATENCY is the cookbook linear chain (EAGLE 3/1/4,
# eagle_topk=1). Tree-verify (eagle_topk>1) is invalid on this stack
# (trtllm_mha + page_size 64 + SPEC_V2) — it raises a hard ValueError at
# startup — so a linear-chain profile is the only runnable option here.
# AGGRESSIVE_LINEAR (EAGLE 5/1/5) is the alternative to benchmark if a
# longer chain helps aggregate throughput at c=4-5.
# TODO(benchmark): A/B LATENCY (3/1/4) vs AGGRESSIVE_LINEAR (5/1/5) vs
# 4/1/5 on the real B200 + 35B-A3B checkpoint (sglang.bench_speculative at
# bs=4-5); the MoE profile is unproven. Watch open bugs #24863
# (trtllm_mha + MTP CUDA IMA) and #23330 (adaptive step-size recapture
# on hybrid GDN) when switching profiles.
MTP_PROFILE = LATENCY

# Concurrency cap = max in-flight inference requests, enforced by
# SGLang's --max-running-requests so its KV-cache budget can't be
# over-committed. @modal.concurrent mirrors this so Modal's autoscaler
# view matches; capacity here is static (max_containers=1).
MAX_RUNNING_REQUESTS = 5

# Chunked prefill size. Smaller than max_model_len so one long prompt
# can't monopolise the GPU for 5-10s while other agents wait — the
# scheduler interleaves decode tokens between prefill chunks. The 16K
# value is the LMSYS cookbook recommendation for high-context workloads.
# TODO(benchmark): UNVALIDATED for this MoE. chunked_prefill_size feeds
# the mem_fraction_static headroom (reserved_mem grows ~1.5× the chunk
# size), so tune this together with mem_fraction from the boot
# Allocated/Reserved memory lines, not from a ported dense-27B number.
CHUNKED_PREFILL_SIZE = 16_384

# mem_fraction_static = (model weights + KV-cache pool) / TOTAL GPU
# memory. The remaining (1 - value) is headroom for activations +
# CUDA-graph buffers + spec-decode draft buffers + the loaded vision
# tower (SGLang auto-applies adjust_mem_fraction_for_vlm). It is NOT a
# "KV-pool only" knob and there is NO "+10% real utilization" rule.
# 0.8 is the SGLang cookbook MoE default starting point.
# TODO(benchmark): tune empirically from the boot "GPU Allocated/Reserved
# Memory" lines and an OOM probe at 262K context; the ~72 GB MoE weights
# are heavier than the dense 27B's ~54 GB, so revisit before trusting 0.8.
MEM_FRACTION_STATIC = 0.8

# Chat template: the CONCURRENT shape uses the upstream template
# (qwen36_upstream.jinja). It lives in the repo at qwen/chat_templates/ and
# is baked into the image with copy=True so it survives as a regular build
# layer. parents[3] of this file resolves to the qwen/ project root
# (…/qwen/deployments/35b/concurrent/serve.py -> …/qwen).
TEMPLATE_FILENAME = "qwen36_upstream.jinja"
TEMPLATE_LOCAL = Path(__file__).resolve().parents[3] / "chat_templates" / TEMPLATE_FILENAME
TEMPLATE_IN_IMAGE = f"/etc/sglang/{TEMPLATE_FILENAME}"


app = modal.App(APP_NAME)

# Image build order: base sglang → chat template baked in via copy=True →
# local python source LAST. Modal forbids further build steps after a
# non-copy add_local_*, so add_local_python_source("_common") must be the
# final layer. copy=True converts the template addition into a regular
# build layer, so it must come BEFORE add_local_python_source.
image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(TEMPLATE_LOCAL.as_posix(), TEMPLATE_IN_IMAGE, copy=True)
    .add_local_python_source("_common")
)

# Persistent volume: hf-cache holds the model weights so they survive
# container restarts (~71.9 GB BF16, 26 shards). Generic name, shared with
# the solo app for cold-boot weight reuse. No torchinductor volume
# (torch.compile is off here) and no DeepGEMM volume (the DeepGEMM MoE
# runner is off — see the disabled block in serve() below).
hf_cache = modal.Volume.from_name("qwen36-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=SPEC.gpu,  # "B200+" — picks B300 when available, bills as B200.
    volumes={
        "/modal-cache/huggingface": hf_cache,
    },
    # HF token is optional here — these weights are ungated. If you want to
    # use one (e.g. for higher rate limits), create a Modal Secret named
    # "huggingface-secret" yourself and uncomment the line below.
    # secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 2,  # 2h: long enough for cold start + long sessions
    scaledown_window=60 * 30,  # 30m: agents are bursty, cold start is long
    max_containers=1,  # Single B200 — hard cap prevents runaway parallel spend
    # No min_containers — set to 1 in production if cold-start cost is
    # worse than always-warm idle cost.
    # NO Modal memory snapshots: TP=1 makes this MoE snapshot-ELIGIBLE,
    # but we keep snapshots OFF for v1 — cold start is dominated by
    # loading ~72 GB of weights (which snapshots don't help) and the BF16
    # config has little JIT to snapshot.
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
    forwards to port 8000. We spawn the subprocess, wait for /health, then
    block on it for the container's lifetime.
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
        # BF16 KV cache (model native). Only 10 of 40 layers are
        # full-attention (the 30 GDN layers carry recurrent state, not
        # paged KV), so KV pressure is modest. FP8 e4m3 KV is the
        # long-context/OOM contingency only — not the default.
        kv_cache_dtype=None,
        # B200 / B300: TRT-LLM MHA kernel beats FlashInfer default.
        # Valid for spec decoding only at eagle_topk=1 (our profiles).
        attention_backend="trtllm_mha",
        # Do NOT hardcode --mm-attention-backend fa4: this is a real VLM,
        # but vision is never driven by our text-only clients and fa4 is
        # unverified on the pinned image, so we omit it (auto-select).
        # V2 Mamba scheduler with overlap (selected purely by this
        # strategy string). page_size 64 is forced independently by
        # trtllm_mha's paged MHA (set in the builder default), not by
        # this scheduler.
        mamba_scheduler_strategy="extra_buffer",
        page_size=64,
        # CONCURRENT shape uses the upstream chat template (baked above).
        chat_template=TEMPLATE_IN_IMAGE,
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

    proc = subprocess.Popen(cmd, env=env)
    wait_for_health(proc, timeout_s=60 * 20, port=SERVE_PORT, label=APP_NAME)
    # Block on the inference server. If it exits, the container exits
    # and Modal restarts on the next request (within scaledown_window).
    proc.wait()
