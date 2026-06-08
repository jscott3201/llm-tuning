"""Deploy google/gemma-4-31B-it (dense) on Modal (B200) via vLLM.

The dense large-model reference — a useful comparison point against
the 26B-A4B sparse MoE: same tier on the leaderboard, very different
inference cost profile (every token activates the full 31B parameter
set, vs ~4B active per token on the MoE).

Most flows don't depend on this endpoint, but running it once for a
baseline-vs-baseline score gives the "which Gemma 4 to deploy" answer
with real numbers attached.

## Setup (one-time)

1. `modal token new` — authenticate the CLI.
2. Gemma 4 is Apache 2.0, NOT gated — no HF token required.
3. `modal deploy serve/vllm/serve_31b.py`. First deploy pulls ~62 GiB
   of weights; cold boot ~3-5 min on first pull, warm ~30 s.

## Consumption

After deploy, Modal prints a PUBLIC base URL (the URL printed by
`modal deploy`, of the form `*.modal.run`). Point an OpenAI-compatible
client at the server root. The vLLM server binds `0.0.0.0` so Modal's
ingress can reach it. Endpoints are public by default; to require
auth, lock it down at the ingress with proxy auth — see
modal.com/docs/guide/webhook-proxy-auth. This script does not
implement auth.

## Hardware: single B200 vs TP2 on dual 80GB

The vLLM Gemma 4 recipe canonically shows 31B as TP2 across two 80GB
GPUs — that's the minimum supported configuration. A single B200
(192 GiB) fits the 62 GiB weights + KV cache comfortably without
needing tensor parallelism, which is what we use here. On H100/A100
hardware, switch to TP2 by passing
`extra_args=["--tensor-parallel-size", "2"]` and configuring `gpu`
to `"H100:2"` or `"A100-80GB:2"`.

## Tool-call parser

ON. Use this endpoint at low-to-moderate concurrency for probes and
ad-hoc inference; vLLM #39392 only fires under sustained concurrent
tool-call traffic which is not this endpoint's job.

## Multi-token prediction (MTP)

Flip `ENABLE_MTP = True` below to pair with the 470M drafter
(`google/gemma-4-31B-it-assistant`). Of the sizes with a drafter, 31B
benefits most from MTP — being dense, every token has the full
forward-pass cost, and a draft-and-verify cycle avoids re-running 31B
for every single output token. Recipe `num_speculative_tokens=4` (or up
to 8 for max throughput at high acceptance rates).

⚠️ Currently UNAVAILABLE in vLLM 0.19.1 — see the `ENABLE_MTP` comment
below and the folder README.

## Hardware + cost

- **GPU**: single B200. 31B dense weights at ~62 GiB leave ~125 GiB
  for KV cache.
- **Cost**: B200 is Modal's top tier — same hourly as 26B-A4B but
  fewer tokens/s per dollar at low concurrency because more parameters
  activate per token.
"""

from __future__ import annotations

import modal

from _common.model_registry import get
from _common.vllm_common import (
    VLLM_VERSION,
    build_serve_cmd,
    make_vllm_image,
    wait_for_health,
)

SPEC = get("31b")

# MTP / speculative decoding is currently UNAVAILABLE for Gemma 4 in
# vLLM 0.19.1 because all published checkpoints are
# `Gemma4*ForConditionalGeneration` (multimodal) and vLLM's spec-decode
# path raises `NotImplementedError: Speculative Decoding with draft
# models or parallel drafting does not support multimodal models yet`
# (vllm/v1/spec_decode/eagle.py). Track upstream resolution before
# flipping this back to True.
ENABLE_MTP = False

app = modal.App("gemma4-31b-solo")
vllm_image = make_vllm_image(VLLM_VERSION)

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("torchinductor-cache", create_if_missing=True)

SECRETS: list[modal.Secret] = [
    # Optional — these checkpoints are ungated/public. `huggingface-secret`
    # is a Modal Secret you create in your own workspace holding `HF_TOKEN`.
    modal.Secret.from_name("huggingface-secret"),
]


@app.function(
    image=vllm_image,
    gpu=SPEC.gpu,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=SECRETS,
    timeout=60 * 60 * 2,
    scaledown_window=60 * 10,
    max_containers=2,
)
@modal.concurrent(max_inputs=SPEC.concurrency)
@modal.web_server(port=8000, startup_timeout=1200)
def serve() -> None:
    """Launch `vllm serve` for 31B dense on port 8000."""
    import subprocess

    speculative_config = None
    if ENABLE_MTP and SPEC.assistant_repo:
        speculative_config = {
            "model": SPEC.assistant_repo,
            "num_speculative_tokens": SPEC.speculative_tokens,
        }

    cmd = build_serve_cmd(
        model_path=SPEC.hf_repo,
        served_model_names=[SPEC.hf_repo],
        max_model_len=SPEC.max_model_len,
        gpu_memory_utilization=SPEC.gpu_memory_utilization,
        max_num_batched_tokens=SPEC.max_num_batched_tokens,
        enable_tool_call_parser=True,
        enable_prefix_caching=True,
        enable_async_scheduling=True,
        fast_boot=False,
        speculative_config=speculative_config,
    )

    proc = subprocess.Popen(cmd)
    wait_for_health(proc, timeout_s=1200, label=f"gemma4-{SPEC.short}")
