"""Deploy google/gemma-4-E2B-it on Modal (L4) via vLLM.

The smallest of the Gemma 4 family — fits comfortably on a single L4
(24 GiB) and is the cheapest of the five sizes to run. Useful as the
lowest-cost probe target when you're iterating on prompts or scenarios.

## Setup (one-time)

1. `modal token new` — authenticate the CLI.
2. Gemma 4 is Apache 2.0, NOT gated — no HF token required.
3. `modal deploy serve/vllm/serve_e2b.py`.

## Ingress

`@modal.web_server(port=8000, ...)` publishes a PUBLIC `*.modal.run`
URL — use the URL printed by `modal deploy`. The vLLM server binds
`0.0.0.0` (set in `build_serve_cmd`) so Modal's ingress can reach it.
Endpoints are public by default; to require auth, lock it down at the
ingress with proxy auth — see modal.com/docs/guide/webhook-proxy-auth.
This script does not implement auth.

## Tool-call parser

ON. E2B traffic is low-concurrency; vLLM #39392 doesn't bite.

## Multi-token prediction (MTP / speculative decoding)

Flip `ENABLE_MTP = True` below to pair the target with its 78M
companion drafter (`google/gemma-4-E2B-it-assistant`). Recipe-
recommended `num_speculative_tokens=2` for E2B; the smaller drafter
amortises better at low values. Reported throughput gain is up to
~3× at low batch sizes; verification is exact so quality is
identical to the target.

⚠️ Currently UNAVAILABLE in vLLM 0.19.1 — see the `ENABLE_MTP` comment
below and the folder README.

## Cost

L4 on Modal: smallest/cheapest of the five. Cold boot ~30-60 s once
the weights are warm in the cache; weights are ~5 GiB.
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

SPEC = get("e2b")

# Optional GPU override. None = use the registry-canonical L4 (the
# cost-appropriate production choice for E2B's ~5 GiB weights). Set to
# "L40S" or "B200" if Modal is queue-bound on L4, or to run all sizes on
# uniform hardware for a like-for-like benchmark.
GPU_OVERRIDE: str | None = None

# MTP / speculative decoding is currently UNAVAILABLE for Gemma 4 in
# vLLM 0.19.1 because all published multimodal checkpoints are
# `Gemma4*ForConditionalGeneration` and vLLM's spec-decode path raises
# NotImplementedError on multimodal targets
# (vllm/v1/spec_decode/eagle.py). See the folder README for the full
# detail. Track upstream before flipping this on.
ENABLE_MTP = False

app = modal.App("gemma4-e2b-solo")
vllm_image = make_vllm_image(VLLM_VERSION)

# Shared HF cache across the serve scripts — different weight trees
# under the same volume; Modal Volumes handle concurrent reads fine.
hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("torchinductor-cache", create_if_missing=True)

SECRETS: list[modal.Secret] = [
    # Optional — gives authenticated rate limits on Hub pulls. These
    # Gemma 4 checkpoints are ungated/public, so a token is NOT
    # required. `huggingface-secret` is a Modal Secret you create in
    # your own workspace holding `HF_TOKEN`. Comment this out if you
    # don't want to mount a token.
    modal.Secret.from_name("huggingface-secret"),
]


@app.function(
    image=vllm_image,
    gpu=GPU_OVERRIDE or SPEC.gpu,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=SECRETS,
    timeout=60 * 60,
    scaledown_window=60 * 10,
    max_containers=2,
)
@modal.concurrent(max_inputs=SPEC.concurrency)
# 15-minute warm-up budget. The "warm cold-boot" docstring time of
# 30-60 s assumes the weights are already in `gemma4-hf-cache` AND
# torch.compile artefacts are warm. First-ever deploys hit both cold
# and need ~5-10 min on L4. 900 s gives headroom without silently
# failing under HuggingFace rate limiting either.
@modal.web_server(port=8000, startup_timeout=900)
def serve() -> None:
    """Launch `vllm serve` for E2B on port 8000."""
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
        # Probe-style traffic; parser bug doesn't fire at this concurrency.
        enable_tool_call_parser=True,
        enable_prefix_caching=True,
        enable_async_scheduling=True,
        fast_boot=False,
        speculative_config=speculative_config,
    )

    proc = subprocess.Popen(cmd)
    wait_for_health(proc, timeout_s=900, label=f"gemma4-{SPEC.short}")
