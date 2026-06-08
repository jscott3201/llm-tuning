"""Deploy google/gemma-4-26B-A4B-it (MoE) on Modal (B200) via vLLM.

A high-concurrency teacher / corpus-generator endpoint. Sparse MoE with
~4B active params per token (the "A4B" suffix) over 26.5B total — cheap
to run because activation cost is dominated by the active-expert subset,
not the full weight set. The high concurrency (target 64-way) is what
makes the tool-call parser flag choice non-obvious.

## Setup (one-time)

1. `modal token new` — authenticate the CLI.
2. Gemma 4 is Apache 2.0, NOT gated — no HF token required.
3. `modal deploy serve/vllm/serve_26b.py`. First deploy pulls ~52 GiB
   of weights (~3-5 min on Modal's network); subsequent warm boots
   ~2-3 min thanks to the persistent volume.

## Consumption

After deploy, Modal prints a PUBLIC base URL (the URL printed by
`modal deploy`, of the form `*.modal.run`). Point an OpenAI-compatible
client at the server root. The vLLM server binds `0.0.0.0` so Modal's
ingress can reach it. Endpoints are public by default; to require
auth, lock it down at the ingress with proxy auth — see
modal.com/docs/guide/webhook-proxy-auth. This script does not
implement auth.

## Tool-call parser — DISABLED on this endpoint

vLLM #39392 reported `<pad>` token leakage under concurrent
`--tool-call-parser gemma4` traffic. Status across 0.19.x is
ambiguous (the 0.19.1 release notes mention concurrent-correctness
work, but the issue isn't explicitly closed). At a target of 64-way
concurrent tool-use, the conservative move is to disable the
server-side parser entirely.

This endpoint runs with the server-side parser **off**. Raw
`<|tool_call>call:fn{k:<|"|>v<|"|>}<tool_call|>` tokens come through
in the content field; `_common/gemma4_parser.py` extracts them
client-side. Side benefit: SFT can train on exactly the same raw
token shape this endpoint emits, so train/serve shapes stay identical.

## Structured-output gotcha (vLLM #40080)

Gemma 4 31B and 26B-A4B have been observed to fall into infinite
repetition under JSON-schema-constrained generation, particularly
when a free-form string field is part of the schema. If your corpus
generator uses `response_format`, set `repetition_penalty` >= 1.05
and/or `frequency_penalty` >= 0.5 on the request to mitigate. A
generator that parses raw `<|tool_call>` tokens (no structured output)
typically doesn't fire this.

## Thinking mode

Per-request via `chat_template_kwargs.enable_thinking=True` in the
OpenAI request body. Server-wide default is left at the template's
own default (no thinking) since corpus generation toggles per turn —
agent turns set it on; user-persona turns set it off so the persona
doesn't preface its message with reasoning. To flip the server
default, pass `default_thinking=True` to `build_serve_cmd`.

## Hardware + cost

- **GPU**: single B200 (192 GiB HBM3e). 26B-A4B weights at ~48.5 GiB
  leave ~140 GiB for KV cache. At 32k context, prefix caching makes
  20-64 way concurrency easy because every corpus-gen call shares the
  same system prompt + tool manifest prefix.
- **Cost**: B200 is Modal's top tier. A typical multi-hour 64-way
  corpus run lands in the low tens of dollars.
- **Cold start**: first pull ~3-5 min, warm ~30 s.
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

SPEC = get("26b")

# MTP / speculative decoding is currently UNAVAILABLE for Gemma 4 in
# vLLM 0.19.1: the spec-decode path rejects multimodal targets and all
# published Gemma 4 checkpoints are multimodal
# (`Gemma4*ForConditionalGeneration`). See the folder README for the
# full detail. (Even if it worked: the 26B is sparse MoE, where MTP has
# batch-size constraints due to expert-loading overhead — benchmark
# carefully before enabling for the 64-way corpus-gen workload.)
ENABLE_MTP = False

app = modal.App("gemma4-26b-concurrent")
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
    # 2-hour soft timeout — corpus-gen tranches typically run 1-2 hr.
    timeout=60 * 60 * 2,
    scaledown_window=60 * 10,
    # Scale out if one replica saturates. 64-way + probe overlap can
    # push past a single replica's KV budget; 4 containers gives
    # headroom without surprise billing.
    max_containers=4,
)
@modal.concurrent(max_inputs=SPEC.concurrency)
# First-pull cold boot ~12 min on 52 GiB weights + 26B torch.compile.
# Warm boots run in ~2-3 min. 1200 s gives headroom for first-pull
# without surprising on warm restarts.
@modal.web_server(port=8000, startup_timeout=1200)
def serve() -> None:
    """Launch `vllm serve` for 26B-A4B on port 8000."""
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
        # OFF — see file docstring on vLLM #39392. Client-side parser
        # extracts raw tokens at our target concurrency.
        enable_tool_call_parser=False,
        enable_prefix_caching=True,
        enable_async_scheduling=True,
        fast_boot=False,
        speculative_config=speculative_config,
    )

    # Do NOT log `cmd` directly — would leak `--api-key` value if that
    # optional hook is set. `wait_for_health` redacts before raising.
    proc = subprocess.Popen(cmd)

    # 20-minute warm-up budget covers the first-pull cold boot.
    wait_for_health(proc, timeout_s=1200, label=f"gemma4-{SPEC.short}")
