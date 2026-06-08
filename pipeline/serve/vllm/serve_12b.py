"""Deploy google/gemma-4-12B-it (dense) on Modal (H100) via vLLM.

The dense mid-size of the Gemma 4 family — ~11.95B parameters. It sits
between the small E-series and the large 26B-A4B / 31B endpoints, and is
the comfortable single-GPU "production solo + moderate concurrency"
target: bf16 weights are ~24 GiB, so a single H100 80GB leaves ample
HBM for KV cache.

Config facts (from the public config.json; the repo is Apache-2.0 and
ungated):

  - model_type=gemma4_unified,
    architectures=[Gemma4UnifiedForConditionalGeneration]
  - num_hidden_layers=48, hidden_size=3840, num_attention_heads=16,
    num_key_value_heads=8, vocab_size=262144, torch_dtype=bfloat16
  - head_dim=256 is EXPLICIT in config (NOT hidden_size/num_heads = 240;
    Gemma fixes head_dim=256). The serving stack must use 256, not the
    inferred value.
  - Hybrid local/global attention: layer_types repeat 5x
    sliding_attention (sliding_window=1024) then 1x full_attention
    across the 48 layers (5:1 local:global), with distinct RoPE for
    full vs sliding layers.
  - Multimodal: vision (mm_embed_dim 3840, patch_size 16,
    num_soft_tokens 280) + audio (audio_embed_dim 640,
    audio_samples_per_token 640). We serve text-only here.
  - Native context 262144 (256K): max_position_embeddings=262144.
    `max_model_len` in the registry is set well below that so KV cache
    has room for concurrency — bump toward 262144 if you need long
    context and can afford the KV-cache footprint.

The chat template is the Gemma-4 unified format and is byte-for-byte
identical to the on-disk 31B upstream template (both 17466 bytes,
identical SHA-256), so any custom template fork applies cleanly with no
per-size adjustment.

## Attention backend — triton

Gemma 4's hybrid sliding/full attention plus the fixed head_dim=256 is
incompatible with vLLM's default flash kernels; the supported path is
the triton attention backend. The registry marks this size
`requires_triton_attention=True`; we pin it below via the
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` environment variable on the serve
subprocess so vLLM doesn't silently fall back to a flash kernel that
can't handle the head_dim=256 / hybrid pattern.

## Setup (one-time)

1. `modal token new` — authenticate the CLI.
2. Gemma 4 is Apache 2.0, NOT gated — no HF token required.
3. `modal deploy serve/vllm/serve_12b.py`. First deploy pulls ~24 GiB
   of weights into `gemma4-hf-cache`; cold boot ~2-3 min on first pull,
   warm ~30 s.

## Consumption

After deploy, Modal prints a PUBLIC base URL (the URL printed by
`modal deploy`, of the form `*.modal.run`). Point any OpenAI-compatible
client at the server root. The vLLM server binds `0.0.0.0` so Modal's
ingress can reach it. Endpoints are public by default; to require
auth, lock it down at the ingress with proxy auth — see
modal.com/docs/guide/webhook-proxy-auth. This script does not
implement auth.

## Tool-call parser

ON by default — this is the registry's moderate-concurrency size
(`concurrency=32`). vLLM #39392's shared-state bug can leak `<pad>`
tokens under sustained concurrent tool-call traffic. If you drive this
endpoint hard at concurrent tool-use, flip `enable_tool_call_parser`
to False and parse raw `<|tool_call>...` tokens client-side via
`_common.gemma4_parser` (same mitigation the 26B corpus endpoint uses).

## Thinking mode

Per-request via `chat_template_kwargs.enable_thinking=True` in the
OpenAI request body. To pin a server-wide default, pass
`default_thinking=True/False` to `build_serve_cmd`. Note vLLM #39130:
`enable_thinking=false` + a `response_format` constraint silently
bypasses xgrammar — set thinking on when grammar enforcement matters.

## Multi-token prediction (MTP / speculative decoding)

NOT available for this size: there is **no published drafter**
(`google/gemma-4-12B-it-assistant` does not exist), and config carries
no MTP / num_nextn_predict head. `SPEC.assistant_repo` is None, so the
`ENABLE_MTP` toggle is a no-op here — speculative decoding would
require a separately sourced or trained draft model. (Even with a
drafter, vLLM 0.19.1's spec-decode path rejects multimodal targets;
see the folder README.)

## KV-cache dtype

bf16 by default (matches torch_dtype=bfloat16), inherited from
`SPEC.kv_cache_dtype=None`. fp8 KV cache is a viable memory
optimisation for high-concurrency long-context serving — set
`kv_cache_dtype="fp8"` on `build_serve_cmd` after validating against
your eval quality bar.

## Hardware + cost

- **GPU**: single H100 80GB (registry default). bf16 weights ~24 GiB
  leave ample room for KV cache. An A100 80GB or even an L40S 48GB also
  works for solo serving. For higher concurrency, an H200 141GB (or
  2xH100 via `--tensor-parallel-size 2`, set `gpu="H100:2"`) gives
  headroom; the 5:1 hybrid KV (only 1/6 layers full-attention) keeps
  long-context KV growth modest, but many concurrent 256K streams still
  pressure HBM.
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

SPEC = get("12b")

# Optional GPU override. None = use the registry-canonical H100. Set to
# "A100-80GB" or "L40S" for solo serving, or to "H200" / "H100:2" (with
# `--tensor-parallel-size 2` in extra_args) for higher concurrency.
GPU_OVERRIDE: str | None = None

# MTP / speculative decoding has NO drafter for 12B-it
# (`SPEC.assistant_repo` is None), so this toggle is inert for this
# size and kept only for symmetry with the other serve scripts. Even
# with a drafter, vLLM 0.19.1 rejects multimodal targets in spec-decode
# (vllm/v1/spec_decode/eagle.py). See the folder README.
ENABLE_MTP = False

app = modal.App("gemma4-12b-concurrent")
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
    gpu=GPU_OVERRIDE or SPEC.gpu,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=SECRETS,
    timeout=60 * 60 * 2,
    scaledown_window=60 * 10,
    # Scale out if one replica saturates under concurrency. Many
    # concurrent long-context streams pressure HBM; 4 containers gives
    # headroom without surprise billing.
    max_containers=4,
)
@modal.concurrent(max_inputs=SPEC.concurrency)
# First-pull cold boot ~3-5 min on ~24 GiB weights + 12B torch.compile.
# Warm boots run in ~30 s. 1200 s gives headroom for the first pull
# without surprising on warm restarts.
@modal.web_server(port=8000, startup_timeout=1200)
def serve() -> None:
    """Launch `vllm serve` for 12B dense on port 8000."""
    import os
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
        # Moderate-concurrency size — parser ON by default. Flip to
        # False (and parse client-side) if you saturate it with
        # concurrent tool-use; see vLLM #39392 and the folder README.
        enable_tool_call_parser=True,
        enable_prefix_caching=True,
        enable_async_scheduling=True,
        fast_boot=False,
        # bf16 KV cache (SPEC.kv_cache_dtype is None → runtime default);
        # set "fp8" here for high-concurrency long-context memory wins.
        kv_cache_dtype=SPEC.kv_cache_dtype,
        speculative_config=speculative_config,
    )

    # Pin the triton attention backend. Gemma 4's hybrid sliding/full
    # attention + fixed head_dim=256 is incompatible with the default
    # flash kernels (SPEC.requires_triton_attention is True). Set via
    # the env var rather than a CLI flag so vLLM does not silently fall
    # back to a flash kernel that can't handle head_dim=256.
    env = dict(os.environ)
    if SPEC.requires_triton_attention:
        env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

    proc = subprocess.Popen(cmd, env=env)
    wait_for_health(proc, timeout_s=1200, label=f"gemma4-{SPEC.short}")
