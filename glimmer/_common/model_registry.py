"""Muse Glimmer model spec, shared across deployments.

Single source of truth for the checkpoint, pinned revisions, the DFlash
drafter, and the per-GPU memory budgets. ``serve.py`` calls ``get("nvfp4")``.

The model
---------
Muse Glimmer 30B (Meta) is a *dense* causal transformer with a ~1.8B
perception encoder (ViT-G/14) for interleaved text + image input. 52
layers in a repeating ``[local, local, local, global]`` pattern: 39
sliding-window layers (window 2048, RoPE) and 13 global layers (NoPE).
GQA 32 query / 2 KV heads, head_dim 128. Native context 131_072.

That attention layout makes the KV cache unusually small:

  - global layers:  13 layers x 2 KV heads x 128 x (K+V) = 6_656 values
                    per token -> ~6.5 KiB/token in FP8 (13 KiB in BF16).
                    A full 131K-token session is ~0.85 GiB of FP8 KV.
  - sliding layers: bounded by the 2048 window, so a fixed ~40 MiB per
                    session in FP8 no matter how long the context is.

SGLang sizes the sliding-window pool as a ratio of the full-attention pool
(``--swa-full-tokens-ratio``, default 0.8). The sliding layers only ever
need ~2K tokens per running request, so the deployments set it to 0.1:
on an RTX PRO 6000 that grows the pool from 1.77M to 3.07M full-context
tokens at identical throughput, and a needle at 118K tokens is still
retrieved correctly.

The checkpoint
--------------
``nvidia/Muse-Glimmer-30B-NVFP4`` is NVIDIA ModelOpt's AutoQuantize
mixed-precision export (~24.7 GB on disk vs ~60 GB BF16):

  - 213 linear layers in W4A16 NVFP4 (4-bit weights, group 16, BF16
    activations),
  - 180 linear layers in FP8 (weights + activations),
  - the perception encoder and everything unmatched stay BF16.

NVIDIA's published evals put it within noise of BF16 (GPQA 83.0 vs 83.8,
MMMU-Pro 73.6 vs 74.2, Terminal-Bench 47.1 vs 45.2). Unlike the
``RadixArk/Muse-Glimmer-NVFP4`` repack the SGLang cookbook uses, this
checkpoint *keeps the vision tower*, so image input works.

NVFP4 needs Blackwell. On SM100 (B200/B300) SGLang uses the native FP4
kernels; on SM120 (RTX PRO 6000) SGLang's Glimmer override picks the
Marlin FP4 GEMM runner automatically.

The drafter
-----------
``meta-models/Muse-Glimmer-30B-assistant`` is a 5-layer DFlash
block-diffusion drafter (block size 16, sliding-window attention,
conditioned on target layers {1, 13, 25, 37, 49}). It ships in BF16
(~5.1 GB); SGLang quantizes its linears to FP8 at load time with
``--speculative-draft-model-quantization fp8`` (no accept-rate loss in the
SGLang cookbook's measurements).

References
----------
- Base model card:   https://huggingface.co/meta-models/Muse-Glimmer-30B
- NVFP4 model card:  https://huggingface.co/nvidia/Muse-Glimmer-30B-NVFP4
- DFlash drafter:    https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant
- SGLang support:    https://github.com/sgl-project/sglang/pull/34262 (v0.5.18+)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftSpec:
    """DFlash speculative-decoding drafter (separate HF repo)."""

    hf_repo: str
    hf_revision: str | None
    quantization: str | None
    """Load-time quantization for the drafter's linears (``fp8``), or
    None to serve it in BF16."""


@dataclass(frozen=True)
class ModelSpec:
    short: str
    """Short CLI key (app names, bench labels)."""

    hf_repo: str
    """HF repo loaded by the runtime. Apache-2.0 and ungated."""

    hf_revision: str | None
    """Pinned commit SHA. SGLang re-captures CUDA graphs on any weight
    change and memory snapshots bake the loaded weights, so a moving
    ``main`` would silently change behavior between cold boots."""

    native_max_model_len: int
    """``max_position_embeddings`` from the checkpoint config."""

    draft: DraftSpec | None

    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "nvfp4": ModelSpec(
        short="nvfp4",
        hf_repo="nvidia/Muse-Glimmer-30B-NVFP4",
        hf_revision="47818374517751c48c55cde2621594926b1888b6",
        native_max_model_len=131_072,
        draft=DraftSpec(
            hf_repo="meta-models/Muse-Glimmer-30B-assistant",
            hf_revision="c676cce91658f9568aaf264a416211e2782735b7",
            quantization="fp8",
        ),
        notes=(
            "Dense 30B + 1.8B perception encoder. ModelOpt mixed NVFP4 "
            "(W4A16) / FP8 / BF16, ~24.7 GB. Vision weights included. "
            "Blackwell only (SM100 native FP4, SM120 via Marlin)."
        ),
    ),
}


# ── Per-GPU profiles ─────────────────────────────────────────────────────
#
# The Modal GPU classes this project is tested on. B200+ is deliberately
# NOT offered: Modal may land it on a B300, which needs CUDA 13.1+, and the
# pinned SGLang image is CUDA 13.0.


@dataclass(frozen=True)
class GpuProfile:
    mem_fraction_static: float
    """Share of GPU memory SGLang claims up front (weights + KV pool). The
    rest is headroom for activations, CUDA graphs, DFlash buffers, and the
    vision encoder on image requests."""

    attention_backend: str | None
    """None lets SGLang choose (FlashInfer on both cards). ``trtllm_mha``
    is SM100-only and measured +17% single-stream decode on B200."""

    usd_per_hour: float
    """Modal list price, for the README's cost math. Check modal.com/pricing."""


GPU_PROFILES: dict[str, GpuProfile] = {
    "RTX-PRO-6000": GpuProfile(mem_fraction_static=0.85, attention_backend=None, usd_per_hour=3.03),
    "B200": GpuProfile(mem_fraction_static=0.85, attention_backend="trtllm_mha", usd_per_hour=6.25),
}


def get(short: str) -> ModelSpec:
    try:
        return REGISTRY[short]
    except KeyError as e:
        valid = ", ".join(REGISTRY)
        raise KeyError(f"unknown model {short!r}; expected one of: {valid}") from e
