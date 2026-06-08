"""Qwen3.6 model specs, shared across deployments.

This registry is the single source of truth for HF repo, GPU class, and
context window. It holds two models that share the same SGLang launch
machinery but differ in architecture:

  - ``27b`` — Qwen3.6-27B, a dense *hybrid* decoder (Gated DeltaNet linear
    layers interleaved with Gated full-attention layers).
  - ``35b`` — Qwen3.6-35B-A3B, a sparse Mixture-of-Experts (MoE) hybrid
    (~35B total params, ~3B active per token).

Each model can be deployed in two shapes (the deployment scripts pick the
knobs; the registry only carries the invariant facts):

  - A *concurrent* shape — several simultaneous agentic sessions on one
    GPU, full native context each, fair-share scheduling between them.
  - A *solo* shape — a single session monopolises the GPU, with the KV
    cache and speculative-decoding budgets tuned for one fast stream.

Per-variant tuning (MTP profile, max_running_requests, chunked_prefill_size)
lives in the deployment script itself, where the intent is most visible.

References
----------
- Qwen3.6-27B model card: https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- LMSYS SGLang cookbook: https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    short: str
    """Short CLI key (app names, bench labels)."""

    hf_repo: str
    """HF repo loaded by the runtime. These weights are ungated."""

    gpu: str
    """Modal GPU class. ``B200+`` lets Modal opt in to B300 when available
    and falls back to B200; Modal bills it as B200 either way (see Modal's
    GPU docs). B300 needs CUDA 13.0+, which the lmsysorg/sglang image
    already ships."""

    max_model_len: int
    """Context window. 262_144 is native; YaRN extends it to ~1M with a
    config-time override we don't apply here."""

    notes: str = ""


# Pinned commit revision — set after first successful deploy. Leaving it
# blank pulls the repo's default branch (`main`). Pin once you've validated
# a deploy; SGLang's CUDA graph capture re-runs on any weight change, so a
# moving `main` can silently change behavior between cold boots.
HF_REVISION: str | None = None


REGISTRY: dict[str, ModelSpec] = {
    "27b": ModelSpec(
        short="27b",
        hf_repo="Qwen/Qwen3.6-27B",
        # `B200+` opts into B300 when available and falls back to B200.
        # Modal bills as B200 regardless. B300 needs CUDA 13.0+ which the
        # lmsysorg/sglang image already ships.
        gpu="B200+",
        max_model_len=262_144,
        notes=(
            "Dense hybrid 27B: 64 layers, pattern (3 x Gated DeltaNet + "
            "1 x Gated Attention) x 16 (16 full-attention + 48 linear "
            "layers). Native 262K context. BF16 weights (~54 GiB). MTP is "
            "architectural (an in-model head, not a separate drafter "
            "model). Multimodal VLM (vision_config present); used here for "
            "text-only agentic workloads, so the tower loads but clients "
            "simply never send image inputs. TP=1 single GPU."
        ),
    ),
    "35b": ModelSpec(
        short="35b",
        hf_repo="Qwen/Qwen3.6-35B-A3B",
        # `B200+` opts into B300 when available and falls back to B200.
        # Modal bills as B200 regardless. B300 needs CUDA 13.0+ which the
        # lmsysorg/sglang image already ships. This MoE serves at TP=1 on a
        # SINGLE GPU (per the SGLang Qwen3.6 cookbook hardware table);
        # single-GPU also makes it Modal-snapshot eligible.
        gpu="B200+",
        max_model_len=262_144,
        notes=(
            "MoE hybrid: 40 layers, ~35B total / ~3B active params. 256 "
            "experts (8 routed + 1 shared per token). Hybrid backbone (30 "
            "Gated-DeltaNet linear + 10 Gated full-attention layers, "
            "full_attention_interval=4). Native 262K context. BF16 weights "
            "(~71.9 GB / ~67 GiB on disk, 26 shards). An FP8 variant "
            "(Qwen/Qwen3.6-35B-A3B-FP8, ~35 GB, vision blocks unquantized) "
            "exists as a long-context/OOM contingency, not the default; "
            "BF16 fits one B200 with ample headroom. MTP is architectural "
            "(mtp_num_hidden_layers=1, fused into the normal safetensors "
            "shards as mtp.* tensors; no standalone mtp.safetensors and no "
            "separate drafter repo). Multimodal VLM (vision_config "
            "present); used here for text-only agentic workloads. TP=1 "
            "single B200 per the SGLang Qwen3.6 cookbook."
        ),
    ),
}


def get(short: str) -> ModelSpec:
    try:
        return REGISTRY[short]
    except KeyError as e:
        valid = ", ".join(REGISTRY.keys())
        raise KeyError(f"unknown model {short!r}; expected one of: {valid}") from e


# Default spec for callers that import a single SPEC. Deployment scripts
# should prefer `get("27b")` / `get("35b")` explicitly so the model choice
# is visible at the call site.
SPEC = get("27b")
