"""Canonical Gemma 4 family registry.

Centralises the (short-name → HF repo + recommended Modal GPU + sizing
hints) mapping so every script in the pipeline picks consistent defaults.
The Hugging Face URLs for reference:

  https://huggingface.co/google/gemma-4-E2B-it
  https://huggingface.co/google/gemma-4-E4B-it
  https://huggingface.co/google/gemma-4-12B-it
  https://huggingface.co/google/gemma-4-26B-A4B-it
  https://huggingface.co/google/gemma-4-31B-it

Most sizes also ship an `-assistant` drafter for multi-token prediction
(MTP) / speculative decoding:

  https://huggingface.co/google/gemma-4-E2B-it-assistant       (78M)
  https://huggingface.co/google/gemma-4-E4B-it-assistant       (79M)
  https://huggingface.co/google/gemma-4-26B-A4B-it-assistant   (420M)
  https://huggingface.co/google/gemma-4-31B-it-assistant       (470M)

(The 12B-it entry has no published drafter — see its `assistant_repo`
note below.)

The MTP drafters are NOT separate chat models — they're tiny companion
networks trained to draft tokens that the target model verifies in
parallel (the speculative-decoding recipe from Leviathan et al.
arXiv:2211.17192). Quality stays identical to the target model
(verification is exact); throughput goes up by up to ~3× depending on
batch size and acceptance rate.

⚠️ **vLLM 0.19.1 cannot currently use these drafters with the Gemma 4
checkpoints.** The published multimodal targets are
`Gemma4*ForConditionalGeneration`; vLLM's spec-decode path raises
`NotImplementedError: Speculative Decoding with draft models or
parallel drafting does not support multimodal models yet` at engine
init. The drafter info is kept in the registry so the toggle is a
one-line flip once upstream ships the multimodal-target path. See the
SGLang notes in `sglang_common.py` for the runtime that does support
the Gemma 4 MTP path today (PR #24436).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Per-model serving + sizing config.

    `gpu` follows Modal's `@app.function(gpu=...)` syntax. `concurrency`
    is the starting point for `@modal.concurrent(max_inputs=...)` —
    push higher in your own deployment until KV cache evictions show
    up in the logs.
    """

    short: str
    """Short CLI key (e2b / e4b / 12b / 26b / 31b)."""

    hf_repo: str
    """Canonical Hugging Face repo ID."""

    gpu: str
    """Modal GPU class string."""

    max_model_len: int
    """vLLM `--max-model-len`. Smaller than the model's native context
    so KV cache has room for many concurrent sequences. Bump toward the
    native window if you need long context and can afford the KV-cache
    footprint."""

    max_num_batched_tokens: int
    """vLLM `--max-num-batched-tokens`. Sized to saturate the GPU's HBM bandwidth."""

    gpu_memory_utilization: float
    """Fraction of HBM vLLM may use for weights + KV cache. The
    canonical recipe recommends 0.90 across sizes; we follow that with a
    small downward tweak on E2B (L4 is the smallest target and benefits
    from a larger activation reserve)."""

    concurrency: int
    """Starting `@modal.concurrent(max_inputs=...)` ceiling."""

    is_moe: bool
    """True for sparse-MoE checkpoints (26B-A4B). Affects PEFT target modules."""

    assistant_repo: str | None = None
    """Canonical drafter for multi-token-prediction (MTP) / speculative
    decoding. The E2B / E4B / 26B-A4B / 31B sizes each ship a tiny
    companion model under `google/<size>-it-assistant`; vLLM accepts it
    via `--speculative-config '{"model": ..., "num_speculative_tokens": N}'`.
    Reported throughput speedup: up to ~3× (Gemma 4 MTP blog post). Set
    None to opt out per model — e.g. 12B-it has no published drafter."""

    speculative_tokens: int = 0
    """Recommended `num_speculative_tokens` for this size, from the
    vLLM Gemma 4 recipe. 0 means MTP is disabled for this entry."""

    head_dim: int | None = None
    """Explicit attention head dim when it differs from
    `hidden_size / num_attention_heads`. Gemma 4 fixes head_dim=256, so
    the serving stack must not infer it — the inferred value is wrong
    and the sliding/full hybrid attention pattern plus head_dim=256 is
    what forces the triton attention backend on this family."""

    requires_triton_attention: bool = False
    """When True, the SGLang serve command should pass
    `--attention-backend triton`. Gemma 4's hybrid local/global
    attention (sliding_window layers interleaved with full_attention)
    together with head_dim=256 is incompatible with the default flash
    kernels; triton is the supported path."""

    kv_cache_dtype: str | None = None
    """Recommended `--kv-cache-dtype` for this size, or None to inherit
    the runtime default (bf16, matching torch_dtype). fp8 is a viable
    high-concurrency long-context memory optimisation but should be
    validated against eval quality before adopting."""

    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "e2b": ModelSpec(
        short="e2b",
        hf_repo="google/gemma-4-E2B-it",
        gpu="L4",
        max_model_len=16_384,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.88,
        concurrency=16,
        is_moe=False,
        assistant_repo="google/gemma-4-E2B-it-assistant",
        speculative_tokens=2,
        head_dim=256,
        requires_triton_attention=True,
        notes=(
            "Smallest member: 5.1B total params, ~2B effective via PLE "
            "(Per-Layer Embeddings). Multimodal (text + vision + audio "
            "towers) but loaded with AutoModelForCausalLM. Fits "
            "comfortably on an L4 (24 GiB) — text-only inference uses "
            "well under half the card. MTP drafter is 78M params and "
            "shares input embeddings with the target."
        ),
    ),
    "e4b": ModelSpec(
        short="e4b",
        hf_repo="google/gemma-4-E4B-it",
        gpu="L40S",
        max_model_len=16_384,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.90,
        concurrency=24,
        is_moe=False,
        assistant_repo="google/gemma-4-E4B-it-assistant",
        speculative_tokens=4,
        head_dim=256,
        requires_triton_attention=True,
        notes=(
            "8.0B total params, ~4B effective via PLE. A practical SFT "
            "target — PLE makes it punch above its weight class on tool "
            "use while keeping LoRA training tractable on a single "
            "L40S. MTP drafter is 79M params."
        ),
    ),
    "12b": ModelSpec(
        short="12b",
        hf_repo="google/gemma-4-12B-it",
        gpu="H100",
        max_model_len=32_768,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.90,
        concurrency=32,
        is_moe=False,
        assistant_repo=None,
        speculative_tokens=0,
        head_dim=256,
        requires_triton_attention=True,
        kv_cache_dtype=None,  # bf16 (matches torch_dtype); fp8 is opt-in.
        notes=(
            "~11.95B dense (config.json does NOT publish a total param "
            "count; layers/dims confirm the ~12B class). Confirmed dims: "
            "num_hidden_layers=48, hidden_size=3840, "
            "num_attention_heads=16, num_key_value_heads=8, "
            "vocab_size=262144. model_type=gemma4_unified, "
            "architectures=[Gemma4UnifiedForConditionalGeneration], "
            "torch_dtype=bfloat16. Multimodal: vision (mm_embed_dim 3840, "
            "patch_size 16, num_soft_tokens 280) + audio (audio_embed_dim "
            "640, audio_samples_per_token 640). "
            "Native context 262144 (256K): max_position_embeddings=262144; "
            "tokenizer_config model_max_length is the sentinel int (1e30), "
            "so the real native window is the config's 262144 — "
            "max_model_len here is set well below that so KV cache has "
            "room for concurrency. "
            "head_dim=256 is EXPLICIT in config (NOT hidden_size/num_heads "
            "= 240; Gemma uses a fixed 256 head_dim). Hybrid local/global "
            "attention: layer_types repeat 5x sliding_attention "
            "(sliding_window=1024) then 1x full_attention across 48 layers "
            "(5:1 local:global), distinct RoPE for full vs sliding layers. "
            "The family-wide triton-attention-backend constraint applies "
            "(the sliding/full hybrid plus head_dim=256 forces triton over "
            "the default flash kernels). "
            "Recommended GPU solo: single H100 80GB (or A100 80GB); bf16 "
            "weights ~24GB leave ample room for a large 256K KV alloc for "
            "one stream (an L40S 48GB also works for solo). For "
            "concurrency: single H100 for moderate, H200 141GB (or "
            "2xH100) for higher — the 5:1 hybrid KV (only 1/6 layers "
            "full-attention) keeps long-context KV growth modest but many "
            "concurrent 256K streams still pressure HBM. "
            "No published MTP/speculative drafter: no "
            "google/gemma-4-12B-it-assistant exists and config has no "
            "MTP / num_nextn_predict head, so assistant_repo is None and "
            "speculative decoding would require a separately sourced/"
            "trained draft model. "
            "Chat template is byte-for-byte identical to the on-disk 31B "
            "upstream template (both 17466 bytes, identical SHA-256) — "
            "the Gemma-4 unified format, so the custom template fork "
            "applies cleanly with no per-size adjustment. Sampling "
            "defaults: temperature=1.0, top_p=0.95, top_k=64 (Gemma "
            "family; applied at generation_config / serving level)."
        ),
    ),
    "26b": ModelSpec(
        short="26b",
        hf_repo="google/gemma-4-26B-A4B-it",
        gpu="B200",
        max_model_len=32_768,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.90,
        concurrency=64,
        is_moe=True,
        assistant_repo="google/gemma-4-26B-A4B-it-assistant",
        speculative_tokens=4,
        head_dim=256,
        requires_triton_attention=True,
        notes=(
            "26.5B total params, sparse MoE with ~4B active per token "
            "(the 'A4B' suffix). A high-concurrency teacher endpoint — "
            "cheap because activation cost is dominated by the "
            "active-expert subset, not the full weight set. MoE has "
            "batch-size constraints under MTP due to expert-loading "
            "overhead — measure before turning speculative decoding on "
            "for high-concurrency workloads."
        ),
    ),
    "31b": ModelSpec(
        short="31b",
        hf_repo="google/gemma-4-31B-it",
        gpu="B200",
        max_model_len=32_768,
        max_num_batched_tokens=32_768,
        gpu_memory_utilization=0.90,
        concurrency=32,
        is_moe=False,
        assistant_repo="google/gemma-4-31B-it-assistant",
        speculative_tokens=4,
        head_dim=256,
        requires_triton_attention=True,
        notes=(
            "32.7B total params, dense (no MoE / no PLE — every token "
            "activates the full parameter set). Same hourly cost as "
            "26B-A4B but fewer tokens/s per dollar at low concurrency. "
            "Recipe canonical setup is TP2 on 2x80GB; a single B200 "
            "(192 GiB) fits comfortably without TP, which is what we "
            "use here. MTP drafter is 470M params (largest of the "
            "family) and benefits from `num_speculative_tokens` up to 8."
        ),
    ),
}


def get(short: str) -> ModelSpec:
    """Look up a `ModelSpec` by short name. Raises a friendly error
    when the caller passes a name we don't recognise — surfaces the
    valid options instead of an opaque KeyError."""
    try:
        return REGISTRY[short]
    except KeyError as e:
        valid = ", ".join(REGISTRY.keys())
        raise KeyError(
            f"unknown Gemma 4 short name {short!r}; expected one of: {valid}"
        ) from e
