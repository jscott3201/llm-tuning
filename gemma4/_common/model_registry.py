"""Gemma 4 family model specs, shared across all deployments.

Five model variants are wired today; more from the family can be added by
appending entries to ``REGISTRY`` and per-shape serve scripts.

| Model           | Architecture                          | Native ctx | Drafter        |
|-----------------|---------------------------------------|------------|----------------|
| E2B-it          | Dense+PLE, multimodal incl audio      | 128K       | 78M (assistant)|
| E4B-it          | Dense+PLE, 42L, sliding 512, MM       | 128K       | 78.8M          |
| 12B-it          | Dense, 48L, 5:1 sliding-1024:global   | 262 144    | none published |
| 26B-A4B-it      | MoE, 30L, 128 experts top-8, sl 1024  | 256K       | 0.4B           |
| 31B-it          | Dense, 60L, sliding 1024              | 262 144    | 0.5B (4L)      |

Shared across the family
------------------------
- Same ``gemma4`` tool-call + reasoning parsers in SGLang.
- Same NEXTN MTP recipe (steps=5, draft=6, eagle_topk=1) — see
  ``sglang_common.MTP_NEXTN_STANDARD``. (12B has no published drafter, so
  it runs without speculative decoding.)
- Same sampling card defaults (temp=1.0, top_p=0.95, top_k=64).
- Same fixed head_dim=256 and 512-wide global-attention geometry → the
  Triton attention backend is required across the family.
- Same chat-template family. The 31B, 26B-A4B, and 12B upstream
  ``chat_template.jinja`` files are byte-identical (all 17466 bytes, same
  SHA-256); the baked-in custom fork
  (``chat_templates/custom_pub_chat_template_gemma4.jinja``) applies
  cleanly to all of them with no per-size adjustment. E2B/E4B ship a
  distinct upstream template (``chat_templates/gemma4_e4b_upstream.jinja``,
  17336 bytes) that differs in the thinking-channel pre-fill behaviour.

Per-variant architecture detail lives in the relevant ``ModelSpec.notes``
block. Deployment-specific knobs (context length, max running requests,
GPU class, chat-template path) live in each shape's serve script.

References
----------
- E2B card:          https://huggingface.co/google/gemma-4-E2B-it
- E4B card:          https://huggingface.co/google/gemma-4-E4B-it
- 12B card:          https://huggingface.co/google/gemma-4-12B-it
- 26B-A4B card:      https://huggingface.co/google/gemma-4-26B-A4B-it
- 31B card:          https://huggingface.co/google/gemma-4-31B-it
- SGLang cookbook:   https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftSpec:
    """Speculative-decoding draft model (MTP).

    Most Gemma 4 targets ship their own ``-assistant`` drafter at
    ~1.5-1.7% of target params, consumed via SGLang's NEXTN algorithm and
    ``--speculative-draft-model-path``. Sizes without a published drafter
    (e.g. 12B) use ``draft=None`` in their :class:`ModelSpec`.
    """

    hf_repo: str
    hf_revision: str | None
    notes: str = ""


@dataclass(frozen=True)
class ModelSpec:
    short: str
    """Short CLI key (app names, bench labels)."""

    hf_repo: str
    """HF repo loaded by the runtime. BF16 only — no official Google
    quantization is published (third-party FP8/INT4 variants exist but are
    model-specific)."""

    hf_revision: str | None
    """Pinned commit SHA, or None for ``main``. SGLang's CUDA-graph
    capture re-runs on any weight change, so a moving ``main`` would
    silently invalidate captured graphs across redeploys — pin a SHA once
    a variant has had a clean deploy run."""

    default_gpu: str
    """Default Modal GPU class for the model. Each serve script can
    override (e.g. a solo shape bumping to a larger card). For multi-GPU
    deployments the GPU spec uses Modal's ``:N`` suffix, e.g. 'B200:2'."""

    native_max_model_len: int
    """The model's native context ceiling (``max_position_embeddings``)."""

    draft: DraftSpec | None
    """The model's MTP drafter — same family, separate HF repo. ``None``
    when no drafter checkpoint is published (e.g. 12B)."""

    notes: str = ""


# ── Pinned revisions ─────────────────────────────────────────────────────
# Pin to specific commits a variant has been validated against; bump
# deliberately alongside a re-run of the chat-template conformance suite
# and a benchmark sweep. Left at None ("main") until first pin.

HF_REVISION_E2B: str | None = None
DRAFT_REVISION_E2B: str | None = None
HF_REVISION_E4B: str | None = None
DRAFT_REVISION_E4B: str | None = None
HF_REVISION_12B: str | None = None
HF_REVISION_26B: str | None = None
DRAFT_REVISION_26B: str | None = None
HF_REVISION_31B: str | None = None
DRAFT_REVISION_31B: str | None = None


REGISTRY: dict[str, ModelSpec] = {
    # ── Gemma 4 E2B-it (Dense+PLE, smallest member, multimodal) ──────────
    "e2b": ModelSpec(
        short="e2b",
        hf_repo="google/gemma-4-E2B-it",
        hf_revision=HF_REVISION_E2B,
        # L4-class is the cost-effective home for the smallest member.
        # Text-only inference uses well under half a 24 GiB L4.
        default_gpu="L4",
        native_max_model_len=128_000,
        draft=DraftSpec(
            hf_repo="google/gemma-4-E2B-it-assistant",
            hf_revision=DRAFT_REVISION_E2B,
            notes=(
                "~78M NEXTN drafter for E2B, sharing input embeddings with "
                "the target. Consistent with the family's drafter sizing."
            ),
        ),
        notes=(
            "Dense + Per-Layer Embeddings (PLE). Smallest member: 5.1B "
            "total params, ~2B effective via PLE. Multimodal (text + "
            "vision + audio towers) but served text-only. ~10 GiB BF16 "
            "weights — fits comfortably on an L4 (24 GiB). Native 128K "
            "context."
        ),
    ),
    # ── Gemma 4 E4B-it (Dense+PLE, multimodal incl audio) ────────────────
    "e4b": ModelSpec(
        short="e4b",
        hf_repo="google/gemma-4-E4B-it",
        hf_revision=HF_REVISION_E4B,
        # Default GPU is L40S — the concurrent/router shape. A solo dev
        # shape can override to a smaller card. The cookbook command shows
        # B200 as the validated platform but the model is small enough that
        # mid-tier GPUs are far more cost-effective for the router workload.
        default_gpu="L40S",
        native_max_model_len=128_000,
        draft=DraftSpec(
            hf_repo="google/gemma-4-E4B-it-assistant",
            hf_revision=DRAFT_REVISION_E4B,
            notes=(
                "78.8M NEXTN drafter for E4B. ~1.7% of target params, "
                "consistent with the family's drafter sizing."
            ),
        ),
        notes=(
            "Dense + Per-Layer Embeddings (PLE). 42 layers, sliding-512 "
            "(half the 31B/26B window), final-layer global. ~8B total "
            "params, 4.5B effective. ~16 GiB BF16 weights. Native 128K "
            "context (smaller than 31B/26B). Multimodal incl audio "
            "(~150M vision + ~300M audio encoders). Served text-only as "
            "a fast info-router for the agentic harness."
        ),
    ),
    # ── Gemma 4 12B-it (dense, no published drafter) ─────────────────────
    "12b": ModelSpec(
        short="12b",
        hf_repo="google/gemma-4-12B-it",
        hf_revision=HF_REVISION_12B,
        # bf16 weights ~24 GiB leave ample room for a large 256K KV
        # allocation for one stream on an 80GB card. H100 80GB (or A100
        # 80GB) is the comfortable solo class with long-context headroom;
        # an L40S 48GB also works for solo. For higher concurrency,
        # override to H200 141GB (or 'H100:2').
        default_gpu="H100",
        native_max_model_len=262_144,
        # No published MTP/speculative drafter checkpoint exists for 12B
        # (no google/gemma-4-12B-it-assistant; config has no MTP/nextn
        # head). Speculative decoding would require a separately
        # sourced/trained draft model — leave draft=None until then.
        draft=None,
        notes=(
            "Dense ~11.95B (config publishes no total param count; "
            "num_hidden_layers=48, hidden_size=3840, num_attention_heads="
            "16, num_key_value_heads=8, vocab_size=262144 confirm the ~12B "
            "class). model_type=gemma4_unified. head_dim=256 (fixed, NOT "
            "hidden_size/num_heads=240). Hybrid local/global: 5x "
            "sliding_attention (sliding_window=1024) then 1x full_attention "
            "across 48 layers (5:1 local:global), distinct RoPE for "
            "full vs sliding layers. Native 262K (256K) context. ~24 GiB "
            "BF16 weights. Multimodal (vision + audio) checkpoint served "
            "text-only. Triton attention backend required (same hybrid "
            "scheme + head_dim=256 as the rest of Gemma 4). No published "
            "drafter, so served without speculative decoding."
        ),
    ),
    # ── Gemma 4 26B-A4B-it (MoE, 3.8B active / 25.2B total) ──────────────
    "26b": ModelSpec(
        short="26b",
        hf_repo="google/gemma-4-26B-A4B-it",
        hf_revision=HF_REVISION_26B,
        # Cookbook recipe is 2xB200 TP=2 — MoE all-to-all dispatch wants
        # expert parallelism across two GPUs. Modal multi-GPU syntax `:2`.
        default_gpu="B200:2",
        native_max_model_len=256_000,
        draft=DraftSpec(
            hf_repo="google/gemma-4-26B-A4B-it-assistant",
            hf_revision=DRAFT_REVISION_26B,
            notes=(
                "~0.4B NEXTN drafter for the 26B-A4B target, sliding 1024 "
                "inherited from base."
            ),
        ),
        notes=(
            "MoE, 30 layers, 128 experts top-8, sliding-1024 + final-layer "
            "global. ~25.2B total / 3.8B active. ~50 GiB BF16 weights. "
            "Native 256K context. Multimodal checkpoint (image+video, no "
            "audio) served text-only. Cookbook calls for TP=2 on B200."
        ),
    ),
    # ── Gemma 4 31B-it (dense, the reasoning workhorse) ──────────────────
    "31b": ModelSpec(
        short="31b",
        hf_repo="google/gemma-4-31B-it",
        hf_revision=HF_REVISION_31B,
        # `B200+` opts into B300 when available, falls back to B200, bills
        # as B200 regardless. cu130 SGLang image satisfies B300 CUDA req.
        default_gpu="B200+",
        native_max_model_len=262_144,
        draft=DraftSpec(
            hf_repo="google/gemma-4-31B-it-assistant",
            hf_revision=DRAFT_REVISION_31B,
            notes=(
                "~0.5B 4-layer Gemma4AssistantForCausalLM. NEXTN drafter "
                "for the 31B target."
            ),
        ),
        notes=(
            "Dense 31B, hybrid softmax attention (50 sliding-1024 + 10 "
            "global, 5:1). Native 262K. BF16 weights (~62-66 GiB, 2 "
            "shards). Multimodal checkpoint served text-only. "
            "final_logit_softcapping=30."
        ),
    ),
}


def get(short: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by short name.

    Raises a friendly ``KeyError`` listing the valid options instead of an
    opaque miss when the caller passes a name we don't recognise.
    """
    try:
        return REGISTRY[short]
    except KeyError as e:
        valid = ", ".join(REGISTRY.keys())
        raise KeyError(
            f"unknown model {short!r}; expected one of: {valid}"
        ) from e
