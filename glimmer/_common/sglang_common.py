"""Shared SGLang image + launch-command builder for Muse Glimmer.

Design choices
--------------
- **Base image is the upstream ``lmsysorg/sglang`` release image**, pinned
  by digest. Glimmer support (model, ``muse`` reasoning/tool parsers,
  DFlash drafter) landed in SGLang via PR #34262 and ships in every
  release from v0.5.18. The cookbook still points at a
  ``dev-muse-glimmer`` branch image; a release tag is the better pin.

- **CUDA 13.0 image, so no B300.** Modal's B300 needs CUDA 13.1+, and
  LMSYS publishes v0.5.22 only as ``cu130``. Deployments therefore request
  ``B200`` or ``RTX-PRO-6000`` explicitly, never ``B200+``.

- **DFlash speculative decoding is on by default.** The drafter proposes a
  16-token block per target forward; the target verifies the block in one
  pass. Output distribution is unchanged (it is exact speculative
  sampling), only latency moves.

- **The server binds ``0.0.0.0``** so Modal's ``@modal.web_server``
  ingress (the public ``*.modal.run`` URL) can reach it. The health and
  warmup helpers talk to it over loopback.

References
----------
- SGLang Glimmer cookbook: https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer
- SGLang releases:         https://github.com/sgl-project/sglang/releases
- DFlash paper:            https://arxiv.org/abs/2602.06036
"""

from __future__ import annotations

import modal

SGLANG_TAG = "v0.5.22-cu130-runtime"
"""LMSYS release image. Bump together with ``SGLANG_DIGEST``."""

SGLANG_DIGEST = "sha256:c3d99f078c329a5b2d0c0c02c4237cf8b0dcd1dc6f875a943ae163c1720854cb"
"""Immutable linux/amd64 digest for ``SGLANG_TAG`` (Docker Hub registry
API). Pinning by digest keeps rebuilds reproducible if the tag is
re-pushed."""


def make_sglang_image(
    sglang_tag: str = SGLANG_TAG,
    *,
    extra_packages: list[str] | None = None,
) -> modal.Image:
    """Canonical SGLang image for Muse Glimmer.

    Do NOT call ``add_local_*`` in here: Modal forbids further build steps
    after a non-copy local add, so the serve script adds
    ``add_local_python_source("_common")`` as its final step.
    """
    pkgs = [
        "huggingface_hub[hf_xet]>=1.11",
        # The image's slim openai install omits `distro`, which SGLang's
        # function-call parser chain imports at launch.
        "distro>=1.9",
    ]
    if extra_packages:
        pkgs.extend(extra_packages)

    if sglang_tag == SGLANG_TAG:
        image_ref = f"lmsysorg/sglang@{SGLANG_DIGEST}"
    else:
        image_ref = f"lmsysorg/sglang:{sglang_tag}"

    return (
        modal.Image.from_registry(image_ref)
        .entrypoint([])
        .uv_pip_install(*pkgs, pre=True)
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                # Mount paths must be empty in the base image, so caches
                # live under /modal-cache/ rather than ~/.cache.
                "HF_HUB_CACHE": "/modal-cache/huggingface",
                # No PYTORCH_CUDA_ALLOC_CONF=expandable_segments here (the
                # qwen/ image sets it): torch_memory_saver, which the
                # snapshot path needs, refuses to start with it.
            }
        )
    )


# ─────────────────────────────────────────────────────────────────────
# DFlash profiles
# ─────────────────────────────────────────────────────────────────────
#
# DFlash has one knob that matters: the block size (tokens drafted and
# verified per target forward). SGLang reads 16 from the drafter config
# when unset. A smaller block wastes less verify compute when acceptance
# is low or the batch is large; a larger block wins at batch 1 where the
# GPU is bandwidth-bound and extra verify tokens are nearly free.

DFLASH_OFF = None
"""Plain autoregressive decoding. For A/B benchmarks and as a one-line
revert if a future SGLang bump breaks the drafter."""

DFLASH_16 = {"algorithm": "DFLASH", "block_size": 16}
"""The drafter's trained block size. Best single-stream latency."""

DFLASH_8 = {"algorithm": "DFLASH", "block_size": 8}
"""Half block. Less verify work per step for busy, many-session GPUs.

Measured (README): best aggregate throughput at 4-16 concurrent requests;
at 64 it trails DFLASH_OFF by ~11% (RTX PRO 6000) to ~28% (B200)."""


# ─────────────────────────────────────────────────────────────────────
# Launch command builder
# ─────────────────────────────────────────────────────────────────────


def build_serve_cmd(
    model_path: str,
    served_model_name: str,
    *,
    context_length: int,
    mem_fraction_static: float,
    chunked_prefill_size: int,
    max_running_requests: int,
    revision: str | None = None,
    speculative_config: dict | None = None,
    draft_model_path: str | None = None,
    draft_revision: str | None = None,
    draft_quantization: str | None = None,
    kv_cache_dtype: str | None = "fp8_e4m3",
    attention_backend: str | None = None,
    swa_full_tokens_ratio: float | None = None,
    language_model_only: bool = False,
    mm_attention_backend: str | None = None,
    cuda_graph_bs: list[int] | None = None,
    max_prefill_tokens: int | None = None,
    tool_call_parser: str = "muse",
    reasoning_parser: str = "muse",
    chat_template: str | None = None,
    enable_memory_saver: bool = False,
    enable_weights_cpu_backup: bool = False,
    enable_metrics: bool = True,
    enable_request_time_stats: bool = True,
    enable_cache_report: bool = True,
    log_requests: bool = True,
    log_requests_level: int | None = 1,
    skip_server_warmup: bool = True,
    api_key_env: str | None = "API_KEY",
    host: str = "0.0.0.0",
    port: int = 8000,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the ``python -m sglang.launch_server`` argv.

    Model- and shape-agnostic: every number comes from the caller's
    constants block.
    """
    import os

    if speculative_config is not None:
        if speculative_config.get("algorithm") != "DFLASH":
            raise ValueError(
                "speculative_config must be a DFLASH profile from "
                "sglang_common (DFLASH_16 / DFLASH_8) or DFLASH_OFF (None)."
            )
        if not draft_model_path:
            raise ValueError("DFLASH needs draft_model_path (ModelSpec.draft.hf_repo).")

    if cuda_graph_bs is not None and max(cuda_graph_bs) < max_running_requests:
        raise ValueError(
            f"cuda_graph_bs tops out at {max(cuda_graph_bs)} but "
            f"max_running_requests={max_running_requests}; decode batches "
            "above the largest captured size fall back to eager mode."
        )

    cmd: list[str] = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(context_length),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--max-running-requests",
        str(max_running_requests),
        "--tool-call-parser",
        tool_call_parser,
        "--reasoning-parser",
        reasoning_parser,
    ]

    if revision:
        cmd += ["--revision", revision]

    if chat_template:
        cmd += ["--chat-template", chat_template]

    if kv_cache_dtype:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]

    if attention_backend:
        cmd += ["--attention-backend", attention_backend]

    if swa_full_tokens_ratio is not None:
        # Size of the sliding-window KV pool relative to the full one. The
        # 39 sliding layers need only ~window tokens per running request,
        # so a small ratio frees memory for full-context KV.
        cmd += ["--swa-full-tokens-ratio", str(swa_full_tokens_ratio)]

    if language_model_only:
        # Skips building/loading the perception encoder and rejects image
        # requests. Frees ~4 GB; only worth it on tight cards.
        cmd.append("--language-model-only")
    elif mm_attention_backend:
        cmd += ["--mm-attention-backend", mm_attention_backend]

    if speculative_config is not None:
        cmd += [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            draft_model_path,
        ]
        if draft_revision:
            cmd += ["--speculative-draft-model-revision", draft_revision]
        if draft_quantization:
            cmd += ["--speculative-draft-model-quantization", draft_quantization]
        if speculative_config.get("block_size"):
            cmd += [
                "--speculative-dflash-block-size",
                str(speculative_config["block_size"]),
            ]

    if cuda_graph_bs is not None:
        # Renamed from --cuda-graph-bs in SGLang 0.5.2x.
        cmd += ["--cuda-graph-bs-decode", *[str(b) for b in cuda_graph_bs]]

    if max_prefill_tokens is not None:
        cmd += ["--max-prefill-tokens", str(max_prefill_tokens)]

    if enable_memory_saver:
        cmd.append("--enable-memory-saver")
    if enable_weights_cpu_backup:
        cmd.append("--enable-weights-cpu-backup")
        if speculative_config is not None:
            # The drafter has its own weights; back them up too so a
            # snapshot restore brings both models back.
            cmd.append("--enable-draft-weights-cpu-backup")

    if enable_metrics:
        cmd.append("--enable-metrics")
    if enable_request_time_stats:
        cmd.append("--enable-request-time-stats-logging")
    if enable_cache_report:
        cmd.append("--enable-cache-report")
    if log_requests:
        cmd.append("--log-requests")
        if log_requests_level is not None:
            cmd += ["--log-requests-level", str(log_requests_level)]

    if skip_server_warmup:
        cmd.append("--skip-server-warmup")

    # Opt-in bearer auth: only when the env var is set in the container
    # (e.g. from a modal.Secret). See docs/securing-endpoints.md.
    if api_key_env and os.environ.get(api_key_env):
        cmd += ["--api-key", os.environ[api_key_env]]

    if extra_args:
        cmd += extra_args

    return cmd
