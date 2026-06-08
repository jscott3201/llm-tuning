"""Shared SGLang image + launch-command builder for Qwen3.6.

This module backs both the dense-hybrid 27B and the MoE 35B-A3B
deployments. The two models share the same SGLang runtime, image, and
speculative-decoding machinery; only a handful of per-deployment knobs
(passed into ``build_serve_cmd``) differ.

Design choices:

  - **Base image is the upstream ``lmsysorg/sglang`` Docker tag**, not a
    raw nvidia/cuda devel image + pip install. The upstream image ships
    the full CUDA toolkit and the flash-attn-4 / deep_gemm runtime
    pre-wired, which sidesteps:
      * deep_gemm's ``_find_cuda_home()`` assert (needs nvcc + headers)
      * flash-attn-4 pre-release pin friction with uv
      * libnuma1 missing from nvidia/cuda minimal images
    Bump ``SGLANG_TAG`` when LMSYS publishes a new tag with your target
    SGLang version.

  - **Algorithm name = "EAGLE"** (not "NEXTN"). Both route to the same
    spec-v2 path internally for Qwen3.6, but EAGLE is what the current
    SGLang cookbook documents — staying consistent reduces "which name
    do I read?" friction.

  - **V1 vs V2 is selected purely by the ``--mamba-scheduler-strategy``
    string** (``extra_buffer`` => V2); there is NO silent fallback to V1.
    ``--page-size 64`` is forced independently by ``trtllm_mha``'s paged
    MHA. FLA only requires page_size and chunk-size to be mutually
    divisible (1/16/32/64 all valid); a non-divisor page size hard-errors.

  - **B200 vision path needs ``--mm-attention-backend fa4``** (FA4 only
    on Blackwell; on Hopper use ``fa3``). Both Qwen3.6-27B and 35B-A3B are
    multimodal VLMs (vision_config present), so the vision tower loads
    regardless — SGLang has no single-flag vision skip. Text-only agent
    clients simply never send image inputs.

  - **The server binds host ``0.0.0.0``** (the ``build_serve_cmd``
    default) so Modal's web endpoint can reach it. Under Modal,
    ``@modal.web_server(port=8000, ...)`` publishes a public ``*.modal.run``
    URL that forwards to the container's port 8000; that public URL is the
    intended ingress.

References
----------
- LMSYS cookbook: https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6
- SGLang releases: https://github.com/sgl-project/sglang/releases
"""

from __future__ import annotations

import modal

# Upstream SGLang Docker tag. Keep this in lockstep with what LMSYS
# publishes for Qwen3.6 cookbook recipes. Pinned by immutable digest
# below for reproducibility (the tag value is kept for humans).
SGLANG_TAG = "v0.5.12-cu130-runtime"
"""LMSYS pre-built image. cu130 = CUDA 13.0 (B300-capable). The
``-runtime`` variant ships the libs needed by SGLang's import chain
without the heavier devel toolkit. Bump when LMSYS publishes a newer tag
(and re-pin the digest in make_sglang_image).

Note: LMSYS dropped the ``-amd64-`` segment from their tag scheme around
0.5.10 — older Modal examples reference tags like
``v0.5.9-cu129-amd64-runtime`` which no longer exist."""

# Immutable content digest for SGLANG_TAG (linux/amd64 manifest), resolved
# via the Docker Hub registry API. Pinning by digest makes the image build
# reproducible even if LMSYS re-pushes the tag.
SGLANG_DIGEST = "sha256:7de5f60ce864919b15af674de1f1b0223121ee42e83bb58f4f3aee16fb18ccfd"


def make_sglang_image(
    sglang_tag: str = SGLANG_TAG,
    *,
    extra_packages: list[str] | None = None,
) -> modal.Image:
    """Canonical SGLang image for Qwen3.6.

    Pulled from ``lmsysorg/sglang`` rather than built from CUDA devel
    because the upstream image:
      - Ships nvcc + CUDA headers
      - Ships libnuma1 (sgl_kernel sm100 ops dlopen succeeds)
      - Bundles flash-attn-4 / FlashInfer at their tested versions for
        this SGLang release
      - Has ``entrypoint=[]`` overridden so we can run our own command.

    Note: do NOT call ``add_local_*`` inside this function. Modal forbids
    further build steps after a local-file addition (unless ``copy=True``).
    The deployment script should add ``add_local_python_source("_common")``
    as the FINAL build step, on top of the image returned here.
    """
    pkgs = [
        "huggingface_hub[hf_xet]>=1.11",
        "hf-transfer>=0.1.8",
        # The lmsysorg/sglang image ships a slim openai install that
        # omits the `distro` runtime dep — newer openai SDKs need it
        # at import time, and SGLang's function_call.function_call_parser
        # transitively imports openai during launch. Without distro,
        # `python -m sglang.launch_server` crashes with
        # ModuleNotFoundError before binding any port.
        "distro>=1.9",
    ]
    if extra_packages:
        pkgs.extend(extra_packages)

    # Pin by immutable digest for the canonical tag so rebuilds are
    # reproducible even if LMSYS re-pushes the tag. If a caller passes a
    # different tag, fall back to tag-based addressing for that override.
    if sglang_tag == SGLANG_TAG:
        image_ref = f"lmsysorg/sglang@{SGLANG_DIGEST}"  # == {SGLANG_TAG}
    else:
        image_ref = f"lmsysorg/sglang:{sglang_tag}"
    return (
        modal.Image.from_registry(image_ref)
        # entrypoint=[] silences the chatty default startup so the serve
        # function runs `python -m sglang.launch_server` directly.
        .entrypoint([])
        # uv_pip_install is materially faster than pip_install for image
        # builds — parallel resolver + downloader. Modal recommends it
        # as the default. pre=True lets us pick up any pre-release
        # wheels that newer SGLang patches occasionally depend on.
        .uv_pip_install(*pkgs, pre=True)
        .env(
            {
                # Saturate Modal egress on first HF weight pull. The
                # OLD knob HF_HUB_ENABLE_HF_TRANSFER is deprecated by
                # huggingface_hub; HF_XET_HIGH_PERFORMANCE is the
                # current replacement and uses the Xet transfer path.
                "HF_XET_HIGH_PERFORMANCE": "1",
                # Spec-V2 is the supported MTP path for Qwen3.6.
                "SGLANG_ENABLE_SPEC_V2": "1",
                # IMPORTANT: mount paths must be EMPTY in the base
                # image. The lmsysorg/sglang image pre-populates
                # /root/.cache/sglang with kernels, so we steer caches
                # to fresh paths under /modal-cache/ via env vars
                # instead. SGLang reads HF_HUB_CACHE.
                "HF_HUB_CACHE": "/modal-cache/huggingface",
                # torch.compile artifact cache. Without setting this,
                # the cache lives under /tmp and is lost between
                # container restarts — costing 1-3 min of JIT every
                # cold boot. Pointing it at a Volume preserves it.
                "TORCHINDUCTOR_CACHE_DIR": "/modal-cache/torchinductor",
                # expandable_segments lets PyTorch's allocator grow
                # its pools without pre-reserving large contiguous
                # chunks, reducing fragmentation. The first warmup
                # cold-boot hit a fragmentation-induced OOM (the
                # error message itself recommended this).
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Speculative-decoding profiles
# ─────────────────────────────────────────────────────────────────────
#
# Recipe semantics (SGLang flags):
#
#   num_steps          — drafts proposed per target forward
#   eagle_topk         — branching factor (1 = linear chain, >1 = tree)
#   num_draft_tokens   — total draft tokens (>= num_steps)
#
# IMPORTANT: on THIS stack (--attention-backend trtllm_mha + --page-size
# 64 + SGLANG_ENABLE_SPEC_V2=1), only LINEAR CHAIN profiles (eagle_topk=1)
# are runnable. Tree-verify (eagle_topk > 1) raises a hard ValueError at
# startup, so the only valid profiles are the two below. build_serve_cmd
# also guards against topk>1 to fail fast and clearly.

LATENCY = {
    # Fastest single-stream throughput. Linear chain, lowest verification
    # overhead. Best at low concurrency or when one stream dominates.
    "algorithm": "EAGLE",
    "num_steps": 3,
    "eagle_topk": 1,
    "num_draft_tokens": 4,
}

AGGRESSIVE_LINEAR = {
    # Linear chain pushed further. Cookbook LATENCY is num_steps=3 /
    # draft=4; this stretches to num_steps=5 / draft=5. At bs=1 the GPU
    # is memory-bandwidth-bound (weight loading dominates), so target
    # verification of a longer chain costs almost nothing extra — every
    # additional accepted draft token is nearly free. Best when:
    #   - concurrency is low (bs 1-3)
    #   - workload is highly predictable (coding, structured output)
    # Worse than LATENCY when acceptance rate drops below ~70%.
    "algorithm": "EAGLE",
    "num_steps": 5,
    "eagle_topk": 1,
    "num_draft_tokens": 5,
}


# ─────────────────────────────────────────────────────────────────────
# Launch command builder
# ─────────────────────────────────────────────────────────────────────


def build_serve_cmd(
    model_path: str,
    served_model_names: list[str],
    *,
    max_model_len: int,
    mem_fraction_static: float,
    chunked_prefill_size: int,
    max_running_requests: int,
    speculative_config: dict | None,
    kv_cache_dtype: str | None = None,
    attention_backend: str | None = "trtllm_mha",
    mm_attention_backend: str | None = None,
    mamba_scheduler_strategy: str | None = "extra_buffer",
    page_size: int | None = 64,
    tool_call_parser: str | None = "qwen3_coder",
    reasoning_parser: str | None = "qwen3",
    chat_template: str | None = None,
    # Solo-tuning knobs (all default off so the concurrent deployment
    # doesn't pick them up by accident).
    enable_fp32_lm_head: bool = False,
    enable_torch_compile: bool = False,
    torch_compile_max_bs: int | None = None,
    cuda_graph_bs: list[int] | None = None,
    num_continuous_decode_steps: int | None = None,
    enable_mixed_chunk: bool = False,
    max_prefill_tokens: int | None = None,
    # Observability — surfaces per-request stats + a Prometheus /metrics
    # endpoint that the bench script scrapes for MTP acceptance rate,
    # prefix-cache hit rate, KV-cache occupancy, etc.
    enable_metrics: bool = False,
    enable_request_time_stats: bool = False,
    log_requests: bool = False,
    log_requests_level: int | None = None,
    # Skip SGLang's boot-time warmup request. Its default warmup uses
    # `modalities=['image']` which probes the vision path even when
    # the user only sends text — wasting ~5 GiB of FP32 LM head scratch
    # on a code path real traffic will never hit. Real client requests
    # still go through the full pipeline; we just skip the redundant
    # boot probe.
    skip_server_warmup: bool = False,
    # Optional bearer-token auth, OFF by default. If you set this env var
    # in the container (e.g. via a modal.Secret), SGLang will require it as
    # the API key; otherwise the server is open and you should rely on
    # Modal's own endpoint protection instead. See Modal's web-endpoint
    # security docs: https://modal.com/docs/guide/webhooks#security
    api_key_env: str | None = "API_KEY",
    # Bind 0.0.0.0 so Modal's web endpoint (the public *.modal.run URL
    # published by @modal.web_server) can reach the server. 0.0.0.0 accepts
    # on every interface, so the in-container /health probe still works
    # over loopback.
    host: str = "0.0.0.0",
    port: int = 8000,
    revision: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the ``python -m sglang.launch_server`` argv.

    Defaults track the current LMSYS Qwen3.6 cookbook for B200. Two
    things that often surprise people:

    - ``page_size=64`` is forced by ``trtllm_mha``'s paged MHA (not by the
      Mamba scheduler). V1 vs V2 is selected purely by the
      ``mamba_scheduler_strategy`` string (``extra_buffer`` => V2); there is
      no silent V1 fallback. FLA only needs page_size and chunk-size to be
      mutually divisible; a non-divisor page size hard-errors.
    - ``host="0.0.0.0"`` so Modal's web endpoint can route public traffic
      to the container's port. The ``@modal.web_server`` decorator on the
      serve function publishes the ``*.modal.run`` URL that forwards here.
    """
    import os

    # Guard: tree-verify (eagle_topk > 1) raises a hard ValueError at
    # startup on this stack (--attention-backend trtllm_mha +
    # SGLANG_ENABLE_SPEC_V2 + --page-size 64). Only linear-chain
    # profiles (eagle_topk == 1) are runnable; fail fast and clearly
    # rather than letting the server crash mid-boot.
    if speculative_config is not None and speculative_config.get("eagle_topk", 1) != 1:
        raise ValueError(
            "speculative_config.eagle_topk must be 1 (linear chain). "
            "Tree-verify (eagle_topk > 1) fails at startup on the "
            "trtllm_mha + SGLANG_ENABLE_SPEC_V2 + page_size 64 stack. "
            "Use the LATENCY or AGGRESSIVE_LINEAR profile."
        )

    # SGLang 0.5.12 expects a single value for --served-model-name. If
    # callers pass multiple aliases we keep the first; the model is
    # also reachable by its HF repo via the --model-path argument.
    served_name = served_model_names[0] if served_model_names else model_path
    cmd: list[str] = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        served_name,
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(max_model_len),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--max-running-requests",
        str(max_running_requests),
    ]

    if revision:
        cmd += ["--revision", revision]

    if reasoning_parser is not None:
        cmd += ["--reasoning-parser", reasoning_parser]

    if tool_call_parser is not None:
        cmd += ["--tool-call-parser", tool_call_parser]

    # Custom chat template baked into the image via add_local_file(copy=True).
    # When None, SGLang falls back to the model's built-in tokenizer template.
    if chat_template is not None:
        cmd += ["--chat-template", chat_template]

    # There is no single-flag vision skip in SGLang. `--disable-mm`
    # never existed; `--language-only` is part of the disaggregated EPD
    # subsystem (it expects a separate vision-encoder container at
    # `--encoder-urls`), not a "skip vision tower" switch. Both
    # Qwen3.6-27B and 35B-A3B are multimodal VLMs, so the vision tower
    # loads regardless (~1-2 GiB); text-only agent clients simply never
    # send image inputs.
    if mm_attention_backend is not None:
        # Only meaningful when MM is enabled. fa4 on Blackwell, fa3 on
        # Hopper. The cookbook recommends both.
        cmd += ["--mm-attention-backend", mm_attention_backend]

    if attention_backend is not None:
        cmd += ["--attention-backend", attention_backend]

    if mamba_scheduler_strategy is not None:
        cmd += ["--mamba-scheduler-strategy", mamba_scheduler_strategy]
        if page_size is not None:
            cmd += ["--page-size", str(page_size)]

    if speculative_config is not None:
        cmd += ["--speculative-algorithm", speculative_config["algorithm"]]
        if "num_steps" in speculative_config:
            cmd += ["--speculative-num-steps", str(speculative_config["num_steps"])]
        if "eagle_topk" in speculative_config:
            cmd += ["--speculative-eagle-topk", str(speculative_config["eagle_topk"])]
        if "num_draft_tokens" in speculative_config:
            cmd += [
                "--speculative-num-draft-tokens",
                str(speculative_config["num_draft_tokens"]),
            ]

    if kv_cache_dtype is not None:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]

    # ── Solo-only precision + perf knobs ────────────────────────────────
    if enable_fp32_lm_head:
        # Keep just the final logit projection in FP32. Tiny cost (one
        # matmul out of ~64 layers), real precision win at the most
        # numerically sensitive step (logits → softmax).
        cmd.append("--enable-fp32-lm-head")

    if enable_torch_compile:
        # torch.compile fuses adjacent ops into single kernels — biggest
        # impact at small batch sizes where per-launch overhead is a
        # larger fraction of total time. Cold-start adds 1-3 min on
        # first deploy as the JIT traces shapes; cached in
        # /root/.cache/sglang volume afterwards.
        cmd.append("--enable-torch-compile")
        if torch_compile_max_bs is not None:
            # Pin the trace ceiling. Default is 32 which causes JIT to
            # trace shapes we'll never hit.
            cmd += ["--torch-compile-max-bs", str(torch_compile_max_bs)]

    if cuda_graph_bs is not None:
        # Enumerate exact decode batch sizes for CUDA graph capture.
        # Faster cold start than auto-discovery and ensures all the
        # bs values we actually use are captured.
        cmd += ["--cuda-graph-bs", *[str(b) for b in cuda_graph_bs]]

    if num_continuous_decode_steps is not None:
        # Run N decode steps per scheduler tick to reduce CPU-side
        # scheduling overhead. With MTP, each step generates up to
        # num_draft_tokens of output, so this multiplies effective
        # tokens-per-scheduler-hop.
        cmd += ["--num-continuous-decode-steps", str(num_continuous_decode_steps)]

    if enable_mixed_chunk:
        # Allow the scheduler to pack decode tokens of completed-prefill
        # streams INTO the prefill chunks of new streams. Cheap on this
        # hybrid arch because the linear-attention layers' decode is
        # near-free.
        cmd.append("--enable-mixed-chunk")

    if max_prefill_tokens is not None:
        # Upper bound on tokens in a single prefill batch. Usually want
        # to align with chunked_prefill_size — they bound related but
        # not identical quantities (per-chunk vs per-batch).
        cmd += ["--max-prefill-tokens", str(max_prefill_tokens)]

    # ── Observability ──────────────────────────────────────────────────
    if enable_metrics:
        # Exposes /metrics on the same port (Prometheus format). The
        # bench script scrapes this for spec_decode_acceptance_rate,
        # prefix_cache_hit_rate, num_running_reqs, kv_cache_usage, etc.
        cmd.append("--enable-metrics")
    if enable_request_time_stats:
        # Per-request prefill / decode / queue timings logged to stdout
        # at request completion. Used to attribute latency to phase.
        cmd.append("--enable-request-time-stats-logging")
    if log_requests:
        cmd.append("--log-requests")
        if log_requests_level is not None:
            # 0=metadata, 1=+sampling params, 2=+partial I/O, 3=full
            cmd += ["--log-requests-level", str(log_requests_level)]

    if skip_server_warmup:
        cmd.append("--skip-server-warmup")

    # Optional auth: only adds --api-key when the named env var is actually
    # set in the container. Left unset, the server is open and you should
    # protect it with Modal's endpoint security instead (see api_key_env
    # docstring). This is intentionally opt-in, not required.
    if api_key_env and os.environ.get(api_key_env):
        cmd += ["--api-key", os.environ[api_key_env]]

    if extra_args:
        cmd += extra_args

    return cmd
