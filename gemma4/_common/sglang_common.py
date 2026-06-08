"""Shared SGLang image + launch-command builder for the Gemma 4 family.

Design choices
--------------

  - **Base image is the upstream ``lmsysorg/sglang`` Docker tag**, not a
    raw nvidia/cuda image + pip install. The upstream image ships the CUDA
    toolkit and the flash-attn / FlashInfer runtime pre-wired, and — for
    Gemma 4 specifically — bundles a ``transformers`` new enough to know
    the ``gemma4`` model_type. SGLang v0.5.12 (released 2026-05-16) is the
    first STABLE release whose notes call out Gemma 4 support explicitly
    (MTP head #24436/#24433, Gemma 3/4 + EAGLE-3 #23976). Pinned for
    reproducibility; bump ``SGLANG_TAG`` deliberately.

  - **No Mamba / GDN / page-size args.** Gemma 4 is a SOFTMAX hybrid —
    sliding-window layers + global layers — with no recurrent state cache,
    so the linear-attention hybrid args (mamba scheduler strategy, forced
    page size) do not apply. We also leave ``--page-size`` at the
    documented default of 1 ("leave at 1 for standard setups" per the
    SGLang server-args reference); the NEXTN constraint is on
    ``--speculative-eagle-topk=1``, which we enforce separately.

  - **``--attention-backend triton``.** Gemma 4's global-attention layers
    use a 512-wide head (``global_head_dim: 512``) with
    ``partial_rotary_factor: 0.25``, and a fixed ``head_dim=256``.
    FlashInfer rejects this geometry, and ``trtllm_mha`` also rejects
    head_dim=512. SGLang's Gemma 4 PR (sgl-project/sglang#21952) and the
    cookbook both auto-select Triton for Gemma 4 — required not just for
    the head_dim but also for bidirectional image-token attention during
    prefill. We pass it explicitly as defence against auto-detection drift
    across SGLang releases. **Failure mode if the wrong backend is
    selected: silent broken output** (empty content, repetition collapse,
    model-card "la la la" artefact, gemma-4-31B-it discussion #79). The
    boot health-check (see ``health.send_warmup_request``) catches this on
    the first real request.

  - **FP8 E5M2 KV-cache.** ``--kv-cache-dtype fp8_e5m2``. Explicitly
    confirmed working on 31B-it in sgl-project/sglang#22277 ("Gemma4 31B
    (num_kv_shared_layers: 0) + fp8_e5m2 ✅ Works, serves requests"). The
    FP8-KV crash in the same issue applies only to E4B
    (``num_kv_shared_layers: 18``), not 31B-it. The earlier "FP8 KV
    interacts badly with Gemma 4's wide global head" warning was
    specifically about the FA-family backends; under Triton it does not
    apply. Halves per-seq KV (~22 GB → ~11 GB at 256K) for negligible
    quality cost — vLLM's official FP8-KV benchmarking measured 0.7 pp
    worst-case delta on AIME25. Default bf16 KV is also fine and is the
    safe baseline for sizes that haven't been FP8-KV-validated.

  - **``gemma4`` tool-call + reasoning parsers.** SGLang ships dedicated
    ``--tool-call-parser gemma4`` and ``--reasoning-parser gemma4`` —
    purpose-built for Gemma 4's special-token wire format (``<|tool_call>``
    / ``<|channel>``), not reused from another model.

  - **Custom chat template.** Pass ``--chat-template <path>`` pointing at a
    baked-in fork of the upstream template
    (``chat_templates/custom_pub_chat_template_gemma4.jinja``). The
    verbatim upstream copies (``gemma4_upstream.jinja`` /
    ``gemma4_e4b_upstream.jinja``) are baked alongside the fork in the
    image for diff inspection. Pinning the model revision still pins the
    *upstream* template — the fork is git-tracked and diff-auditable
    against it. When ``chat_template`` is None, SGLang loads the model's
    own template from the pinned revision.

  - **MTP is ON by default where a drafter exists.** SGLang v0.5.12
    release notes explicitly list "Gemma 4 MTP #24436" as a supported new
    model. The family's ``-assistant`` drafters are shipped by Google under
    Apache-2.0 and work with the NEXTN algorithm at the cookbook recipe
    (num_steps=5, num_draft_tokens=6, eagle_topk=1). MTP is exact-output —
    it changes latency, not text — so the harness needs no changes when it
    is on. (12B ships no drafter, so it runs without speculative decoding.)

Known SGLang issues this module is built around
-----------------------------------------------
- **#25073 / #25099** — the ``gemma4`` tool-call parser mis-indexes
  streaming ``tool_calls`` when one function is called repeatedly; the fix
  PR was unmerged at v0.5.12's release, so v0.5.12 ships with the bug.
  Mitigation is harness-side: prefer non-streaming for multi-tool fanout.
- **#25545** — ``trtllm_mha`` for the MTP *draft* attention was still a WIP
  PR; the draft may fall back to Triton. Harmless (the draft is small, and
  we're on Triton for the target anyway).

References
----------
- SGLang Gemma 4 cookbook: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
- SGLang Gemma 4 PR:        https://github.com/sgl-project/sglang/pull/21952
- SGLang FP8-KV thread:     https://github.com/sgl-project/sglang/issues/22277
- SGLang releases:          https://github.com/sgl-project/sglang/releases
"""

from __future__ import annotations

import modal

# Upstream SGLang Docker tag. v0.5.12 = first stable release with Gemma 4 +
# MTP + EAGLE-3 in the notes. cu130 = CUDA 13.0 (satisfies B300's CUDA
# requirement). `-runtime` ships the libs SGLang's import chain needs
# without the heavier devel toolkit. Bump deliberately and re-run the
# chat-template conformance suite on any change.
#
# Note: the Gemma 4 cookbook also advertises convenience tags
# (`lmsysorg/sglang:cu13-gemma4`), but those are MOVING tags — we pin the
# versioned tag instead so a redeploy is byte-identical.
SGLANG_TAG = "v0.5.12-cu130-runtime"


def make_sglang_image(
    sglang_tag: str = SGLANG_TAG,
    *,
    extra_packages: list[str] | None = None,
) -> modal.Image:
    """Canonical SGLang image for the Gemma 4 family.

    Pulled from ``lmsysorg/sglang`` rather than built from a CUDA devel
    image because the upstream image already carries a Gemma-4-aware
    ``transformers``, the CUDA toolkit, and FlashInfer/flash-attn at their
    tested versions for this SGLang release.
    """
    pkgs = [
        "huggingface_hub[hf_xet]>=1.11",
        "hf-transfer>=0.1.8",
        # The lmsysorg/sglang image ships a slim openai install that omits
        # the `distro` runtime dep; newer openai SDKs import it, and
        # SGLang's function-call parser transitively imports openai during
        # launch. Without distro, `python -m sglang.launch_server` crashes
        # with ModuleNotFoundError before binding any port.
        "distro>=1.9",
    ]
    if extra_packages:
        pkgs.extend(extra_packages)

    # IMPORTANT: do NOT call `add_local_*` here. Modal forbids further
    # build steps after a non-copy local-file addition. The deployment
    # script adds chat-template `add_local_file(copy=True)` calls and
    # finally `add_local_python_source("_common")` as the LAST step.
    return (
        modal.Image.from_registry(f"lmsysorg/sglang:{sglang_tag}")
        # entrypoint=[] clears the lmsysorg/sglang image's chatty default
        # startup so Modal runs the function body directly.
        .entrypoint([])
        # uv_pip_install is materially faster than pip_install for image
        # builds (parallel resolver + downloader). pre=True picks up any
        # pre-release wheels recent SGLang patches occasionally need.
        .uv_pip_install(*pkgs, pre=True)
        .env(
            {
                # Saturate Modal egress on the first HF weight pull.
                # HF_XET_HIGH_PERFORMANCE is the current replacement for
                # the deprecated HF_HUB_ENABLE_HF_TRANSFER.
                "HF_XET_HIGH_PERFORMANCE": "1",
                # Steer the HF weight cache to a Volume-mounted path.
                # The base image pre-populates /root/.cache, so we use a
                # fresh path under /modal-cache/ (Modal Volume mount
                # points must be empty in the base image).
                "HF_HUB_CACHE": "/modal-cache/huggingface",
                # NOTE: TORCHINDUCTOR_CACHE_DIR was previously set here for
                # torch.compile artifact reuse. SGLang has since deprecated
                # --enable-torch-compile ("out of maintenance" per the
                # server-args reference) and the default deployments disable
                # it, so the env var was removed alongside the Modal Volume
                # mount. Re-add together with a `torchinductor-cache` volume
                # if torch.compile is ever re-enabled.
                # expandable_segments lets PyTorch's allocator grow pools
                # without pre-reserving large contiguous chunks, reducing
                # fragmentation-induced OOM at cold boot.
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                # Single-threaded TorchInductor at compile time. Required
                # for Modal Memory Snapshots to be reliably created when
                # --enable-torch-compile is on (multi-threaded compile
                # can race with the snapshotter). Harmless when
                # torch.compile is off (the default).
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                # NOTE on SGLANG_ENABLE_SPEC_V2: set per-deployment when
                # MTP is enabled (see serve script); v0.5.12+ defaults to V2
                # so the env var is defensive against future opt-in flips.
            }
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Speculative-decoding (MTP) profiles
# ─────────────────────────────────────────────────────────────────────
#
# Gemma 4's MTP drafter is the separate `google/<size>-it-assistant` repo
# (see model_registry.ModelSpec.draft), consumed via SGLang's NEXTN
# algorithm and `--speculative-draft-model-path`. (12B has no published
# drafter — it runs with MTP_OFF.)
#
# IMPORTANT — all profiles use `eagle_topk: 1` (a LINEAR draft chain).
# The Triton attention backend + NEXTN speculative decoding requires
# `--speculative-eagle-topk 1` (paged MHA constraint). Tree-verify
# profiles (topk 3/5) are therefore NOT available here — the profile space
# is linear-only, varying num_steps / num_draft_tokens. That happens to be
# the regime that already won on B200 at small batch sizes.

MTP_OFF = None
"""Speculative decoding disabled — exposed for benchmarking with/without
MTP, as a one-line revert if MTP misbehaves on a future SGLang bump, and as
the only option for sizes with no published drafter (e.g. 12B)."""

MTP_NEXTN_STANDARD = {
    # The SGLang Gemma 4 cookbook's published recipe verbatim. This is
    # the production-default profile.
    "algorithm": "NEXTN",
    "num_steps": 5,
    "num_draft_tokens": 6,
    "eagle_topk": 1,
}

MTP_LATENCY = {
    # Shorter linear chain — lower verification overhead, best when a
    # single stream dominates and acceptance is high. Benchmark before
    # preferring over NEXTN_STANDARD.
    "algorithm": "NEXTN",
    "num_steps": 3,
    "num_draft_tokens": 4,
    "eagle_topk": 1,
}

MTP_AGGRESSIVE = {
    # Longer linear chain. At bs=1 on B200 the GPU is bandwidth-bound, so
    # verifying a longer chain is nearly free — each extra accepted draft
    # token is a near-free win, IF the drafter's acceptance rate holds.
    # Benchmark acceptance length before preferring over NEXTN_STANDARD.
    "algorithm": "NEXTN",
    "num_steps": 6,
    "num_draft_tokens": 7,
    "eagle_topk": 1,
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
    tp_size: int = 1,
    revision: str | None = None,
    # Parsers — Gemma 4 defaults. Both always on for an agentic
    # tool-calling + thinking deployment.
    tool_call_parser: str = "gemma4",
    reasoning_parser: str = "gemma4",
    # Gemma 4's required backend. FlashInfer + trtllm_mha both reject
    # head_dim=512; Triton is what SGLang auto-selects (and we pin it
    # explicitly as a defence against auto-detection drift).
    attention_backend: str = "triton",
    # Custom chat template (absolute path inside the container image,
    # baked via add_local_file(copy=True) in the deployment script). When
    # None, SGLang loads the model's own template from the pinned revision.
    chat_template: str | None = None,
    # KV-cache page size — see module docstring. Left at the documented
    # default (1) unless a deployment has a specific reason to set it.
    page_size: int | None = None,
    # Speculative decoding (MTP). None = no spec decode. When set, pass a
    # profile dict from above AND draft_model_path.
    speculative_config: dict | None = None,
    draft_model_path: str | None = None,
    # Pins the draft model's commit (parallel to `revision` for the
    # target). Ignored unless speculative_config is set. `--speculative-
    # draft-model-revision` is a real v0.5.12 flag.
    draft_revision: str | None = None,
    # FP8 E5M2 KV — production-validated on 31B-it (sgl-project/sglang
    # #22277). Halves per-seq KV at negligible quality cost. Set to None
    # to bisect a suspected quality regression back to BF16 KV.
    kv_cache_dtype: str | None = None,
    # Memory-snapshot enablers for SGLang. When True, SGLang co-operates
    # with Modal's snapshot lifecycle: --enable-memory-saver lets the
    # server move active memory off-GPU on demand (via the
    # /release_memory_occupation endpoint, called pre-snapshot from
    # @modal.enter(snap=True)); --enable-weights-cpu-backup keeps a CPU
    # mirror of weights so the post-restore /resume_memory_occupation
    # can re-populate GPU memory without re-reading from disk. Default
    # off — only the snapshot-enabled deployments turn them on.
    enable_memory_saver: bool = False,
    enable_weights_cpu_backup: bool = False,
    # Solo-tuning knobs (default off so the concurrent deployment doesn't
    # pick them up by accident).
    enable_torch_compile: bool = False,
    torch_compile_max_bs: int | None = None,
    cuda_graph_bs: list[int] | None = None,
    num_continuous_decode_steps: int | None = None,
    max_prefill_tokens: int | None = None,
    # Observability — Prometheus /metrics + per-request stats. A bench
    # script scrapes /metrics for spec-decode acceptance, cache hit rate,
    # KV occupancy. On for both shapes.
    enable_metrics: bool = True,
    enable_request_time_stats: bool = True,
    log_requests: bool = True,
    log_requests_level: int | None = 1,
    # Returns per-request cached-token counts in usage.prompt_tokens_details
    # — finer-grained than scraping /metrics, and the dominant cost signal
    # for the prefix-sharing concurrent shape.
    enable_cache_report: bool = True,
    # SGLang's boot warmup probes the multimodal path. We serve text-only,
    # so skipping it saves a wasted boot step. Real client requests still
    # go through the full pipeline.
    skip_server_warmup: bool = True,
    # Optional API-key gate. Default off. If you want to require a key,
    # set this to the name of an env var holding the key (and inject it via
    # a modal.Secret). Leaving auth to the operator is intentional — Modal
    # has its own endpoint-security options (proxy auth tokens); see
    # https://modal.com/docs/guide/webhook-proxy-auth . When the named env
    # var is unset, no --api-key flag is added and the server is open.
    api_key_env: str | None = None,
    # Bind 0.0.0.0 (NOT 127.0.0.1) so Modal's @modal.web_server can reach
    # the process across the container's external interface. Under Modal
    # this publishes a public *.modal.run URL, which is the intended
    # ingress; pair with the optional API-key gate or Modal proxy auth if
    # you want to restrict access.
    host: str = "0.0.0.0",
    port: int = 8000,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the ``python -m sglang.launch_server`` argv for Gemma 4.

    ``host="0.0.0.0"`` (NOT ``127.0.0.1``): the server must bind the
    container's external interface so Modal's ``@modal.web_server`` ingress
    can route to it. Modal publishes a public ``*.modal.run`` URL for the
    endpoint.
    """
    import os

    # Guardrail: a speculative profile must name its algorithm.
    if speculative_config is not None and "algorithm" not in speculative_config:
        raise ValueError(
            "speculative_config must contain an 'algorithm' key; use a "
            "profile from sglang_common.py (MTP_NEXTN_STANDARD / "
            "MTP_LATENCY / MTP_AGGRESSIVE) or MTP_OFF (None)."
        )

    # Guardrail: NEXTN + paged MHA requires eagle_topk=1. Catch a
    # tree-verify profile at build time, not at boot.
    if (
        speculative_config is not None
        and speculative_config.get("eagle_topk", 1) != 1
    ):
        raise ValueError(
            "Gemma 4 NEXTN speculative decoding requires eagle_topk=1 "
            "(paged MHA constraint); got eagle_topk="
            f"{speculative_config.get('eagle_topk')}. Use a linear MTP "
            "profile (all profiles in sglang_common.py already are)."
        )
    if speculative_config is not None and not draft_model_path:
        raise ValueError(
            "speculative_config set but draft_model_path is missing — "
            "pass model_registry.ModelSpec.draft.hf_repo."
        )

    # SGLang expects a single --served-model-name; the model is also
    # reachable by its HF repo via --model-path.
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
        "--tp-size",
        str(tp_size),
        "--context-length",
        str(max_model_len),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--max-running-requests",
        str(max_running_requests),
        # Gemma 4 special-token parsers — see module docstring.
        "--tool-call-parser",
        tool_call_parser,
        "--reasoning-parser",
        reasoning_parser,
        # Required for Gemma 4's 512-wide global head — see docstring.
        "--attention-backend",
        attention_backend,
    ]

    if revision:
        # Pinning the revision pins the *upstream* chat_template.jinja
        # SGLang would otherwise load. We override that with --chat-template
        # (next block) pointing at the baked-in fork; the upstream pin
        # remains useful for everything else (weights, generation config).
        cmd += ["--revision", revision]

    if chat_template:
        # Absolute path inside the container image. The deployment script
        # bakes the .jinja file via add_local_file(copy=True). SGLang's
        # OpenAI-compat layer (conversation.py) loads it at boot.
        cmd += ["--chat-template", chat_template]

    if page_size is not None:
        cmd += ["--page-size", str(page_size)]

    # ── Speculative decoding (MTP) ──────────────────────────────────────
    if speculative_config is not None:
        cmd += [
            "--speculative-algorithm",
            speculative_config["algorithm"],
            "--speculative-draft-model-path",
            draft_model_path,  # type: ignore[list-item]  (validated above)
        ]
        if "num_steps" in speculative_config:
            cmd += ["--speculative-num-steps", str(speculative_config["num_steps"])]
        if "num_draft_tokens" in speculative_config:
            cmd += [
                "--speculative-num-draft-tokens",
                str(speculative_config["num_draft_tokens"]),
            ]
        if "eagle_topk" in speculative_config:
            cmd += ["--speculative-eagle-topk", str(speculative_config["eagle_topk"])]
        if draft_revision:
            # Pin the drafter's commit, parallel to --revision for the
            # target — keeps a speculative redeploy byte-reproducible.
            cmd += ["--speculative-draft-model-revision", draft_revision]

    if kv_cache_dtype is not None:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]

    # ── Memory-snapshot co-operation flags ──────────────────────────────
    # See Modal's official SGLang snapshot example for the lifecycle.
    if enable_memory_saver:
        cmd.append("--enable-memory-saver")
    if enable_weights_cpu_backup:
        cmd.append("--enable-weights-cpu-backup")

    # ── Optional perf knobs (kept around for future re-enable) ─────────
    if enable_torch_compile:
        # NOTE: --enable-torch-compile is marked "out of maintenance.
        # Not recommended." in the SGLang server-args reference as of
        # v0.5.12. The default deployments leave it off; this branch is
        # preserved so a future SGLang release that revives the feature
        # is one-line away. Cold-start cost adds 1-3 min the first time.
        cmd.append("--enable-torch-compile")
        if torch_compile_max_bs is not None:
            cmd += ["--torch-compile-max-bs", str(torch_compile_max_bs)]

    if cuda_graph_bs is not None:
        # Enumerate exact decode batch sizes for CUDA-graph capture —
        # faster cold start than auto-discovery, and guarantees every bs
        # we actually hit is captured.
        cmd += ["--cuda-graph-bs", *[str(b) for b in cuda_graph_bs]]

    if num_continuous_decode_steps is not None:
        cmd += ["--num-continuous-decode-steps", str(num_continuous_decode_steps)]

    if max_prefill_tokens is not None:
        cmd += ["--max-prefill-tokens", str(max_prefill_tokens)]

    # ── Observability ───────────────────────────────────────────────────
    if enable_metrics:
        cmd.append("--enable-metrics")
    if enable_cache_report:
        cmd.append("--enable-cache-report")
    if enable_request_time_stats:
        cmd.append("--enable-request-time-stats-logging")
    if log_requests:
        cmd.append("--log-requests")
        if log_requests_level is not None:
            # 0=metadata, 1=+sampling params, 2=+partial I/O, 3=full
            cmd += ["--log-requests-level", str(log_requests_level)]

    if skip_server_warmup:
        cmd.append("--skip-server-warmup")

    if api_key_env and os.environ.get(api_key_env):
        cmd += ["--api-key", os.environ[api_key_env]]

    if extra_args:
        cmd += extra_args

    return cmd
