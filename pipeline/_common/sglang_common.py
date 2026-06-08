"""Shared Modal image + `sglang.launch_server` command builder for the
SGLang serve scripts.

Mirrors the `_common/vllm_common.py` shape so the per-deploy SGLang
scripts (`serve_e2b.py` / `e4b` / `12b` / `26b` / `31b`) only carry the
genuine differences: GPU class, context window, concurrency, MTP / LoRA
toggles.

Ingress model: same as the vLLM side — wrap `sglang.launch_server` in a
Modal function decorated with `@modal.web_server(port=8000,
startup_timeout=...)`, which publishes a PUBLIC `*.modal.run` URL (use
the URL printed by `modal deploy`). The server MUST bind `0.0.0.0` (the
builder hardcodes that) so Modal's ingress can reach it. Endpoints are
public by default; to require auth, pass `requires_proxy_auth=True` to
the web decorator — see modal.com/docs/guide/webhook-proxy-auth. This
module does not implement auth.

## Why a second runtime at all

The vLLM serve scripts get blocked on two specific things:

  - **MTP / speculative decoding**: vLLM 0.19.1 raises
    `NotImplementedError: Speculative Decoding with draft models or
    parallel drafting does not support multimodal models yet`
    (`vllm/v1/spec_decode/eagle.py::_raise_if_multimodal`). The Gemma 4
    checkpoints are `Gemma4*ForConditionalGeneration` and trip that
    guard at engine init.
  - **Dynamic LoRA hot-swap**: vLLM 0.19.1 ships static LoRA via
    `--lora-modules` at boot but the runtime swap UX is rougher than
    SGLang's clean `/load_lora_adapter` + `/unload_lora_adapter` REST
    endpoints with pinned-slot support.

SGLang took the opposite design choice on the first one: its in-flight
MTP path (PR #24436) introduces a new `FROZEN_KV_MTP` algorithm that
explicitly supports both text and multimodal Gemma 4 targets. The
`-it-assistant` drafters share the target's KV cache and carry a
recurrent hidden state across draft steps, which doesn't fit cleanly
under EAGLE / EAGLE3 / NEXTN — hence the new algorithm rather than a
flag on an existing one.

## Status of the upstream pieces (verified 2026-05-07)

  - **Base Gemma 4 model class**: merged
    (https://github.com/sgl-project/sglang/pull/21952, 2026-04-07).
  - **SWAKVPool determinism / sentinel fix**: merged
    (https://github.com/sgl-project/sglang/pull/24395, 2026-05-05).
  - **Gemma 4 VLM optimization**: merged
    (https://github.com/sgl-project/sglang/pull/24048, 2026-05-04).
  - **Gemma 4 + MTP runtime (`FROZEN_KV_MTP`)**: open, mergeable,
    blocked on review/CI
    (https://github.com/sgl-project/sglang/pull/24436). Quality
    benchmarks already match the target-only baseline within ±0.02 on
    GPQA-Diamond and GSM8K across all four sizes. Speed numbers are a
    follow-up.
  - **Gemma 3/4 + EAGLE3** (community-trained drafters, separate path
    from Google's MTP heads): open
    (https://github.com/sgl-project/sglang/pull/23976).

The serve scripts here pin upstream `sglang==0.5.11` and default
`ENABLE_MTP = False`. Flip True per script once #24436 lands; the
`build_serve_cmd_sgl` plumbing is already in place.

## Attention backend

Gemma 4's hybrid local/global attention (sliding_window layers
interleaved with full_attention) together with the family's fixed
head_dim=256 is incompatible with the default flash kernels — pass
`--attention-backend triton`. This is also a hard requirement when MTP
is enabled (the in-flight FROZEN_KV_MTP path requires triton because
FlashInfer is incompatible with the recurrent hidden-state path the
Gemma 4 assistant drafters use). `model_registry.ModelSpec` carries a
`requires_triton_attention` flag per size to drive this.

## Tool-call + reasoning parsers

SGLang ships a `gemma4` reasoning parser and a `gemma4` tool-call
detector. The serve command exposes both via:

  - `--reasoning-parser gemma4`
  - `--tool-call-parser gemma4`

Both default ON for low-concurrency endpoints and OFF for a
high-concurrency corpus-generator, mirroring the vLLM-side
`enable_tool_call_parser` choice. The same `_common/gemma4_parser.py`
client-side parser still extracts raw `<|tool_call>` tokens when the
server-side parser is disabled.
"""

from __future__ import annotations

import modal


# ─────────────────────────────────────────────────────────────────────
# Pinned versions
# ─────────────────────────────────────────────────────────────────────

SGLANG_VERSION = "0.5.11"
"""Canonical SGLang pin. 0.5.11 ships base Gemma 4 support, SWAKVPool
determinism fix, Gemma 4 VLM optimisation, and the MTP cookbook docs.
The runtime MTP support (PR #24436) is not in this release; flip
`ENABLE_MTP = True` in a serve script only after that PR merges and a
release containing it ships, OR pin a newer version here."""

# Pinned only when MTP is enabled — PR #24436 currently requires this
# specific transformers commit until the upstream HF release picks up
# the matching MTP changes. Until #24436 merges and a SGLang release
# bundles a stable transformers floor, this commit pin is the safest
# value to use.
MTP_TRANSFORMERS_REF = (
    "git+https://github.com/huggingface/transformers.git"
    "@2c7d385621c80fee70c1472f3a622fcba2c93fb9"
)

DEFAULT_MAX_MODEL_LEN = 16_384
"""Matches the vLLM-side default. Smaller than the model's native window
(128K-256K depending on size) so KV cache has room for concurrent
requests without backing them off into eviction churn."""


# ─────────────────────────────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────────────────────────────


def make_sglang_image(
    sglang_version: str = SGLANG_VERSION,
    *,
    include_mtp_transformers: bool = False,
) -> modal.Image:
    """Canonical SGLang image: Debian slim + uv + hf-transfer.

    `uv pip install` resolves the CUDA wheel graph an order of magnitude
    faster than vanilla pip, which matters every time the image rebuilds.
    `hf-transfer` saturates Modal's egress on first weight pull.

    `sglang[all]` brings in the optional vendored kernels (FlashInfer,
    Triton kernel registry, vLLM-style tensor parallel utilities). The
    plain `sglang` extra omits some of these and trips up Gemma 4's
    SWA path.

    `include_mtp_transformers=True` pins the specific HF transformers
    commit that PR #24436 requires for the `gemma4_assistant` model
    type. Leave False on base-serve images — it adds image build time
    and isn't needed unless MTP is wired up.

    `add_local_python_source("_common")` mounts the shared package into
    the container so the per-deploy serve scripts can import it.

    These Gemma 4 checkpoints are ungated/public, so no HF token is
    required. If you point a deploy at a gated repo, attach one with
    `modal.Secret.from_name("huggingface-secret")` (a secret you create
    in your own Modal workspace holding `HF_TOKEN`).
    """
    base = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            f"sglang[all]=={sglang_version}",
            "transformers>=5.5.0",
            "huggingface_hub[hf_xet]>=1.11",
            "hf-transfer>=0.1.8",
        )
        .env(
            {
                # Saturate Modal egress on the first 50+ GiB pull.
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
    )

    if include_mtp_transformers:
        # Force-reinstall the MTP-required transformers commit on top
        # of whatever sglang's extras pulled. Has to come AFTER the
        # base install so the resolver doesn't downgrade it.
        base = base.uv_pip_install(MTP_TRANSFORMERS_REF)

    return base.add_local_python_source("_common")


# ─────────────────────────────────────────────────────────────────────
# Serve command builder
# ─────────────────────────────────────────────────────────────────────


def build_serve_cmd_sgl(
    model_path: str,
    served_model_names: list[str],
    *,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    mem_fraction_static: float = 0.92,
    chunked_prefill_size: int = 16_384,
    enable_tool_call_parser: bool = True,
    enable_reasoning_parser: bool = True,
    enable_prefix_caching: bool = True,
    attention_backend: str | None = None,
    speculative_config: dict | None = None,
    kv_cache_dtype: str | None = None,
    max_running_requests: int | None = None,
    lora_paths: list[str] | None = None,
    max_lora_rank: int | None = None,
    max_loras_per_batch: int | None = None,
    lora_target_modules: list[str] | None = None,
    api_key_env: str | None = "API_KEY",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Assemble the `python -m sglang.launch_server` argv for a Gemma 4
    endpoint.

    Defaults match upstream SGLang 0.5.11 sane behaviour for Gemma 4.
    Per-deploy tuning (GPU class, concurrency, MTP, LoRA paths) lives
    in the caller.

    The server always binds `--host 0.0.0.0 --port 8000` so the
    enclosing `@modal.web_server(port=8000, ...)` can reach it and
    publish the public `*.modal.run` URL — do not change the host to
    127.0.0.1 under Modal or routing fails.

    Notable knobs (mirrored against `vllm_common.build_serve_cmd` so
    the per-deploy diff is small):

    - `mem_fraction_static`: SGLang's analogue of vLLM's
      `--gpu-memory-utilization`. Same semantics — fraction of HBM
      reserved for static (weights + KV cache) state.
    - `chunked_prefill_size`: SGLang's analogue of vLLM's
      `--max-num-batched-tokens`. Prefill is chunked into pieces this
      size to amortise across batches.
    - `enable_tool_call_parser` / `enable_reasoning_parser`: emit
      `--tool-call-parser gemma4` and `--reasoning-parser gemma4`.
      Default both ON for probe traffic; flip the tool-call one OFF
      on a high-concurrency corpus-gen endpoint where the server-side
      parser may surface concurrency bugs (mirroring the vLLM-side
      mitigation for #39392). Client-side `_common/gemma4_parser.py`
      extracts raw tokens when the server-side parser is off.
    - `attention_backend`: emits `--attention-backend <name>`. Set to
      `"triton"` for Gemma 4 — its hybrid sliding/full attention plus
      head_dim=256 is incompatible with the default flash kernels, and
      the in-flight FROZEN_KV_MTP path requires triton as well
      (FlashInfer is incompatible with the recurrent hidden-state path
      the Gemma 4 assistant drafters use). Drive this from
      `model_registry.ModelSpec.requires_triton_attention`.
    - `speculative_config`: a dict with the keys SGLang's CLI exposes
      as separate flags. Pass:

          {"algorithm": "NEXTN",  # auto-promotes to FROZEN_KV_MTP for
                                  # the gemma4_assistant model_type
           "draft_model": "google/gemma-4-E4B-it-assistant",
           "num_steps": 5,
           "eagle_topk": 1,
           "num_draft_tokens": 6}

      Equivalents:
        algorithm        -> --speculative-algorithm
        draft_model      -> --speculative-draft-model-path
        num_steps        -> --speculative-num-steps
        eagle_topk       -> --speculative-eagle-topk
        num_draft_tokens -> --speculative-num-draft-tokens

      Per the cookbook (PR #24433): topk=1 is fast-path; topk=3
      with num_draft_tokens=12 or topk=5 with num_draft_tokens=16
      enables tree verification with higher acceptance rates.
    - `kv_cache_dtype`: emits `--kv-cache-dtype <value>` (e.g. `"fp8_e4m3"`).
      Halves KV cache footprint at a tiny accuracy cost — useful when
      long-context throughput is bottlenecked by KV cache.
    - `max_running_requests`: emits `--max-running-requests <n>`. SGLang's
      analogue of vLLM's `--max-num-seqs`. The MTP path force-defaults
      this to 48 if unset; explicitly setting it documents the choice.
    - `lora_paths`: list of `name=hf_repo` strings. Emits
      `--enable-lora --lora-paths name1=repo1 name2=repo2 ...`.
      Together with `--max-lora-rank` (auto-inferred otherwise) and
      `--max-loras-per-batch` (default 8) this is the static LoRA
      bring-up. Dynamic adapters land via `POST /load_lora_adapter`
      and `POST /unload_lora_adapter` after the server is live, with
      per-request selection via `model: "base:adapter-name"` in the
      OpenAI body.
    - `lora_target_modules`: explicit union set; only required when
      dynamic adapters may have different target_modules than the
      ones in `lora_paths`. For Gemma 4 SFT the canonical set is
      ["q_proj", "k_proj", "v_proj", "o_proj",
       "gate_proj", "up_proj", "down_proj"].
    - `api_key_env` names an environment variable the container reads
      to OPTIONALLY gate the endpoint with `--api-key`. It is off
      unless that env var is set (e.g. via `modal.Secret.from_name(...)`).
      Auth is your choice — Modal endpoints are public by default and
      can instead be locked down at the ingress with proxy auth; see
      modal.com/docs/guide/webhook-proxy-auth.
    - `extra_args`: forwarded verbatim. Use this for size-specific
      flags like `--tp-size N`, `--enable-overlap-schedule` /
      `--disable-overlap-schedule`, or any flag SGLang adds after this
      module's last sync.

    The MTP path force-disables overlap scheduling and mixed chunked
    prefill regardless of what's passed here — those are SGLang
    invariants for FROZEN_KV_MTP, not knobs we control.
    """
    import os

    cmd: list[str] = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        *served_model_names,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--context-length",
        str(max_model_len),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
    ]

    if enable_reasoning_parser:
        cmd += ["--reasoning-parser", "gemma4"]

    if enable_tool_call_parser:
        cmd += ["--tool-call-parser", "gemma4"]

    if not enable_prefix_caching:
        cmd.append("--disable-radix-cache")

    if attention_backend is not None:
        cmd += ["--attention-backend", attention_backend]

    if speculative_config is not None:
        cmd += ["--speculative-algorithm", speculative_config["algorithm"]]
        if "draft_model" in speculative_config:
            cmd += [
                "--speculative-draft-model-path",
                speculative_config["draft_model"],
            ]
        if "num_steps" in speculative_config:
            cmd += [
                "--speculative-num-steps",
                str(speculative_config["num_steps"]),
            ]
        if "eagle_topk" in speculative_config:
            cmd += [
                "--speculative-eagle-topk",
                str(speculative_config["eagle_topk"]),
            ]
        if "num_draft_tokens" in speculative_config:
            cmd += [
                "--speculative-num-draft-tokens",
                str(speculative_config["num_draft_tokens"]),
            ]

    if kv_cache_dtype is not None:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]

    if max_running_requests is not None:
        cmd += ["--max-running-requests", str(max_running_requests)]

    if lora_paths:
        cmd += ["--enable-lora", "--lora-paths", *lora_paths]
    if max_lora_rank is not None:
        cmd += ["--max-lora-rank", str(max_lora_rank)]
    if max_loras_per_batch is not None:
        cmd += ["--max-loras-per-batch", str(max_loras_per_batch)]
    if lora_target_modules:
        cmd += ["--lora-target-modules", *lora_target_modules]

    if api_key_env and os.environ.get(api_key_env):
        cmd += ["--api-key", os.environ[api_key_env]]

    if extra_args:
        cmd += extra_args

    return cmd


# ─────────────────────────────────────────────────────────────────────
# Health poll — re-export from vllm_common
# ─────────────────────────────────────────────────────────────────────


# SGLang's `/health` endpoint is OpenAI-compatible at the same path
# vLLM uses, so the existing poll helper works as-is. Re-exported here
# so SGLang serve scripts only need to import from one module.
from _common.vllm_common import wait_for_health  # noqa: E402, F401
