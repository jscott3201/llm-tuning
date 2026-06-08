"""Deploy google/gemma-4-E4B-it on Modal (L40S) via vLLM.

Serves the **base** Gemma-4-E4B-it model (no LoRA / adapter
attachment). It's a practical SFT target — train an adapter against
it, then deploy a separate adapter-aware endpoint to A/B the
fine-tuned model against this baseline.

L40S is the cost-appropriate choice: E4B's ~4B active params + PLE
leave the 48 GiB of HBM with plenty of KV cache headroom, and L40S
runs a few times cheaper per hour than a B200 for this workload class.

## Setup (one-time)

1. `modal token new` — authenticate the CLI.
2. Gemma 4 is Apache 2.0, NOT gated — no HF token required.
3. `modal deploy serve/vllm/serve_e4b.py`. First deploy pulls ~8 GiB
   of weights into `gemma4-hf-cache`; cold boot ~1-2 min, warm ~20 s.

## Consumption

After deploy, Modal prints a PUBLIC base URL (the URL printed by
`modal deploy`, of the form `*.modal.run`). Point any OpenAI-compatible
client at the server root. The vLLM server binds `0.0.0.0` so Modal's
ingress can reach it. Endpoints are public by default; to require
auth, lock it down at the ingress with proxy auth — see
modal.com/docs/guide/webhook-proxy-auth. This script does not
implement auth.

## Tool-call parser

ON. Probe and eval workloads run sequentially or at low concurrency,
so vLLM #39392's shared-state bug (which fires at 2+ concurrent tool
calls) doesn't bite. If you parallelise this endpoint above 2-way,
flip `enable_tool_call_parser=False` below and let the eval harness
parse raw `<|tool_call>...` tokens client-side via
`_common.gemma4_parser`.

## Thinking mode

Per-request via `chat_template_kwargs.enable_thinking=True` in the
OpenAI request body. Run a base-capability probe without thinking to
isolate base capability; run the post-SFT eval with thinking on to
match production inference.

## Serving a fine-tuned LoRA adapter

vLLM 0.19.1 ships LoRA support for `Gemma4ForConditionalGeneration`
via PR #39291 (which fixed feature request #39246). Scope:

  - **Language backbone is LoRA-able** — q_proj, k_proj, v_proj,
    o_proj, gate_proj, up_proj, down_proj. A text-only SFT preset
    targets exactly these.
  - **Vision and audio towers are NOT yet LoRA-able** — they still
    use HF's auto-model path internally and are out of scope for the
    initial PR. For text-only fine-tuning this is a non-issue.

Recipe — copy this file to a sibling `serve_e4b_adapter.py`, rename
the app to `gemma4-e4b-adapter`, and pass through `extra_args`:

    extra_args=[
        "--enable-lora",
        "--lora-modules", "my-adapter=your-username/my-adapter",
        "--max-lora-rank", "16",   # match LoraConfig.r in the preset
    ]

Separate apps keep base-serve and adapter-serve deploy lifecycles
independent: a broken adapter release doesn't knock the baseline
endpoint offline, and the A/B comparison has two distinct URLs to
point the eval harness at.

Alternative: **merge then serve**. Call `model.merge_and_unload()` at
the end of the SFT run, push the merged weights as a full model, and
serve with this exact base-serve script pointed at the merged repo.
Loses dynamic-adapter flexibility but produces a self-contained
deployable artefact and shaves cold-boot a touch.
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

SPEC = get("e4b")

# Optional GPU override. None = use the registry-canonical L40S (the
# cost-appropriate production choice for E4B). Set to "B200" if Modal
# is queue-bound on L40S, or to run all sizes on uniform hardware for a
# like-for-like benchmark.
GPU_OVERRIDE: str | None = None

# MTP / speculative decoding is currently UNAVAILABLE for Gemma 4 in
# vLLM 0.19.1: the spec-decode path rejects multimodal targets and all
# published Gemma 4 checkpoints are multimodal
# (`Gemma4*ForConditionalGeneration`). See the folder README for the
# full detail. Track upstream before flipping this on.
ENABLE_MTP = False

app = modal.App("gemma4-e4b-solo")
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
    timeout=60 * 60,
    scaledown_window=60 * 10,
    max_containers=2,
)
@modal.concurrent(max_inputs=SPEC.concurrency)
# 15-minute warm-up budget — first deploys hit cold weight cache and
# cold torch.compile, both of which can stretch to several minutes on
# L40S. Warm boots return in <30s and won't hit this ceiling.
@modal.web_server(port=8000, startup_timeout=900)
def serve() -> None:
    """Launch `vllm serve` for E4B on port 8000."""
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
        # E4B probe traffic is low-concurrency; parser is safe here.
        # Flip to False if you parallelise above 2-way concurrent.
        enable_tool_call_parser=True,
        enable_prefix_caching=True,
        enable_async_scheduling=True,
        fast_boot=False,
        speculative_config=speculative_config,
    )

    proc = subprocess.Popen(cmd)
    wait_for_health(proc, timeout_s=900, label=f"gemma4-{SPEC.short}")
