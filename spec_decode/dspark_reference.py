"""DeepSeek DSpark (Gemma 4) draft-model reference forward, on Modal.

What this is
------------
A small, self-contained reference harness for the **DSpark** speculative-decoding
draft model that DeepSeek ships with `DeepSpec`
(https://github.com/deepseek-ai/DeepSpec, MIT). It loads the *public* released
draft checkpoint `deepseek-ai/dspark_gemma4_12b_block7`, pairs it with its
target `google/gemma-4-12B-it`, and runs a single **block-7 draft forward** so you
can inspect exactly what the drafter proposes: its base logits, its rank-256
Markov-corrected logits, its confidence head, and the greedy draft tokens.

Why it exists
-------------
DSpark is an *offline train + eval* framework — it ships no inference kernels and
no serving integration. To port or debug a DSpark drafter inside a fast runtime
(vLLM, SGLang, MLX, a custom CUDA-graph loop, ...) you first need a ground-truth
PyTorch reference: "given these target hidden states, the drafter must produce
*these* logits and *these* tokens". This harness is that reference. Point your
own implementation at the same prompts and diff the tensors op-by-op; a drafter
that never matches the reference will show 0% acceptance no matter how fast it is.

It emits a compact JSON fixture per prompt (top-k logits by default, full logits
optional) that any external re-implementation can compare against.

Hardware
--------
Runs on a single **B200** (Blackwell, ~180 GB). The bf16 12B target (~24 GB) +
the 3.43 B draft (~7 GB) + live hidden states do not fit a 24 GB card, so this is
deliberately a big-GPU *offline reference* job — not a serving path.

Faithfulness
------------
Every call mirrors DeepSpec's own eval path (`deepspec/eval/dspark/evaluator.py`,
`draft_ops.py`) at the pinned commit below. It bypasses the `Gemma4DSparkEvaluator`
class on purpose so no `torch.distributed`/NCCL process group is required for a
single forward; it calls the same modeling + `draft_ops` functions the evaluator
calls. See `docs/dspark-spec-decode.md` for the architecture writeup.

Setup (deploy on your own account — nothing here is hosted for you)
------------------------------------------------------------------
1. Create a Modal secret named ``huggingface-secret`` holding ``HF_TOKEN`` with an
   HF token that can pull the models below.
2. ``modal run spec_decode/dspark_reference.py::main`` — writes fixtures under
   ``spec_decode/out/``. (Two entrypoints exist, so name one: ``::main`` for the
   forward, ``::preflight`` for the cheap env/access check.)

    # cheap CPU env + gated-access check first:
    modal run spec_decode/dspark_reference.py::preflight

    # default smoke prompt:
    modal run spec_decode/dspark_reference.py::main

    # your own prompt + full dense logits:
    modal run spec_decode/dspark_reference.py::main --prompt "What is the capital of France?" --full-logits

DeepSpec is MIT-licensed and the Gemma 4 base is Apache-2.0; the DSpark checkpoint
carries DeepSeek's own license. Credit DeepSeek's DeepSpec and respect each.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

# ─────────────────────────────────────────────────────────────────────
# Constants — tune here. Pins match DeepSpec @ afdfa7c9 and the released
# checkpoint; changing them is how you retarget to a different draft/target.
# ─────────────────────────────────────────────────────────────────────
APP_NAME = "dspark-reference"
GPU = "B200"

DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec"
DEEPSPEC_COMMIT = "afdfa7c9382a3341a3e6f17756dd816da79f132c"
DEEPSPEC_DIR = "/opt/DeepSpec"

DRAFT_MODEL = "deepseek-ai/dspark_gemma4_12b_block7"   # 5-layer DSpark draft
TARGET_MODEL = "google/gemma-4-12B-it"                 # the draft's target model
TARGET_LAYER_IDS = [5, 17, 29, 41, 46]                # EAGLE3-style multi-taps

HF_CACHE_DIR = "/root/.cache/huggingface"
HF_SECRET_NAME = "huggingface-secret"                  # generic; you create it
OUT_DIR = Path(__file__).resolve().parent / "out"

# Pinned deps straight from DeepSpec's requirements.txt (transformers must ship
# `models.gemma4`; the eval path uses SDPA so no flash-attn / triton is needed).
# huggingface_hub is intentionally left unpinned — transformers==5.10.2 requires
# huggingface-hub>=1.5.0,<2.0 and resolves it transitively (HfApi lives there).
_PIP = [
    "torch==2.9.1",
    "transformers==5.10.2",
    "numpy==2.4.4",
    "safetensors==0.7.0",
    "sentencepiece==0.2.1",
]


def _build_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        # uv resolver (fast, reproducible) — matches this repo's uv-everywhere convention.
        .uv_pip_install(*_PIP)
        # DeepSpec is NOT pip-installable (no setup.py/pyproject): vendor the
        # source tree at a pinned commit and put it on PYTHONPATH.
        .run_commands(
            f"git clone {DEEPSPEC_REPO} {DEEPSPEC_DIR}",
            f"cd {DEEPSPEC_DIR} && git checkout {DEEPSPEC_COMMIT}",
        )
        .env({"PYTHONPATH": DEEPSPEC_DIR, "PYTHONUNBUFFERED": "1"})
    )


image = _build_image()
app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("dspark-hf-cache", create_if_missing=True)


def _topk(logits, k: int):
    """Return [(token_id, logit), ...] top-k for a 1-D logits row (as floats)."""
    import torch

    vals, idx = torch.topk(logits.float(), k)
    return [[int(i), float(v)] for i, v in zip(idx.tolist(), vals.tolist())]


@app.function(
    gpu=GPU,
    timeout=3600,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def dspark_reference(prompts: list[str], top_k: int = 8, full_logits: bool = False) -> list[dict]:
    """Run one block-7 DSpark draft forward per prompt; return reference fixtures.

    Mirrors DeepSpec eval: target prefill with output_hidden_states -> tap
    concat -> draft block forward -> softcapped base logits -> autoregressive
    rank-256 Markov correction -> confidence head -> greedy draft tokens.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    # DeepSpec's own classes (from the vendored source tree on PYTHONPATH).
    from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
    from deepspec.modeling.dspark.common import extract_context_feature
    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
    from deepspec.utils.sampling import logits_to_probs, sample_from_probs

    device = "cuda"
    torch.manual_seed(0)

    print(f"[dspark] loading target {TARGET_MODEL} (bf16, sdpa)...", flush=True)
    target = (
        AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    print(f"[dspark] loading draft {DRAFT_MODEL} (bf16, sdpa)...", flush=True)
    draft = (
        Gemma4DSparkModel.from_pretrained(
            DRAFT_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)  # tokenizer from the TARGET

    block_size = int(draft.block_size)
    mask_id = int(draft.mask_token_id)
    layer_ids = list(getattr(draft, "target_layer_ids", TARGET_LAYER_IDS))
    print(f"[dspark] block_size={block_size} mask_token_id={mask_id} taps={layer_ids}", flush=True)

    fixtures: list[dict] = []
    for pi, prompt in enumerate(prompts):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        num_input = int(input_ids.shape[1])
        start = num_input
        position_ids = torch.arange(num_input + block_size + 8, device=device).unsqueeze(0)

        with torch.inference_mode():
            # (a) target prefill, capturing hidden states for the taps
            out = target(
                input_ids=input_ids,
                position_ids=position_ids[:, :num_input],
                past_key_values=DynamicCache(),
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            first_tok = sample_from_probs(logits_to_probs(out.logits, 0.0))  # greedy anchor

            # (b) fused target context: concat raw taps [B,S,5H]
            ctx = extract_context_feature(out.hidden_states, layer_ids)

            # (c) one block-7 draft forward (all-mask block, slot 0 = anchor token)
            draft_input_ids = torch.full(
                (1, block_size), mask_id, dtype=torch.long, device=device
            )
            draft_input_ids[:, 0] = first_tok[:, 0]
            block_hidden = forward_dspark_draft_block(
                draft,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=DynamicCache(),
                target_hidden_states=ctx,
                start=start,
                block_size=block_size,
            )[:, :block_size, :]

            # (d) heads: softcapped base -> autoregressive rank-256 Markov -> greedy
            base_logits = draft.compute_logits(block_hidden)  # [1,7,V], softcapped
            greedy_tokens, markov_logits = draft.sample_draft_tokens(
                base_logits,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=0.0,
                hidden_states=block_hidden,
            )

            # (e) confidence head (optional in general; present on this ckpt)
            confidence_logits = None
            try:
                prev = torch.cat([draft_input_ids[:, :1], greedy_tokens[:, :-1]], dim=1)
                confidence_logits = draft.predict_confidence_step(
                    hidden_states=block_hidden, prev_token_ids=prev
                )
            except Exception as exc:  # keep the fixture; record the gap
                print(f"[dspark] confidence head skipped: {exc}", flush=True)

        base_row = base_logits[0]      # [7,V]
        markov_row = markov_logits[0]  # [7,V]
        fixture = {
            "prompt": prompt,
            "target_model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL,
            "deepspec_commit": DEEPSPEC_COMMIT,
            "block_size": block_size,
            "mask_token_id": mask_id,
            "target_layer_ids": layer_ids,
            "input_token_ids": input_ids[0].tolist(),
            "anchor_token_id": int(first_tok[0, 0]),
            "greedy_draft_tokens": greedy_tokens[0].tolist(),
            "target_hidden_taps_shape": list(ctx.shape),
            "target_last_hidden_shape": list(out.hidden_states[-1].shape),
            "dspark_base_top_k": [_topk(base_row[t], top_k) for t in range(block_size)],
            "dspark_markov_top_k": [_topk(markov_row[t], top_k) for t in range(block_size)],
            "confidence_logits": (None if confidence_logits is None else confidence_logits[0].float().tolist()),
        }
        if full_logits:
            fixture["dspark_base_logits"] = base_row.float().cpu().tolist()
            fixture["dspark_markov_logits"] = markov_row.float().cpu().tolist()
        fixtures.append(fixture)
        print(
            f"[dspark] prompt {pi}: greedy={fixture['greedy_draft_tokens']} "
            f"conf={fixture['confidence_logits']}",
            flush=True,
        )

    return fixtures


@app.function(
    cpu=2.0,
    timeout=1200,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def env_check() -> dict:
    """Cheap (CPU) preflight: confirm imports, that transformers ships
    `models.gemma4`, and that the HF token can reach both gated repos — before
    spending a B200 on the real forward. Does not download weights."""
    import importlib
    import os

    from huggingface_hub import HfApi

    report: dict = {"ok": True, "checks": {}}

    def check(name, fn):
        try:
            report["checks"][name] = fn()
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            report["checks"][name] = f"FAIL: {exc}"
            report["ok"] = False

    import transformers

    report["checks"]["transformers_version"] = transformers.__version__
    check("gemma4_in_transformers", lambda: "ok" if importlib.import_module("transformers.models.gemma4") else "missing")
    check("import_deepspec", lambda: "ok" if importlib.import_module("deepspec.modeling.dspark.gemma4") else "missing")
    check("hf_token_present", lambda: "ok" if os.environ.get("HF_TOKEN") else "HF_TOKEN empty")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    check("access_target", lambda: sorted(s.rfilename for s in api.model_info(TARGET_MODEL).siblings)[:4])
    check("access_draft", lambda: sorted(s.rfilename for s in api.model_info(DRAFT_MODEL).siblings)[:4])
    return report


@app.local_entrypoint()
def preflight() -> None:
    """Run the cheap env/access check. `modal run spec_decode/dspark_reference.py::preflight`"""
    report = env_check.remote()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        print("\n[FAIL] preflight found problems (see checks above).", file=sys.stderr)
        sys.exit(1)
    print("\n[OK] preflight passed — ready for the B200 reference run.")


@app.local_entrypoint()
def main(prompt: str = "What is the capital of France?", top_k: int = 8, full_logits: bool = False) -> None:
    """Generate DSpark reference fixtures for one or more prompts.

    Pass --prompt multiple times for a batch; --full-logits also dumps the dense
    base/Markov logits (large). Fixtures land under spec_decode/out/.
    """
    prompts = prompt if isinstance(prompt, list) else [prompt]
    print(f"\n=== DSpark reference forward ({GPU}) ===")
    for p in prompts:
        print(f"  prompt: {p!r}")

    fixtures = dspark_reference.remote(prompts=prompts, top_k=top_k, full_logits=full_logits)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, fx in enumerate(fixtures):
        dest = OUT_DIR / f"dspark_reference_{stamp}_{i}.json"
        dest.write_text(json.dumps(fx, indent=2, sort_keys=True))
        print(f"  wrote {dest}")

    if not fixtures:
        print("[FAIL] no fixtures produced", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(fixtures)} fixture(s) written to {OUT_DIR}")
