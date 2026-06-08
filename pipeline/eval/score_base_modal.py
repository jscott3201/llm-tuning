#!/usr/bin/env python3
"""Score the base Gemma 4 model on the chinook eval set.

Loads the model with HF Transformers in a Modal container, runs each
scenario in the eval set through `model.generate(...)`, parses the
output with `_common.gemma4_parser`, and scores it against the rubric
in `_common.eval_scoring`. Writes a per-axis + per-category report to
`/output/<label>/eval_results.json` on the eval-output volume.

This is intentionally Transformers-based, not vLLM-based. Two reasons:

  1. If you fine-tune with HF Transformers (SFT/LoRA on the same
     stack), computing the baseline here means it runs against exactly
     the same forward-pass code path the fine-tuned model is
     benchmarked under. No "is this a different sampler?" doubt later.
  2. vLLM 0.19.1 has the open #39392 / #39130 quirks (see
     `_common/vllm_common.py`); for a deterministic `do_sample=False`
     baseline it's cleaner to skip vLLM entirely.

Five Gemma 4 specifics this script preserves (any matching SFT script
must preserve them too):

  1. `add_special_tokens=False` when calling the tokenizer — the chat
     template already emits BOS.
  2. `AutoTokenizer.from_pretrained(...)` + load the sibling
     `chat_template.jinja` manually; don't rely on `tokenizer_config`.
  3. `enable_thinking` passed consistently to `apply_chat_template`
     and `model.generate(...)`; fall back without it if the tokenizer
     version rejects the kwarg.
  4. Label masking on assistant turns only. Gemma 4's chat template
     uses `<|turn>role\\n` to open and `<turn|>\\n` to close (a change
     from Gemma 2/3, which used `<start_of_turn>` / `<end_of_turn>`).
     Look these up via `tokenizer.convert_tokens_to_ids(...)` rather
     than hardcoding IDs — the actual integer values are subject to
     tokenizer revisions.
  5. `eos_token_id=[<eos>, <turn|>]` for generation so the model
     terminates on the correct turn marker rather than rambling.

Run:

    # 1. Upload the eval scenarios to the eval-data volume.
    modal volume put gemma4-eval-data \\
        eval/scenarios/chinook_eval_v1.jsonl

    # 2. Score the base model.
    modal run --detach eval/score_base_modal.py::score_base \\
        --model e4b \\
        --output-label base-e4b-v1

    # 3. Read the report back from the output volume.
    modal volume get gemma4-eval-output \\
        base-e4b-v1/eval_results.json /tmp/
"""

from __future__ import annotations

import modal

APP_NAME = "gemma4-eval"

# Volumes — created on first use; the HF cache, eval data, and output
# all have natural lifetimes that benefit from being kept across runs.
# The model-cache volume name matches the convention used by the serve
# scripts so a weight pull is shared across stages on the same workspace.
model_volume = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("gemma4-eval-data", create_if_missing=True)
output_volume = modal.Volume.from_name("gemma4-eval-output", create_if_missing=True)

VOLUME_MOUNTS = {
    "/models": model_volume,
    "/data": data_volume,
    "/output": output_volume,
}

# CUDA 12.8 + cu128 torch wheels: required for B200 (sm_100). cu126
# wheels only compile for sm_50..sm_90 and die at first CUDA allocation.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "curl", "libcurl4-openssl-dev", "libssl-dev")
    .uv_pip_install(
        "torch==2.11.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        # peft is kept in the eval image so its presence matches a
        # Transformers SFT image — saves an image rebuild when wiring
        # up a fine-tune step that scores against this baseline.
        "transformers==5.5.4",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "datasets>=4.8",
        "huggingface_hub[hf_xet]>=1.11,<2",
        "safetensors>=0.7",
        "sentencepiece",
        "protobuf",
        "einops",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    # Mount the shared library and the tool manifest so the in-container
    # scoring code can import them.
    .add_local_python_source("_common")
    .add_local_python_source("tool_manifest")
)

app = modal.App(APP_NAME, image=image)

# These Gemma 4 checkpoints are ungated/public, so no HF token is
# required to pull them — by default NO secret is attached and the run
# works with zero setup on a fresh workspace. The in-container code
# reads `HF_TOKEN` from the environment if present and otherwise pulls
# anonymously. To point a deploy at a gated repo, create a secret in
# your own Modal workspace named "huggingface-secret" holding
# `HF_TOKEN`, then add it to `hf_secrets` below:
#     hf_secrets = [modal.Secret.from_name("huggingface-secret")]
hf_secrets: list[modal.Secret] = []


# ─────────────────────────────────────────────────────────────────────
# Helpers — Gemma 4 model loading
# ─────────────────────────────────────────────────────────────────────


def _patch_clippable_linear(*, verbose: bool = True) -> None:
    """Make `Gemma4ClippableLinear` inherit `nn.Linear` so PEFT survives.

    Transformers' Gemma 4 implementation wraps an inner `nn.Linear`
    under `.linear`. PEFT's module-walker keys on the outer module's
    `.weight` attribute and crashes on the vision/audio sub-towers.
    Swapping the class to a real `nn.Linear` subclass keeps eval-time
    forward correct and unblocks any LoRA/PEFT path that scores against
    this baseline.

    Idempotent: re-runs are no-ops once the swap has been applied.
    """
    import torch
    import torch.nn as nn
    try:
        from transformers.models.gemma4 import modeling_gemma4
        orig = modeling_gemma4.Gemma4ClippableLinear
        if issubclass(orig, nn.Linear):
            return

        class _Patched(nn.Linear):
            def __init__(self, config, in_features, out_features):
                nn.Linear.__init__(self, in_features, out_features, bias=False)
                self.use_clipped_linears = getattr(
                    config, "use_clipped_linears", False,
                )
                if self.use_clipped_linears:
                    for buf in ("input_min", "input_max",
                                "output_min", "output_max"):
                        default = -float("inf") if "min" in buf else float("inf")
                        self.register_buffer(buf, torch.tensor(default))

            def forward(self, x):
                if self.use_clipped_linears:
                    x = torch.clamp(x, self.input_min, self.input_max)
                out = nn.Linear.forward(self, x)
                if self.use_clipped_linears:
                    out = torch.clamp(out, self.output_min, self.output_max)
                return out

        modeling_gemma4.Gemma4ClippableLinear = _Patched
        if verbose:
            print("[patch] ClippableLinear -> nn.Linear", flush=True)
    except (ImportError, AttributeError) as e:
        if verbose:
            print(f"[patch] skipped: {e}", flush=True)


def _stage_model(model_id: str, hf_token: str | None) -> str:
    """Download weights to the /models volume once; return local path.

    Re-runs short-circuit on the cached snapshot — saves 30 s to 5 min
    every iteration depending on model size.
    """
    import os
    model_path = f"/models/{model_id.replace('/', '_')}"
    config_check = os.path.join(model_path, "config.json")

    if os.path.exists(config_check):
        size = sum(
            os.path.getsize(os.path.join(model_path, f))
            for f in os.listdir(model_path)
            if os.path.isfile(os.path.join(model_path, f))
        ) / (1024 ** 3)
        print(f"[stage] volume hit: {size:.1f} GB", flush=True)
        return model_path

    print(f"[stage] volume miss; downloading {model_id}", flush=True)
    from huggingface_hub import snapshot_download
    os.makedirs(model_path, exist_ok=True)
    snapshot_download(
        model_id, local_dir=model_path, token=hf_token,
        ignore_patterns=["*.gguf", "*.bin"],
    )
    model_volume.commit()
    print("[stage] committed snapshot", flush=True)
    return model_path


def _load_tokenizer(model_path: str, hf_token: str | None):
    """Load tokenizer + chat template + look up turn-marker IDs.

    Gemma 4's chat template uses `<|turn>` to open a turn and `<turn|>`
    to close it (a rename from Gemma 2/3, which used
    `<start_of_turn>` / `<end_of_turn>`). We resolve their IDs by name
    rather than hardcoding so a tokenizer revision that shifts the
    integer values doesn't silently break label masking or EOS
    termination downstream.
    """
    import os
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    template_path = os.path.join(model_path, "chat_template.jinja")
    if os.path.exists(template_path):
        with open(template_path) as f:
            tokenizer.chat_template = f.read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _resolve_turn_marker_ids(tokenizer)  # raises if missing
    return tokenizer


def _resolve_turn_marker_ids(tokenizer) -> tuple[int, int]:
    """Resolve the (turn-start, turn-end) special-token IDs for Gemma 4.

    Cached on the tokenizer object so repeat calls are free. Raises
    `RuntimeError` if either marker is missing or maps to UNK — which
    means the tokenizer doesn't match the Gemma 4 chat template, and
    everything downstream (label masking, EOS termination) would
    silently misbehave. Better to crash here than to ship a model
    that learned to never stop.
    """
    cached = getattr(tokenizer, "_gemma4_turn_ids", None)
    if cached is not None:
        return cached
    start_id = tokenizer.convert_tokens_to_ids("<|turn>")
    end_id = tokenizer.convert_tokens_to_ids("<turn|>")
    unk = tokenizer.unk_token_id
    if start_id is None or start_id == unk or end_id is None or end_id == unk:
        raise RuntimeError(
            "Gemma 4 turn-marker tokens missing from tokenizer. "
            f"Got <|turn>={start_id}, <turn|>={end_id}, unk={unk}. "
            "Confirm the model is a Gemma 4 checkpoint with the "
            "shipped chat_template.jinja."
        )
    tokenizer._gemma4_turn_ids = (start_id, end_id)
    return start_id, end_id


# ─────────────────────────────────────────────────────────────────────
# In-container eval loop
# ─────────────────────────────────────────────────────────────────────


def _in_container_score_base(
    model_short: str,
    output_label: str,
    max_scenarios: int,
    eval_filename: str,
    seed: int,
):
    """Score one base model against `eval_filename` and write a report."""
    import json
    import os
    import statistics
    import sys
    import time
    from collections import defaultdict

    import torch

    sys.path.insert(0, "/data")
    sys.path.insert(0, os.path.dirname(__file__))

    from _common.eval_scoring import extract_first_turn, run_scoring
    from _common.gemma4_parser import parse_model_output
    from _common.model_registry import get as get_spec

    spec = get_spec(model_short)
    model_id = spec.hf_repo
    enable_thinking = False  # Eval is deterministic; thinking would add noise.

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except (ValueError, OSError) as exc:
            print(f"[hf] login skipped: {type(exc).__name__}: {exc}", flush=True)

    _patch_clippable_linear(verbose=True)
    model_path = _stage_model(model_id, hf_token)

    from transformers import AutoModelForCausalLM

    print(f"\n{'=' * 60}\n  SCORE BASE: {output_label} ({model_id})\n{'=' * 60}",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        token=hf_token,
        # SDPA matches the canonical attention impl per the HF Gemma 4
        # docs. Staying consistent across any train and eval step avoids
        # forward-pass divergence that could mask a real model
        # regression as "just an attention path difference."
        attn_implementation="sdpa",
    )
    tokenizer = _load_tokenizer(model_path, hf_token)

    # Load the chinook tool manifest — passed to the chat template
    # so the model sees the full surface area at every scenario.
    from tool_manifest import MANIFEST as DEFAULT_TOOLS

    eval_path = f"/data/{eval_filename}"
    if not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"Eval set not found: {eval_path}. "
            f"Upload with `modal volume put gemma4-eval-data {eval_filename}`."
        )

    scenarios = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    # Stratified random sample so categories stay roughly balanced.
    if 0 < max_scenarios < len(scenarios):
        import random as _r
        rng = _r.Random(seed)
        by_cat = defaultdict(list)
        for s in scenarios:
            by_cat[s.get("category", "unknown")].append(s)
        sampled = []
        total = len(scenarios)
        for items in by_cat.values():
            n = max(1, round(len(items) / total * max_scenarios))
            sampled.extend(rng.sample(items, min(n, len(items))))
        rng.shuffle(sampled)
        scenarios = sampled[:max_scenarios]

    print(f"[score] {len(scenarios)} scenarios (seed={seed})", flush=True)
    t0 = time.time()
    results = []
    held_out_flags = []
    for i, sc in enumerate(scenarios):
        prompt_msgs, exp_tools, exp_args, cat, exp_checks = extract_first_turn(sc)
        # Per-scenario tool override is allowed for hand-crafted cases;
        # otherwise fall back to the canonical chinook manifest so the
        # model sees a consistent surface across the whole run.
        tools = sc.get("tools") or DEFAULT_TOOLS
        held_out_flags.append(bool(sc.get("held_out", False)))
        try:
            try:
                inputs = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=enable_thinking, tools=tools,
                )
            except (ValueError, TypeError):
                # Older tokenizer revisions reject `enable_thinking`
                # — drop it and continue. Production deployments should
                # pin the tokenizer version, but this keeps the eval
                # working across minor jitters.
                inputs = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True,
                    tools=tools,
                )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            _, turn_end_id = _resolve_turn_marker_ids(tokenizer)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    # Stop on either the canonical EOS or the
                    # end-of-turn marker. Without the turn marker,
                    # the model has been observed to keep generating
                    # past `<turn|>` and produce garbled multi-turn
                    # rollouts.
                    eos_token_id=[tokenizer.eos_token_id, turn_end_id],
                )
            gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=False)
            p_tools, p_args, p_text = parse_model_output(gen)
        except Exception as e:
            if i < 3:
                print(f"  err {i}: {e}", flush=True)
            p_tools, p_args, p_text, gen = [], {}, "", f"ERR:{e}"
        results.append(run_scoring(
            index=i, category=cat, expected_tools=exp_tools, expected_args=exp_args,
            predicted_tools=p_tools, predicted_args=p_args, predicted_text=p_text,
            raw_output=gen[:2000], expected_arg_checks=exp_checks,
        ))
        if (i + 1) % 5 == 0:
            ts = statistics.mean(r.tool_selection for r in results)
            ov = statistics.mean(r.overall for r in results)
            print(
                f"  [{i + 1:3d}/{len(scenarios)}] ts={ts:.1%} overall={ov:.1%}",
                flush=True,
            )

    in_dist = [r for r, h in zip(results, held_out_flags) if not h]
    held_out = [r for r, h in zip(results, held_out_flags) if h]

    def _applicable_semantic_mean(rs):
        applicable = [r for r in rs if r.semantic_checks_applicable]
        if not applicable:
            return None
        return round(statistics.mean(r.semantic_arg_correctness for r in applicable), 4)

    def _axes(rs):
        if not rs:
            return {}
        applicable_semantic = [r for r in rs if r.semantic_checks_applicable]
        return {
            "tool_selection": round(statistics.mean(r.tool_selection for r in rs), 4),
            "argument_correctness": round(
                statistics.mean(r.argument_correctness for r in rs), 4),
            "semantic_arg_correctness": round(
                statistics.mean(r.semantic_arg_correctness for r in rs), 4),
            "semantic_arg_correctness_applicable_only": _applicable_semantic_mean(rs),
            "semantic_checks_applicable_count": len(applicable_semantic),
            "sql_syntax": round(statistics.mean(r.sql_syntax for r in rs), 4),
            "safety": round(statistics.mean(r.safety for r in rs), 4),
            "overall": round(statistics.mean(r.overall for r in rs), 4),
        }

    def _per_category_row(rs):
        applicable_semantic = [r for r in rs if r.semantic_checks_applicable]
        return {
            "count": len(rs),
            "tool_selection": round(statistics.mean(r.tool_selection for r in rs), 4),
            "arg_correctness": round(
                statistics.mean(r.argument_correctness for r in rs), 4),
            "semantic_arg_correctness": round(
                statistics.mean(r.semantic_arg_correctness for r in rs), 4),
            "semantic_arg_correctness_applicable_only": _applicable_semantic_mean(rs),
            "semantic_checks_applicable_count": len(applicable_semantic),
            "overall": round(statistics.mean(r.overall for r in rs), 4),
        }

    in_dist_axes = _axes(in_dist) if in_dist else None
    held_out_axes = _axes(held_out) if held_out else None
    all_axes = _axes(results)

    passed = sum(1 for r in results if r.overall >= 0.7)
    wall = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"  BASELINE — {output_label}", flush=True)
    print(f"  Passed (>=0.7): {passed}/{len(results)} "
          f"({passed / len(results):.0%})", flush=True)
    print(f"  In-dist:  {in_dist_axes}", flush=True)
    print(f"  Held-out: {held_out_axes}", flush=True)
    print(f"  Overall:  {all_axes['overall']:.1%}  ({wall:.0f}s)", flush=True)
    print(f"{'=' * 60}", flush=True)

    cat_results: dict[str, list] = defaultdict(list)
    for r in results:
        cat_results[r.category].append(r)

    result_data = {
        "label": output_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model_id": model_id,
        "adapter_id": None,
        "eval_corpus": eval_filename,
        "seed": seed,
        "scenarios": len(results),
        "passed": passed,
        "wall_clock_s": round(wall, 2),
        "per_axis": all_axes,
        "in_distribution": in_dist_axes,
        "held_out": held_out_axes,
        "overall": all_axes["overall"],
        "per_category": {
            c: _per_category_row(rs) for c, rs in cat_results.items()
        },
    }

    out_dir = f"/output/{output_label}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    output_volume.commit()
    print(f"[score] wrote {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Modal function wrappers
# ─────────────────────────────────────────────────────────────────────


@app.function(
    gpu="B200:1",
    volumes=VOLUME_MOUNTS,
    secrets=hf_secrets,
    timeout=2 * 3600,
    cpu=8,
    memory=32768,
)
def run_score_base(
    model: str = "e4b",
    output_label: str = "",
    max_scenarios: int = 200,
    eval_filename: str = "",
    seed: int = 42,
):
    if not output_label:
        raise ValueError("--output-label is required")
    _in_container_score_base(
        model_short=model,
        output_label=output_label,
        max_scenarios=max_scenarios,
        eval_filename=eval_filename or "chinook_eval_v1.jsonl",
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────


@app.local_entrypoint()
def score_base(
    model: str = "e4b",
    output_label: str = "",
    max_scenarios: int = 200,
    eval_filename: str = "",
    seed: int = 42,
    dry_run: bool = False,
):
    """Score the base Gemma 4 model against the chinook eval set.

    Produces `/output/<output-label>/eval_results.json` on the
    eval-output volume. This is the baseline a fine-tuned adapter is
    later scored against.
    """
    from _common.model_registry import get as get_spec

    spec = get_spec(model)
    if not output_label:
        output_label = f"base-{model}"
    eval_file = eval_filename or "chinook_eval_v1.jsonl"
    print("\n=== Gemma 4 Modal Score (base) ===")
    print(f"  Model:       {spec.hf_repo} ({model})")
    print(f"  Label:       {output_label}")
    print(f"  Eval set:    /data/{eval_file}")
    print(f"  Scenarios:   {max_scenarios}")
    print(f"  Seed:        {seed}")
    if dry_run:
        print("[dry-run] config printed — not launching")
        return
    run_score_base.remote(
        model=model,
        output_label=output_label,
        max_scenarios=max_scenarios,
        eval_filename=eval_file,
        seed=seed,
    )
