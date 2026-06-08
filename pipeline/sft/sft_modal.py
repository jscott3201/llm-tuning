#!/usr/bin/env python3
"""LoRA SFT for Gemma 4 on Modal — tool-use corpus.

Trains a LoRA adapter on top of `google/gemma-4-E4B-it` using a
chat-formatted corpus, then optionally pushes the trained adapter
weights to Hugging Face Hub. Designed to run on a single B200 (or
2x L40S) with the preset's default settings.

Five Gemma 4 fine-tune fixes that this script preserves:

  (1) `add_special_tokens=False` — the chat template emits BOS.
  (2) Load the sibling `chat_template.jinja` manually after
      `AutoTokenizer.from_pretrained(...)`.
  (3) `enable_thinking` passed consistently with a graceful fall-back
      for older tokenizers that reject the kwarg.
  (4) Label masking on assistant turns only. Gemma 4's chat template
      uses `<|turn>` / `<turn|>` as turn markers (renamed from Gemma
      2/3's `<start_of_turn>` / `<end_of_turn>`). The masking logic
      below resolves their IDs at runtime via
      `tokenizer.convert_tokens_to_ids(...)` and uses a prefix-delta
      strategy that doesn't depend on the IDs at all — the assistant
      message token range is computed by re-encoding successive
      message prefixes and diffing.
  (5) EOS set to `[<eos>, <turn|>]` for generation, looked up by name
      so a tokenizer revision that shifts integer IDs can't silently
      break termination.

Plus the sixth gotcha — `_patch_clippable_linear` makes
`Gemma4ClippableLinear` inherit `nn.Linear` so PEFT's module walker
doesn't crash on Gemma 4's wrapped linear layers (vision + audio
sub-towers carry these and PEFT's default walker keys on the outer
`.weight` attribute).

## Setup (one-time)

1. `modal token new` — authenticate the CLI.
2. Gemma 4 is Apache 2.0, NOT gated — no HF token required to pull the
   base weights. A `huggingface-secret` (a Modal Secret you create in
   your own workspace holding `HF_TOKEN`) is only needed to
   `--push-to-hub` the trained adapter to a private repo, and gives
   authenticated rate limits on Hub pulls.

## Run

    modal run --detach sft/sft_modal.py::train \\
        --preset sft/presets/e4b_sft.json
"""

from __future__ import annotations

import modal

APP_NAME = "gemma4-e4b-sft"

# Shared HF cache across the pipeline — the serve/eval scripts use the
# same volume name, so base weights pulled here are already warm there
# (and vice-versa). Modal Volumes handle concurrent reads fine.
model_volume = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
# Training corpus lives here (put it with `modal volume put gemma4-data ...`).
data_volume = modal.Volume.from_name("gemma4-data", create_if_missing=True)
# Trained adapters + run metadata land here.
output_volume = modal.Volume.from_name("gemma4-sft-output", create_if_missing=True)

VOLUME_MOUNTS = {
    "/models": model_volume,
    "/data": data_volume,
    "/output": output_volume,
}

# CUDA devel base so any wheels that need nvcc at install time build
# cleanly. uv resolves the CUDA wheel graph far faster than vanilla pip.
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
        "transformers==5.5.4",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "datasets>=4.8",
        "huggingface_hub[hf_xet]>=1.11,<2",
        "safetensors>=0.7",
        "sentencepiece",
        "protobuf",
        "einops",
        "trl>=0.16",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("_common")
)

app = modal.App(APP_NAME, image=image)

# Optional — only needed for `--push-to-hub` to a private repo, and for
# authenticated Hub rate limits. The Gemma 4 base weights are
# ungated/public, so pulling them needs no token. `huggingface-secret`
# is a Modal Secret you create in your own workspace holding `HF_TOKEN`.
SECRETS: list[modal.Secret] = [
    modal.Secret.from_name("huggingface-secret"),
]


# ─────────────────────────────────────────────────────────────────────
# Gemma 4 patches (shared shape with the eval-time forward path)
# ─────────────────────────────────────────────────────────────────────


def _patch_clippable_linear(*, verbose: bool = True) -> None:
    """Make `Gemma4ClippableLinear` inherit `nn.Linear` so PEFT survives.

    Gemma 4's vision/audio sub-towers wrap a real `nn.Linear` under an
    outer module whose `.weight` PEFT's default walker keys on; the
    stock `Gemma4ClippableLinear` does not inherit `nn.Linear`, so the
    LoRA initialisation pass crashes when it walks those layers. Making
    it a true `nn.Linear` subclass keeps the clamp semantics intact
    while satisfying the walker.
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
    import os
    model_path = f"/models/{model_id.replace('/', '_')}"
    if os.path.exists(os.path.join(model_path, "config.json")):
        return model_path
    print(f"[stage] downloading {model_id}", flush=True)
    from huggingface_hub import snapshot_download
    os.makedirs(model_path, exist_ok=True)
    snapshot_download(
        model_id, local_dir=model_path, token=hf_token,
        ignore_patterns=["*.gguf", "*.bin"],
    )
    model_volume.commit()
    return model_path


def _load_tokenizer(model_path: str, hf_token: str | None):
    """Load tokenizer + chat template + look up turn-marker IDs.

    Gemma 4's chat template uses `<|turn>` to open a turn and `<turn|>`
    to close it (a rename from Gemma 2/3, which used
    `<start_of_turn>` / `<end_of_turn>`). Resolve them by name rather
    than hardcoding integer IDs — keeps the SFT robust against
    tokenizer revisions that shift the integer values.
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
    _resolve_turn_marker_ids(tokenizer)
    return tokenizer


def _resolve_turn_marker_ids(tokenizer) -> tuple[int, int]:
    """Cache + return (turn-start, turn-end) IDs. Raises on missing."""
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
# Dataset prep — chat-template formatting + label masking
# ─────────────────────────────────────────────────────────────────────


def _format_record(
    rec: dict, tokenizer, *, max_seq_length: int,
) -> dict | None:
    """Convert one corpus record into `(input_ids, attention_mask, labels)`.

    Label-masking strategy (in priority order):

      1. **Modern path — `return_assistant_tokens_mask=True`.** Some
         chat templates wrap assistant generation regions in
         `{% generation %}...{% endgeneration %}` Jinja blocks; when
         present, the tokenizer can return a parallel mask flagging
         which tokens belong to assistant turns. This is the cleanest
         signal — no token-ID guesswork, robust to template rewrites.
      2. **Fallback — prefix-delta encoding.** Render the chat
         template once for `messages[:i+1]` for every `i`. The token
         range that appears in iteration `i` but not iteration `i-1`
         is exactly the contribution of `messages[i]`. Mark only the
         deltas from assistant messages as supervised. This is O(N²)
         in the message count but works against any chat template
         shape and never relies on hardcoded special-token IDs.

    Returns None if the record is malformed (missing messages, empty
    template output, or no supervised tokens after masking) — we drop
    those rather than crash the run.
    """
    messages = rec.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    # Path 1: assistant-tokens-mask via a generation-block-aware template.
    modern = _format_via_assistant_mask(messages, tokenizer, max_seq_length)
    if modern is not None:
        return modern

    # Path 2: prefix-delta encoding.
    return _format_via_prefix_delta(messages, tokenizer, max_seq_length)


def _format_via_assistant_mask(messages, tokenizer, max_seq_length):
    """Try `return_assistant_tokens_mask=True`. Returns None if the
    chat template doesn't expose generation regions (most current
    templates don't, including the stock Gemma 4 one — but we try
    anyway because it's cheap and future-proofs against a template
    update)."""
    try:
        try:
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False, enable_thinking=False,
            )
        except (ValueError, TypeError):
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
            )
    except Exception:
        return None

    mask = result.get("assistant_masks")
    ids = result.get("input_ids")
    if not mask or not ids or not any(mask):
        return None

    # `apply_chat_template` may return a 2-D batch even with one
    # conversation; flatten to 1-D for our single-record case.
    if isinstance(ids[0], list):
        ids = ids[0]
        mask = mask[0]

    ids = ids[:max_seq_length]
    mask = mask[:max_seq_length]
    labels = [tok if m else -100 for tok, m in zip(ids, mask)]
    if not any(lbl != -100 for lbl in labels):
        return None
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


def _format_via_prefix_delta(messages, tokenizer, max_seq_length):
    """Compute label mask by encoding successive prefixes.

    For each `i`, encode `messages[:i+1]` and compare its length to the
    previous prefix's length — the delta is the token range introduced
    by `messages[i]`. Mark deltas from assistant messages as supervised
    (loss-bearing); leave everything else as -100.

    O(N²) in message count but bulletproof: works against any chat
    template, with or without thinking enabled, with or without
    custom turn markers.
    """
    def _encode(msgs):
        try:
            try:
                ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True,
                    add_generation_prompt=False, enable_thinking=False,
                )
            except (ValueError, TypeError):
                ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False,
                )
        except Exception:
            return None
        return ids

    full_ids = _encode(messages)
    if not full_ids:
        return None
    full_ids = full_ids[:max_seq_length]
    labels = [-100] * len(full_ids)

    prev_len = 0
    for i, msg in enumerate(messages):
        prefix_ids = _encode(messages[: i + 1])
        if not prefix_ids:
            continue
        cur_len = min(len(prefix_ids), max_seq_length)
        if cur_len <= prev_len:
            # Pathological: prefix didn't grow. Skip but keep going.
            continue
        if msg.get("role") == "assistant":
            for j in range(prev_len, cur_len):
                labels[j] = full_ids[j]
        prev_len = cur_len
        if cur_len >= max_seq_length:
            break

    if not any(lbl != -100 for lbl in labels):
        return None
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _build_dataset(corpus_path: str, tokenizer, *, max_seq_length: int):
    import json
    from datasets import Dataset

    rows: list[dict] = []
    skipped = 0
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            formatted = _format_record(
                rec, tokenizer, max_seq_length=max_seq_length,
            )
            if formatted is None:
                skipped += 1
                continue
            rows.append(formatted)
    print(
        f"[data] {len(rows)} records loaded, {skipped} skipped",
        flush=True,
    )
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────


def _in_container_train(preset: dict, push_to_hub: bool) -> None:
    import json
    import os

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except (ValueError, OSError) as exc:
            print(f"[hf] login skipped: {type(exc).__name__}: {exc}", flush=True)

    _patch_clippable_linear(verbose=True)

    base_id = preset["base_model_id"]
    out_repo = preset["output_repo_id"]
    data_cfg = preset["data"]
    lora_cfg = preset["lora"]
    train_cfg = preset["training"]

    model_path = _stage_model(base_id, hf_token)
    tokenizer = _load_tokenizer(model_path, hf_token)

    print(f"[load] base model {base_id}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        # SDPA is the canonical attention implementation per the
        # Hugging Face Gemma 4 model docs and Google's official
        # fine-tuning recipe. Avoids FlashAttention2 quirks on the
        # Gemma 4 sliding-window pattern.
        attn_implementation="sdpa",
    )

    print("[lora] wrapping base in PEFT LoRA", flush=True)
    # `target_modules` accepts either a literal list or the special
    # string "all-linear". The latter is Google's canonical
    # recommendation in the Gemma 4 fine-tuning recipe — it scopes
    # to the language model's linear layers via PEFT's regex matcher
    # and sidesteps Gemma 4's vision/audio `Gemma4ClippableLinear`
    # wrappers (which don't inherit from `nn.Linear` and would
    # otherwise trip the module walker, see PEFT issues around 0.19).
    target_modules = lora_cfg["target_modules"]
    if isinstance(target_modules, list):
        target_modules = list(target_modules)
    modules_to_save = lora_cfg.get("modules_to_save")
    if modules_to_save:
        modules_to_save = list(modules_to_save)
    else:
        modules_to_save = None

    # `ensure_weight_tying` is a PEFT 0.19+ kwarg. Critical when
    # `modules_to_save` includes BOTH `embed_tokens` and `lm_head`:
    # Gemma 4 ties these by default and full fine-tuning of both
    # without re-tying lets them drift apart (the model can still
    # generate but with degraded coherence). Pass-through is opt-in
    # via the preset so older PEFT versions don't break.
    ensure_tying = bool(lora_cfg.get("ensure_weight_tying", False))
    lora_kwargs: dict = dict(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
        bias="none",
    )
    if ensure_tying:
        lora_kwargs["ensure_weight_tying"] = True

    lora = LoraConfig(**lora_kwargs)
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    dataset = _build_dataset(
        data_cfg["corpus_jsonl"],
        tokenizer,
        max_seq_length=int(data_cfg.get("max_seq_length", 8192)),
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset is empty after formatting; aborting")

    output_dir = f"/output/sft-{out_repo.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("epochs", 1.0)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(train_cfg.get("per_device_batch_size", 4)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        seed=int(train_cfg.get("seed", 42)),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    # Padding-to-multiple-of-8 keeps tensor cores happy on bf16.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"[train] starting — {len(dataset)} examples, "
          f"{train_cfg.get('epochs', 1.0)} epochs", flush=True)
    trainer.train()
    print("[train] complete", flush=True)

    final_dir = f"{output_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    output_volume.commit()
    print(f"[save] adapter written to {final_dir}", flush=True)

    # Stash the preset alongside the adapter for reproducibility — it's
    # the only "what was this run?" answer that survives Modal's
    # log retention window.
    with open(f"{final_dir}/preset.json", "w") as f:
        json.dump(preset, f, indent=2)
    output_volume.commit()

    if push_to_hub and out_repo:
        from huggingface_hub import HfApi
        if not hf_token:
            raise RuntimeError("HF_TOKEN required for --push-to-hub")
        print(f"[push] uploading adapter to hf.co/{out_repo}", flush=True)
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=out_repo, exist_ok=True, private=True)
        api.upload_folder(
            repo_id=out_repo, folder_path=final_dir, commit_message="LoRA adapter",
        )
        print("[push] done", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Modal wrappers
# ─────────────────────────────────────────────────────────────────────


@app.function(
    gpu="B200:1",
    volumes=VOLUME_MOUNTS,
    secrets=SECRETS,
    timeout=6 * 3600,
    cpu=8,
    memory=65536,
)
def run_train(preset: dict, push_to_hub: bool = False) -> None:
    _in_container_train(preset, push_to_hub)


@app.local_entrypoint()
def train(
    preset: str = "sft/presets/e4b_sft.json",
    push_to_hub: bool = False,
    dry_run: bool = False,
):
    """Run a LoRA SFT against the Gemma 4 base model.

    `--preset` points at a JSON file with the LoRA + training config
    (see `sft/presets/e4b_sft.json` for the canonical shape).
    `--push-to-hub` uploads the trained adapter to the
    `output_repo_id` in the preset (requires `HF_TOKEN` in the mounted
    `huggingface-secret`).
    """
    import json as _json
    from pathlib import Path

    preset_data = _json.loads(Path(preset).read_text())
    print("\n=== Gemma 4 LoRA SFT ===")
    print(f"  Base:        {preset_data['base_model_id']}")
    print(f"  Output:      {preset_data['output_repo_id']}")
    print(f"  Corpus:      {preset_data['data']['corpus_jsonl']}")
    print(f"  Push to Hub: {push_to_hub}")
    if dry_run:
        print("[dry-run] config printed — not launching")
        return
    run_train.remote(preset_data, push_to_hub)
