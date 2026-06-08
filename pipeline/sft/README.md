# SFT — LoRA fine-tune E4B and verify no regression

LoRA fine-tune `google/gemma-4-E4B-it` on a chat-formatted corpus, then
verify the resulting adapter (a) hasn't regressed on the eval rubric the
base model scored on, and (b) hasn't lost capability on a 4-axis
public-benchmark probe ("subliminal-learning" guard).

## Files

| File | What it does |
| --- | --- |
| `sft_modal.py` | Modal LoRA SFT trainer with the Gemma 4-specific patches. |
| `presets/e4b_sft.json` | LoRA + training hyperparameters for the E4B SFT run. |
| `compare_eval.py` | Compare a baseline `eval_results.json` against a post-SFT one. |
| `probes/subliminal_v1/` | 4-axis capability probe (MMLU-STEM, GSM8K, IFEval, HumanEval) with anchor + compare scripts. |

## Why two regression gates

A successful SFT can do three things at once: improve the in-domain
metric, leave general capability untouched, and *not* fundamentally
shift behaviour on tasks unrelated to the training data. The two gates
catch different failure modes:

1. **Eval rubric regression** (`compare_eval.py`) — the adapter scored
   below the base model on the tool-use rubric. Means SFT didn't help,
   or actively hurt, on the in-domain task.
2. **Subliminal-learning regression** (`probes/subliminal_v1/`) — the
   adapter dropped on a *different* task family the SFT corpus didn't
   touch. Same-family distillation can transmit unintended behaviour
   shifts via unrelated data, so a 4-axis sanity probe is the cheapest
   guard against it.

A clean adapter clears both.

## Run it

```bash
# 1. Put the training corpus on the data volume.
uv run modal volume put gemma4-data \
    corpus/corpus_v1.scrubbed.jsonl /corpus_v1.scrubbed.jsonl

# 2. Pre-SFT subliminal anchor — score the BASE model. Point --endpoint
#    at the URL `modal deploy serve/vllm/serve_e4b.py` printed.
uv run python sft/probes/subliminal_v1/fetch_probe.py
uv run python sft/probes/subliminal_v1/score_probe.py \
    --endpoint <base-e4b-url> \
    --model google/gemma-4-E4B-it \
    --label pre-sft-anchor \
    --out sft/probes/subliminal_v1/anchors/e4b-pre-sft.json

# 3. Train the adapter. (Edit output_repo_id in the preset first.)
uv run modal run --detach sft/sft_modal.py::train \
    --preset sft/presets/e4b_sft.json \
    --push-to-hub

# 4. Re-deploy with the adapter loaded (see the LoRA-serving section of
#    serve/vllm/serve_e4b.py — copy it to serve_e4b_adapter.py).
# 5. Score the adapter on the eval set.
uv run modal run --detach eval/score_base_modal.py::score_base \
    --model e4b \
    --output-label sft-e4b-v1

# 6. Compare against the baseline.
uv run modal volume get gemma4-eval-output base-e4b-v1/eval_results.json /tmp/
uv run modal volume get gemma4-eval-output sft-e4b-v1/eval_results.json /tmp/
uv run python sft/compare_eval.py \
    --baseline /tmp/base-e4b-v1.json \
    --candidate /tmp/sft-e4b-v1.json

# 7. Re-run the subliminal probe against the served adapter and compare.
uv run python sft/probes/subliminal_v1/score_probe.py \
    --endpoint <adapter-url> \
    --model my-adapter \
    --label post-sft-candidate \
    --out /tmp/post-sft-candidate.json

uv run python sft/probes/subliminal_v1/compare_anchor.py \
    --anchor sft/probes/subliminal_v1/anchors/e4b-pre-sft.json \
    --candidate /tmp/post-sft-candidate.json
```

`compare_anchor.py` exits 0 on pass, 1 on fail. Both gates must pass
before promoting the adapter to "shippable."

The training job mounts three Modal Volumes (auto-created on first run):
`gemma4-hf-cache` (shared base-weight cache with the serve/eval stages),
`gemma4-data` (training corpus), and `gemma4-sft-output` (trained
adapters + run metadata).

## The five Gemma 4 fine-tune fixes

These are all baked into `sft_modal.py`. They show up as numbered
inline notes in the code:

1. `add_special_tokens=False` when calling the tokenizer — the chat
   template already emits BOS.
2. `AutoTokenizer.from_pretrained(...)` + load the sibling
   `chat_template.jinja` manually; don't rely on `tokenizer_config`.
3. `enable_thinking` passed consistently to `apply_chat_template` and
   `model.generate()`; fall back without it if the tokenizer version
   rejects the kwarg.
4. Label masking on **assistant turns only**. Gemma 4's chat template
   uses `<|turn>` to open a turn and `<turn|>` to close it — a rename
   from Gemma 2/3's `<start_of_turn>` / `<end_of_turn>`. The trainer
   resolves these IDs by name (`tokenizer.convert_tokens_to_ids(
   "<|turn>")`) and uses a *prefix-delta* strategy: each assistant
   message's token range is computed by re-encoding `messages[:i+1]`
   for each `i` and diffing against the previous prefix's length.
   That means label masking works regardless of which integer IDs
   Gemma's tokenizer assigns to its special tokens — bulletproof
   against tokenizer revisions.
5. EOS set to `[<eos>, <turn|>]` for generation, looked up by name
   so termination doesn't rely on hardcoded integer IDs.

The `Gemma4ClippableLinear` patch (the `_patch_clippable_linear`
function in `sft_modal.py`) is the sixth gotcha — PEFT's module walker
chokes on Gemma 4's wrapped linear layers (vision + audio sub-towers
nest a real `nn.Linear` under `.linear` and PEFT walks for an outer
`.weight`) without it.

## Serving a trained LoRA adapter

vLLM 0.19.1 ships LoRA support for `Gemma4ForConditionalGeneration`
via PR #39291 (which fixed feature request #39246). Scope:

- **Language backbone is LoRA-able** — q_proj, k_proj, v_proj,
  o_proj, gate_proj, up_proj, down_proj. The preset in
  `presets/e4b_sft.json` targets exactly these.
- **Vision and audio towers are NOT yet LoRA-able** — they still
  use HF's auto-model path internally and are deferred to follow-up
  work. Text-only fine-tuning is unaffected.

Recipe — copy `serve/vllm/serve_e4b.py` to a sibling
`serve_e4b_adapter.py`, rename the Modal app, and pass through
`extra_args`:

```python
extra_args=[
    "--enable-lora",
    "--lora-modules", "my-adapter=your-hf-username/gemma4-e4b-sft",
    "--max-lora-rank", "16",   # match LoraConfig.r in the preset
]
```

The adapter and the base each get their own served-model name and
URL, so the A/B comparison can score both side by side.

**Alternative: merge then serve.** Call `model.merge_and_unload()` at
the end of the SFT run, push the merged weights to your HF account as
a full model, and serve with the existing base-serve script pointed
at the merged repo. Loses dynamic-adapter flexibility but produces a
self-contained deployable artefact.
