# Subliminal-learning probe v1

A 4-axis capability-regression guard for the SFT stage. Same-family
distillation (e.g. using a larger Gemma 4 as a teacher to fine-tune
Gemma 4 E4B) can transmit unintended behaviour shifts via tasks the
training corpus doesn't directly cover; this probe samples four public
benchmarks the SFT corpus *doesn't* touch and gates the adapter against
drift on any of them.

## When to run

- **Pre-SFT** — score the **base** E4B endpoint to produce the
  baseline anchor.
- **Post-SFT** — score the adapter-aware endpoint with the same probe
  and `compare_anchor.py` against the anchor.

## Pass thresholds

- Composite (unweighted mean of per-axis accuracy): `|delta| <= 0.02`
- Any single axis: `|delta| <= 0.03`

The thresholds are two-sided. A *rise* on any capability axis is also
flagged as drift — large unexpected swings in either direction
indicate the SFT corpus changed something it shouldn't have.

If either threshold fails, the adapter is quarantined pending
investigation (corpus sampling, teacher-output review, etc.).

## The four axes

| Axis | n | Source | What it catches |
|------|---|--------|-----------------|
| `mmlu_stem` | 50 | `cais/mmlu` (7 STEM subjects, stratified) | General reasoning regression. |
| `gsm8k` | 50 | `openai/gsm8k` (main / test) | Grade-school arithmetic word problems. |
| `ifeval` | 50 | `google/IFEval` | Machine-checkable instruction-following constraints (sentence count, forced ending, no-comma, all-caps, keyword presence). |
| `humaneval` | 50 | `openai/openai_humaneval` | Python function synthesis. Filtered to canonical solutions ≤20 LOC so sandbox execution stays cheap. |

## How to run

```bash
# Install probe deps (or `uv sync --extra probe` if your top-level
# pyproject declares the extra).
pip install -r requirements.txt

# 1. Fetch + commit samples (only needed if samples/ is empty).
uv run python sft/probes/subliminal_v1/fetch_probe.py

# 2. Pre-SFT anchor. Point --endpoint at the URL `modal deploy` printed.
uv run python sft/probes/subliminal_v1/score_probe.py \
    --endpoint <base-e4b-url> \
    --model google/gemma-4-E4B-it \
    --label pre-sft-anchor \
    --out sft/probes/subliminal_v1/anchors/e4b-pre-sft.json

# 3. Post-SFT candidate.
uv run python sft/probes/subliminal_v1/score_probe.py \
    --endpoint <adapter-url> \
    --model my-adapter \
    --label post-sft-candidate \
    --out /tmp/post-sft-candidate.json

# 4. Compare. Exit code 0 on PASS, 1 on FAIL.
uv run python sft/probes/subliminal_v1/compare_anchor.py \
    --anchor sft/probes/subliminal_v1/anchors/e4b-pre-sft.json \
    --candidate /tmp/post-sft-candidate.json \
    --out /tmp/comparison.json
```

`OPENAI_API_KEY` is read from the environment when set and sent as a
Bearer header. Modal endpoints are public by default and ignore it; if
you locked yours down with proxy auth, set the matching token here. See
modal.com/docs/guide/webhook-proxy-auth.

## Reproducibility

- `fetch_probe.py` defaults to `--seed 42`. Re-running with the same
  seed against the same `datasets` release yields a byte-identical
  JSONL.
- Commit the `samples/` directory so running the probe end to end does
  not need network access to Hugging Face.
- `requirements.txt` pins `datasets>=4.0,<5.0`; bumping the major
  version may shift dataset revisions and force a fresh anchor.

## Output schema

`score_probe.py --out` writes:

```json
{
  "label": "pre-sft-anchor",
  "model": "google/gemma-4-E4B-it",
  "endpoint": "<the URL printed by modal deploy>",
  "axes": {
    "mmlu_stem":   {"accuracy": 0.78, "n": 50, "correct": 39, "skipped": 0, "errors": 0},
    "gsm8k":       {"accuracy": 0.62, "n": 50, "correct": 31, "skipped": 0, "errors": 0},
    "ifeval":      {"accuracy": 0.71, "n": 50, "correct": 32, "skipped": 5, "errors": 0},
    "humaneval":   {"accuracy": 0.48, "n": 50, "correct": 24, "skipped": 0, "errors": 0}
  },
  "composite": 0.6475,
  "timestamp": "..."
}
```

`skipped` only fires for `ifeval` items whose constraint kind isn't
in the scorer's supported set. Skipped items are excluded from the
denominator (they don't count against the axis score).
