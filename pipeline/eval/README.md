# Eval — multi-axis tool-use scoring and base-model baseline

A 5-axis tool-use eval rubric for a SQL agent over the public
[chinook](https://github.com/lerocha/chinook-database) sample database.
Run the base Gemma 4 model against the scenarios here to get a baseline
score that a fine-tuned model can later be measured against.

## Files

| File | What it does |
| --- | --- |
| `tool_manifest.py` | OpenAI-compatible tool schemas for the 15-tool chinook SQL agent. Imported by any script that talks to a served endpoint. |
| `scenarios/chinook_eval_v1.jsonl` | Hand-authored eval scenarios (one per line, OAI-messages shape). |
| `score_base_modal.py` | Modal app that loads the base model with HF Transformers, runs each scenario, scores against the rubric, and writes a per-axis JSON report. |
| `generate_arg_checks.py` | Auto-generator for `expected_arg_checks` (the semantic-arg axis input) — scans a scenario file and emits SQL-shaped checks where it can recognise a high-confidence pattern. Plain local script, no cloud dependency. |
| `test_eval_scoring.py` | Pytest suite for `_common/eval_scoring.py` — the reference for what each axis is supposed to do. |

The scoring core itself lives in `_common/eval_scoring.py`; this stage
only adds the chinook-specific tool manifest, scenarios, and the Modal
runner. Scripts import the shared package with
`from _common.<module> import <name>` and mount it into the Modal image
via `.add_local_python_source("_common")`.

## The five axes

| Axis | Weight | Signal |
| --- | --- | --- |
| `tool_selection` | 0.35 | F1 with partial credit for siblings in `RELATED_GROUPS`. |
| `argument_correctness` | 0.15 | Key overlap + value match on shared tool calls. |
| `semantic_arg_correctness` | 0.25 | Declared `expected_arg_checks` (Contains / Equals / Matches) — fires only on scenarios that opt in, neutral 1.0 otherwise. |
| `sql_syntax` | 0.10 | Light parse: balanced parens/brackets, recognised SQL opener, no NoSQL-isms. Skipped on tools whose query argument is natural-language (`find_artist`, etc.). |
| `safety` | 0.15 | Control-surface tools without `operator_confirmed=true`, plus regex-flagged unsafe patterns in the predicted text. |

`overall = Σ axis × weight`. Default weights live in
`_common/eval_scoring.WEIGHTS`; pass `weights=` to `run_scoring(...)`
if you want to lock in a specific historical weighting for a
reproducibility anchor.

## Run it

These Gemma 4 checkpoints are ungated/public, so no Hugging Face token
is needed and a fresh Modal workspace can run this with zero setup. (To
point a deploy at a gated repo, create a Modal secret named
`huggingface-secret` holding `HF_TOKEN` and attach it — see the comment
at the top of `score_base_modal.py`.)

```bash
# 1. Upload the eval scenarios to the eval-data Modal volume.
uv run modal volume put gemma4-eval-data \
    eval/scenarios/chinook_eval_v1.jsonl

# 2. Score the base 4B model.
uv run modal run --detach eval/score_base_modal.py::score_base \
    --model e4b \
    --output-label base-e4b-v1

# 3. Read the report back from the output volume.
uv run modal volume get gemma4-eval-output \
    base-e4b-v1/eval_results.json /tmp/
cat /tmp/eval_results.json | jq .
```

`--model` accepts any short name in `_common/model_registry`
(`e2b` / `e4b` / `12b` / `26b` / `31b`). The default GPU class in the
runner is `B200:1`; adjust it on the `@app.function(...)` decorator to
match the size you score (an L40S is plenty for `e4b`).

## Auto-generating semantic checks

The semantic-arg axis is the most "expensive" axis to author —
specifying every `expected_arg_checks` entry by hand for a large
scenario set is real work. `generate_arg_checks.py` reads a scenario
file and emits checks for the patterns it can recognise with high
confidence:

- Chinook table names → `equals` on the SQL `FROM <Table>` shape
- Numeric values on identifier-shaped keys (`limit`, `count`, `offset`,
  `top_k`) → `equals`
- Boolean flags → `equals`
- Track/album/artist names on the natural-language search tools →
  `contains`

Scenarios that already declare checks are preserved verbatim
(idempotent). Pass `--force-regenerate` to overwrite.

```bash
uv run python eval/generate_arg_checks.py \
    --in eval/scenarios/chinook_eval_v1.jsonl \
    --out eval/scenarios/chinook_eval_v1.enriched.jsonl
```

## Testing the rubric

```bash
# From the pipeline root (the directory that contains _common/):
uv run pytest eval/test_eval_scoring.py -v
```

## Authoring a new scenario

```jsonc
{
  "id": "catalog-list-tables",
  "category": "catalog_inspection",
  "messages": [
    {"role": "system", "content": "You are a SQL analyst..."},
    {"role": "user",   "content": "What tables are in this database?"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"function": {"name": "list_tables", "arguments": "{}"}}
    ]}
  ],
  "expected_arg_checks": [],   // optional — auto-fill via the generator
  "tools": [...]               // optional — defaults to `tool_manifest.MANIFEST`
}
```

The scorer reads the **first** assistant turn after the **first** user
turn as the expectation. Anything beyond that is ignored — the eval
measures one-shot tool selection, not multi-turn dialog.

A scenario can set `"held_out": true` to be reported separately from
the in-distribution split in the JSON report.
