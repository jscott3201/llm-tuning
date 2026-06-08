# pipeline

The research path: take a base model, measure it, generate a synthetic training
corpus, fine-tune a smaller model, and check it didn't lose general capability.
The running example is a SQL agent over the public
[Chinook](https://github.com/lerocha/chinook-database) music-store database.
Serving here is on vLLM (the SGLang production path lives in `../gemma4` and
`../qwen`).

## The four stages

| Stage | Dir | What it does |
|---|---|---|
| serve | `serve/vllm/` | Deploy a Gemma 4 size on vLLM/Modal (`serve_{e2b,e4b,12b,26b,31b}.py`) |
| eval | `eval/` | Score the model on a five-axis tool-use rubric over a Chinook scenario set |
| corpus | `corpus/` | Generate a multi-turn SFT corpus with a larger model as both user and agent |
| sft | `sft/` | LoRA fine-tune a small model, then gate it against capability drift |

Read them in order. Each has its own README.

## Setup

```bash
uv sync
uv run modal token new
./scripts/download_chinook.sh        # fetch the sample DB (stages 2–4)
```

## The eval rubric

Five weighted axes: tool selection (F1 with partial credit for related tools),
argument correctness, semantic argument checks, SQL syntax, and safety. The
semantic axis is what makes it discriminating — it catches calls that are
structurally valid but wrong. The scoring core is in `_common/eval_scoring.py`
and is model-agnostic; the Chinook-specific tool clusters are a default you can
override.

```bash
uv run --group dev pytest eval/test_eval_scoring.py -v
```

## Corpus generation

`corpus/generate_corpus.py` drives a served model as both a user persona and the
agent, running real tool calls against a local Chinook database and writing
OpenAI-format JSONL ready for SFT. The control tools (delete, drop, truncate) are
wired to refuse rather than mutate, so the model learns safety from the scenario,
not from accidents.

## SFT

`sft/sft_modal.py` LoRA fine-tunes on Modal with the Gemma 4-specific handling
baked in (turn markers, label masking, `ClippableLinear`, weight tying). After
training, re-run the eval and the subliminal-learning probe under
`sft/probes/` to confirm the adapter didn't regress on held-out public
benchmarks.

## Notes

- vLLM and Modal are pinned in the image builders; the local environment only
  needs `modal` and `openai`.
- No weights, corpora, or secrets live in the tree. Chinook is fetched by script;
  SFT writes to your own Modal volumes.
- Endpoints are public by default — see
  [../docs/securing-endpoints.md](../docs/securing-endpoints.md).
