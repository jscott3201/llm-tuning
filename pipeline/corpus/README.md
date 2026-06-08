# Corpus — generate a synthetic SFT corpus with a served model as teacher

A Python corpus generator that drives a served, OpenAI-compatible
endpoint as both user-persona AND agent-persona, executes tool calls
against a real chinook database, and writes the full multi-turn message
history to a JSONL file ready for the SFT stage.

The teacher endpoint is any model you have deployed from the serve
stage. A high-concurrency deployment (for example the
`gemma4-26b-concurrent` app) is the cost-appropriate target — the
sparse-MoE checkpoint is cheap to run at concurrency. A dense
deployment such as `gemma4-31b-solo` or `qwen36-27b-concurrent` works
just as well; the generator only needs an OpenAI-compatible
`/v1/chat/completions` URL and the served-model-name the endpoint
advertises.

## Files

| File | What it does |
| --- | --- |
| `chinook_tools.py` | sqlite3-backed executor for the 15-tool chinook agent. Read-only by default; control tools refused for safety. |
| `personas/analyst.json` | "Domain-aware analyst" persona — concise queries, expects structured output. |
| `personas/curious_user.json` | "Exploratory new user" persona — vague questions, open-ended follow-ups. |
| `seeds/v1.jsonl` | Seed user-message prompts the generator draws from. |
| `generate_corpus.py` | The main generator. Drives the agent-loop end to end. |
| `scrub_corpus.py` | Cleans up `<think>...</think>` pollution and renames the `thinking` field to `reasoning` so the chat template renders correctly under SFT. |

The 15-tool manifest itself is defined once in the eval stage
(`eval/tool_manifest.py`) and reused here so the corpus and the eval
rubric share an identical tool surface.

## How the generator works

The agent-loop, per session:

```
seed user prompt
  → agent (tool_call: list_tables)
     → executor returns {tables: [...]}
  → agent (tool_call: describe_table)
     → executor returns {columns: [...]}
  → agent (final assistant message, no tool calls)
  → optionally: user-persona asks a follow-up
  → agent (...)
  → final assistant message
```

The teacher is hit via the OpenAI-compatible `/v1/chat/completions`
route. A high-concurrency teacher endpoint typically runs with the
server-side tool-call parser DISABLED (see the serve stage's notes on
vLLM #39392), so the response comes back with raw
`<|tool_call>...<tool_call|>` tokens in the content; we extract them
with `_common/gemma4_parser.parse_model_output` before executing.

## Run it

```bash
# 1. Make sure a teacher endpoint is up. Deploy one of the serve scripts
#    and use the public *.modal.run URL it prints.
uv run modal deploy serve/sglang/serve_26b.py   # or any serve script

# 2. Make sure chinook.db is present locally.
./scripts/download_chinook.sh

# 3. Generate 100 sessions split across both personas. Pass the URL
#    printed by `modal deploy` as --endpoint and the served-model-name
#    the endpoint advertises as --model.
uv run python corpus/generate_corpus.py \
    --endpoint <the URL printed by modal deploy> \
    --model <served-model-name the endpoint advertises> \
    --persona-dir corpus/personas \
    --seeds corpus/seeds/v1.jsonl \
    --chinook data/chinook.db \
    --out corpus/corpus_v1.jsonl \
    --num-sessions 100 \
    --concurrency 8

# 4. Scrub thinking-format pollution before SFT.
uv run python corpus/scrub_corpus.py \
    corpus/corpus_v1.jsonl \
    corpus/corpus_v1.scrubbed.jsonl
```

## Auth

Auth is your choice and is OFF by default. Modal endpoints are public by
default and can be locked down at the ingress with proxy auth — see
[Modal's endpoint-security docs](https://modal.com/docs/guide/webhook-proxy-auth).
If you do gate the endpoint, point `--api-key-env` at an environment
variable holding the bearer token; the generator reads it and adds an
`Authorization: Bearer` header. Leave it unset for a public endpoint.

## Why we DON'T let the agent execute control tools

`delete_record`, `drop_table`, `truncate_table` are wired in
`chinook_tools.py` to refuse with an explicit error rather than
mutate the database. The agent must learn safety behaviour from the
**scenarios** (the hand-authored eval cases like `safety-drop-no-confirm`),
not from accidental side effects in the corpus. Two reasons:

1. The downloaded chinook.db is a shared file; mutating it would mean
   every regeneration produces a different corpus.
2. The corpus is supposed to teach the model "you may call destructive
   tools when `operator_confirmed=true` is present in the user's
   request." If the executor silently allows them anyway, the corpus
   doesn't carry that signal — the safety axis is uncalibrated.

## Concurrency

The generator's `--concurrency` flag drives an `asyncio.Semaphore` that
caps how many sessions are in flight at once. Match it to whatever the
teacher endpoint can absorb without KV cache evictions; defaults to 8.

If you see `WARN ... CacheEngine: Evicting ... blocks` in the Modal
logs, drop concurrency or lower the serve script's prefill budget
(`max_num_batched_tokens` on vLLM, `chunked_prefill_size` on SGLang).
If you never see those warnings, push higher — idle GPU is paid-for
waste.

## Structured output and Gemma 4 (vLLM #40080)

The default generator does NOT use OpenAI `response_format` /
`json_schema` — it parses raw `<|tool_call>` tokens client-side via
`_common/gemma4_parser.py`. That's deliberate: a high-concurrency
teacher endpoint runs with `--tool-call-parser` disabled (the vLLM
#39392 mitigation), so client-side parsing is the only path that works
at concurrency anyway.

If you adapt the generator to use `response_format` for some other
purpose, be aware that Gemma 4 31B and 26B-A4B can fall into infinite
repetition loops under JSON-schema-constrained generation, especially
with free-form string fields. Mitigations:

- `repetition_penalty >= 1.05` on the request.
- `frequency_penalty >= 0.5`.
- Constrain string fields via regex in the schema where possible —
  unconstrained `"type": "string"` is the worst case.
- Keep `enable_thinking: true` in `chat_template_kwargs`; vLLM #39130
  silently bypasses xgrammar when thinking is off, which can mask the
  problem during testing and resurface in production.
