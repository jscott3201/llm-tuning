# Testing the Qwen3.6 chat-template fork

The fork is derived from the Qwen3.6-27B template and is served for both the 27B
and 35B-A3B deployments. How the Q1-Q8 patches were validated. Three layers: an offline conformance
suite that renders the Jinja directly, a live agentic probe that drives a real
endpoint, and an A/B coding benchmark that checks the fork costs nothing on the
happy path.

## 1. Offline conformance suite

`../tests/test_custom_pub_chat_template_qwen36.py`. 39 tests. They render both
`qwen36_upstream.jinja` and `custom_pub_chat_template_qwen36.jinja` with Jinja2
and compare the output. No model, no GPU, no network. This is where every patch
gets its contract pinned.

Run it with uv from the `qwen/` directory:

```bash
uv run --with jinja2 --with pytest \
  python -m pytest tests/test_custom_pub_chat_template_qwen36.py -v
```

You should see `39 passed`. Plain pytest works too if jinja2 and pytest are
already installed:

```bash
python3 -m pytest tests/test_custom_pub_chat_template_qwen36.py -v
```

The suite asserts three contracts.

**Byte-identity to upstream (prefix-cache invariant).** With the three gated
patches disabled (`preserve_thinking=false`, `unwrap_tool_envelope=false`,
`verbose_tool_instructions=false`) and on inputs that don't hit a bug surface,
the fork renders byte-for-byte identical to upstream. T0-T11 cover bare user,
system+user, tools, full tool round-trips, multi-tool fanout, scalar and nested
argument types, and multimodal content. T5 pins `preserve_thinking=true`
byte-identity. T18 is the tight one: even pathological `</think>` bodies
(multiple close tags, nested openers, a reasoning body that literally contains
the token `<think>`) must match upstream's last-opener / first-closer split
exactly. Any drift here means a patch leaked into a path it shouldn't have.

**Strict-where-upstream-silent.** Every shape upstream handles is handled
identically or strictly safer. T12 (Q2) accepts the `developer` role. T13 (Q3)
raises a clear error on string-typed `tool_call.arguments`. T14/T15 (Q4)
extract reasoning from `</thinking>` and rescue an unclosed `<think>` with a
tool call inside. T16 (Q5) unwraps the OpenAI tool envelope. T17 (Q6) gates the
verbose IMPORTANT block.

**Agentic-coding scenarios (Sxx).** These reproduce the documented public bug
shapes end-to-end through the renderer and prove the fork prevents them. S01/S02
are the pi#3325 multi-turn thinking case. S03 is the opencode developer role.
S05 is the Vercel AI SDK string-arguments crash. S07/S07b/S07c/S07d are the
mid-conversation system/developer steering messages (Q7). S08/S08b/S08c are the
single-mapping tool-call drop (Q8). T7b checks three mixed parallel tool calls
render well-formed. T19 checks the OpenAI `name` field is inert and
byte-identical. S07e checks multiple leading steering messages render as ordered
frames. S09 checks non-string tool content.

## 2. Live agentic probe

`live_agentic_probe.py`. The offline suite renders Jinja; the probe runs the
template end-to-end through a live SGLang or vLLM OpenAI endpoint, so it
exercises tokenization, the tool-call and reasoning parsers, and the actual
model. It is model-agnostic.

Point it at your own endpoint. The `--endpoint` flag defaults to the `ENDPOINT`
environment variable, then to `http://localhost:8000`.

```bash
export ENDPOINT=http://localhost:8000

uv run --with openai python live_agentic_probe.py \
  --endpoint http://localhost:8000 --model qwen3.6-27b

uv run --with openai python live_agentic_probe.py \
  --endpoint http://localhost:8000 --model qwen3.6-35b-a3b --scenario degradation
```

Before trusting results, confirm the endpoint is serving the fork:

```bash
curl http://localhost:8000/get_server_info | grep chat_template
```

It should point at the baked `custom_pub_chat_template_qwen36.jinja`, not
`None`.

The probe has two scenarios.

**singleshot** — message shapes the fork handles and upstream rejects or
garbles. Each shape passes if the request returns 200 with usable output and no
template crash:

- `Q2_developer_index0` — a `developer` role at index 0. Upstream returns HTTP
  400 `Unexpected message role`.
- `Q7_developer_midconv` — a `developer` message mid-conversation. Upstream
  raises on any system/developer message after index 0.
- `Q1_multiturn_preserve_thinking` — a multi-turn exchange with thinking
  preserved across turns; checks the response stays coherent and doesn't get
  truncated inside `<think>`.
- `Q3Q5_tool_roundtrip` — a single tool call out and back. Checks the envelope
  and the arguments wire format survive the round trip.

**degradation** — the controlled A/B for Q1. It runs the same multi-turn
tool-calling loop twice, once with `preserve_thinking=true` (the fork default)
and once with `false`, and watches whether the tool-call `arguments` collapse to
`{}` after a few turns. That is the exact pi#3325 failure mode: when prior
`<think>` blocks are dropped from history, the model loses its own record of how
it picked the arguments last time. This is a diagnostic, not a hard assertion.
The degradation is probabilistic, so the probe prints per-turn argument health
for both settings and a human reads whether the fix mattered on this stack. Use
`--turns` to set the loop length.

Note on this stack: the empty-arguments collapse Q1 targets did not reproduce on
SGLang v0.5.12 + `qwen3_coder` in either setting. The hard failure appears to be
GGUF/llama.cpp-specific. On SGLang the value of `preserve_thinking=true` is
reasoning continuity and prefix-cache reuse, not empty-args prevention. The
probe is how you check that for yourself on your own runtime.

## 3. A/B coding benchmark

`results/`. Single-user serving numbers for the fork vs the stock template on a
short coding-prompt profile. See `results/README.md` for the variant naming
(`fixed`, `upstream`, `fixedreal`, `-warm`) and how to read the JSON.

The point of the A/B is the prefill token count, which is identical across every
variant. On well-formed coding prompts the fork tokenizes to the exact same
prompt the stock template does, so it costs nothing on prefill. The fork's extra
IMPORTANT bullets (Q6) only appear when tools are sent, and they live in the
cacheable system prefix, so even then the cost is prefilled once and KV-cached
for the session, not paid per turn. The warm runs land on top of upstream on
TTFT and decode. The fork is performance-neutral at single user.

These are the author's own single-B200-run measurements. Reproduce them by
serving each template behind your own endpoint and running the bench tooling,
or sanity-check the live behavior with `live_agentic_probe.py`.

## What each patch addresses

The symptoms below are the documented findings each patch targets. The
conformance test IDs in parentheses pin them.

- **Q1 — multi-turn tool argument collapse** (T5, S01, S02). In a multi-turn
  session calling the same tool repeatedly, after 2-3 turns the model emits
  `arguments: {}` even though its own reasoning named the arguments correctly.
  Cause: with `preserve_thinking` undefined, prior-turn `<think>` blocks are
  dropped from history. Fix: default `preserve_thinking=true`. Documented on
  GGUF/llama.cpp ([pi#3325](https://github.com/earendil-works/pi/issues/3325));
  did not reproduce on SGLang. Recover with `preserve_thinking=false`.

- **Q2 — developer role rejected** (T12, S03). opencode, Claude Code, openclaw,
  and Continue send a `developer` role (OpenAI Responses API convention).
  Upstream raises `Unexpected message role` and the request fails before any
  tokens generate. Fix: accept `developer` as an alias for `system`.
  ([sudoingX gist](https://gist.github.com/sudoingX/c2facf7d8f7608c65c1024ef3b22d431))

- **Q3 — string-typed tool arguments** (T13, S05). When a harness hands
  `tool_calls[].function.arguments` back as a JSON-encoded string (the OpenAI
  wire format, and what the Vercel AI SDK does in some flows), upstream's
  `arguments | items` throws the cryptic `Can only get item pairs from a
  mapping`. Fix: type-check first and raise a message that names the bug and
  says to deserialize once on ingest.
  ([pi#3325](https://github.com/earendil-works/pi/issues/3325))

- **Q4 — `</think>` variants and unclosed-think rescue** (T14, T15, T18). When
  the model emits `</thinking>` (long form) or a tool call inside an unclosed
  `<think>`, upstream treats the whole content as non-reasoning text and the
  literal tags leak into history. Fix: handle `</think>`, `</thinking>`,
  `</ think>`, `</think >`, and rescue the unclosed-think-with-tool_call case.
  Plain `</think>` stays byte-identical to upstream, even for pathological
  bodies (T18). ([ollama#14493](https://github.com/ollama/ollama/issues/14493),
  [Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475))

- **Q5 — OpenAI tool envelope wastes tokens** (T16). Harnesses send tool defs
  wrapped in `{"type":"function","function":{...}}`; upstream passes the whole
  wrapper through `tool | tojson`, costing ~12 tokens per tool. Fix: unwrap to
  the inner function spec, matching Qwen's own
  [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/chat_template.jinja).
  Gated by `unwrap_tool_envelope`, default `true`.

- **Q6 — strengthened IMPORTANT block** (T17). Under load with many tools the
  Qwen3-Coder family emits malformed tool calls: missing the opening
  `<tool_call>` tag, leading whitespace, or nested instead of parallel calls.
  Fix: add three bullets (complete `<tool_call>` wrapping, no leading
  indentation, separate closed blocks per call). Gated by
  `verbose_tool_instructions`, default `true`. The bullet on the missing
  opening tag is citation-backed
  ([Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475); it also
  helps the content-channel XML leak in
  [goose#6883](https://github.com/block/goose/issues/6883)); the indentation and
  nesting bullets are defensive guidance.

- **Q7 — mid-conversation system/developer crash** (S07, S07b, S07c, S07d).
  Agents inject steering messages mid-session ("prefer minimal diffs", "review
  mode"). Both upstream and the original fork raised `System/developer message
  must be at the beginning` on any such message after index 0, killing the loop.
  Fix: render it inline as a valid `<|im_start|>system` frame. Gated by
  `strict_system_position`, default `false`; set `true` to restore the raise.

- **Q8 — single tool-call mapping silently dropped** (S08, S08b, S08c). Some
  OpenAI-compat adapters hand back a single `tool_call` object instead of a
  one-element list. Upstream's `is not mapping` guards then silently drop the
  call, leaving an empty assistant turn and desyncing the following `tool`
  message. For an agent loop that silent corruption is worse than a crash. Fix:
  normalize a lone mapping into a one-element list. Always on.

The byte-identity proof from the live 27B runs: prompt token counts were
identical between fork and upstream on coding and agentic prompts, not just in
the offline render. Q2 and Q7 (developer role) were verified to serve live where
upstream returns HTTP 400.
