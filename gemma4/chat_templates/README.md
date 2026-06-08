# Gemma 4 chat templates

Three Jinja chat templates live here. Two are verbatim upstream copies. One is
a patched fork for agentic coding harnesses.

## Files

- **`gemma4_upstream.jinja`** — verbatim copy of `google/gemma-4-31B-it`. The
  baseline. The conformance suite diffs the fork against this file.

- **`gemma4_e4b_upstream.jinja`** — verbatim upstream template for the E4B
  model. This is the default the router serves for E4B. Do not patch it.

- **`custom_pub_chat_template_gemma4.jinja`** — patched fork of the 31B
  template (pinned at `fcf2302760ae9c6e528a8dbba9dd636e56848237`). Five patches
  (P1–P5) fix edge cases that hurt multi-turn tool calling. Apache 2.0, same as
  upstream.

## Which template to use

- **E4B router**: `gemma4_e4b_upstream.jinja`. Stock behavior, no patches.
- **31B / 26B / 12B dense**: `custom_pub_chat_template_gemma4.jinja`. These
  models drive coding harnesses and hit the bug sites the fork fixes.

## Patch inventory (P1–P5)

The fork carries five patches against the 31B upstream template. Full comments
sit next to each patch site in the `.jinja` file.

- **P1 — JSON `null` instead of bare `"None"`.** `format_argument` ran Python
  `None` through `str()`, emitting the literal `None` into the DSL (e.g.
  `after:None` in a search call). Optional fields are everywhere in coding
  tools. The fork emits `null`. The `is none` branch must come first, before
  `is string` / `is mapping` / `is sequence`.

- **P2 — `enable_thinking` defaults to `True`.** Upstream defaults it to
  `False`, and most OpenAI-compatible adapters drop unknown request fields, so
  thinking ends up permanently off with no failure signal. Tool-call accuracy
  suffers. See https://github.com/anomalyco/opencode/issues/24264

- **P3 — string-typed `arguments` raises instead of corrupting.** When
  `tool_call.arguments` arrives as a JSON string (Vercel AI SDK and several
  OpenAI-compatible adapters serialize this way), upstream wraps it in extra
  braces and produces invalid DSL like `call:fn{{"city":"Tokyo"}}` — nested
  braces, JSON colons, quoted keys, none of it trained on. The fork raises so
  the bug surfaces at the adapter boundary instead of silently degrading. See
  https://github.com/earendil-works/pi/issues/3325

- **P4 — `preserve_thinking` kwarg, defaults to `True`.** Upstream drops
  prior-turn reasoning from history. For chat that's correct; for multi-step
  agentic tool calling it's harmful — after 2–3 turns tool calls collapse to
  `arguments: {}`. The fork keeps prior `<|channel>` reasoning visible. Same
  issue as the Qwen analogue: https://github.com/earendil-works/pi/issues/3325

- **P5 — symmetric turn-tag close for HF discussion #62.** Two back-to-back
  text-only assistant messages rendered with one open and two closes
  (`<|turn>model\npart 1<turn|>\npart 2<turn|>\n`) — malformed. The model reads
  it as a truncated, re-opened turn, which destabilizes long agentic histories.
  The fork forward-scans for the next non-tool message and suppresses the
  redundant close. Conformance test T13 locks this in.

## Passing the template to a server

Both servers take the file via `--chat-template`.

vLLM:

```
vllm serve google/gemma-4-31B-it \
  --chat-template gemma4/chat_templates/custom_pub_chat_template_gemma4.jinja
```

SGLang:

```
python -m sglang.launch_server --model-path google/gemma-4-31B-it \
  --chat-template gemma4/chat_templates/custom_pub_chat_template_gemma4.jinja
```

For E4B, point at `gemma4_e4b_upstream.jinja` instead.

## Gating patches off per request

The two new behaviors are kwargs. Pass them through `chat_template_kwargs`:

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false,
      "preserve_thinking": false
    }
  }
}
```

- `enable_thinking: false` turns off P2.
- `preserve_thinking: false` turns off P4 and recovers upstream history
  handling exactly.

For opencode-style providers this maps to the `chat_template_args` field in the
models config. For pi, set `thinkingFormat` in the provider compat block and pi
injects these kwargs.

## Byte-identity invariant

With **both** `enable_thinking=False` and `preserve_thinking=False` passed
explicitly, the fork renders byte-for-byte identical to upstream on every input
that doesn't hit a P1, P3, or P5 bug site. That's the prefix-cache contract: any
drift means a patch leaked into a path it shouldn't touch.

P1, P3, and P5 are strict-where-upstream-is-silent. They cannot match upstream
because upstream was producing corrupt or malformed output there — that's the
point of the patch.

## Running the conformance suite

The suite at `gemma4/tests/test_custom_chat_template.py`
covers 20 cases: a byte-identity group (kwargs off, asserts the fork equals
upstream) and a strict-contract group (P1 `null`, P3 raise, P5 merge).

```
python3 -m pytest tests/test_custom_chat_template.py -v
```

Run from the `gemma4/` directory. Needs `jinja2` and `pytest`.

## Testing

Full testing notes are in [TESTING.md](TESTING.md): the symptom, root cause, and
fix for each patch; the offline conformance suite and its byte-identity
invariant; the live A/B procedure; and the maintenance contract for upstream
revision bumps.

- **Offline conformance suite** — 20 Jinja tests at
  `gemma4/tests/test_custom_chat_template.py`. No model, no GPU. Run with `uv`:

  ```
  uv run --with jinja2 --with pytest pytest tests/test_custom_chat_template.py -v
  ```

- **Live probe** — [live_agentic_probe.py](live_agentic_probe.py) drives the
  fork through any OpenAI-compatible endpoint and prints PASS/FAIL per scenario
  (thinking-on default, preserve_thinking across a tool loop, optional/null
  argument, two-tool parallel call). `--endpoint` defaults to `$ENDPOINT` or
  `http://localhost:8000`:

  ```
  uv run --with openai python live_agentic_probe.py --endpoint http://localhost:8000
  ```

The test count is reproducible. The probe's behavioral results are not a
published benchmark; this fork was never deployed live. Run the A/B in TESTING.md
against your own endpoints to get your own numbers.
