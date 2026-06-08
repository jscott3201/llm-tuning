# Qwen3.6 chat templates

A public, harness-friendly fork of the Qwen3.6 chat template (forked from the
Qwen3.6-27B template, served for both the 27B and 35B-A3B deployments), tuned for
open-source agentic coding harnesses ([opencode](https://github.com/anomalyco/opencode),
[pi](https://github.com/earendil-works/pi), openclaw, and similar Claude-Code-style
tools) pointed at a self-hosted SGLang / vLLM / llama.cpp endpoint.

## Files

- **`qwen36_upstream.jinja`** — Verbatim copy of `Qwen/Qwen3.6-27B/chat_template.jinja`.
  153 lines. MD5 `52b6d51ae5b203cb67e64b648494dad2`. The byte-identity reference
  for the conformance suite. Do not edit.
- **`custom_pub_chat_template_qwen36.jinja`** — The fork. Forked 2026-05-25. Apache 2.0.
  Carries the Q1-Q8 patches below. The file header documents every patch site inline.

## Why fork

Upstream is correct for chat. It bites agentic coding harnesses in a handful of
real edge cases: prior-turn thinking gets dropped (empty-argument loops), the
`developer` role crashes the request, string-typed tool arguments throw a cryptic
Jinja error, `</think>` variants leak reasoning into content, and the OpenAI tool
envelope wastes tokens. The patches fix those without changing behavior for plain
chat.

## Patch inventory (Q1-Q8)

| Patch | What it does | Fixes |
|-------|--------------|-------|
| **Q1** | `preserve_thinking` default flipped `false` → `true` | Multi-turn tool argument collapse. With `preserve_thinking=false`, prior-turn `<think>` blocks are dropped from history; after 2-3 calls of the same tool the model emits `arguments: {}`. The model card says Qwen3.6 was post-trained for thinking preservation in agent scenarios, so `true` is the recommended setting. [pi#3325](https://github.com/earendil-works/pi/issues/3325) |
| **Q2** | `developer` role accepted as an alias for `system` | opencode, Claude Code, openclaw, and Continue send a `developer` role (OpenAI Responses API convention). Upstream raises "Unexpected message role" and crashes the request. [sudoingX gist](https://gist.github.com/sudoingX/c2facf7d8f7608c65c1024ef3b22d431) |
| **Q3** | Clear, debuggable raise on string-typed `tool_call.arguments` | The Vercel AI SDK (used by opencode) and other OpenAI-compat adapters hand `arguments` back as a JSON-encoded string. Upstream's `arguments \| items` then throws "Can only get item pairs from a mapping" — impossible to debug from the message. Q3 type-checks first and tells you to deserialize once on ingest. [pi#3325](https://github.com/earendil-works/pi/issues/3325) |
| **Q4** | Robust `</think>` variant handling + unclosed-think rescue | Upstream only recognizes a properly closed `</think>`. Q4 also handles `</thinking>`, whitespace variants (`</ think>`, `</think >`), and `<tool_call>` emitted inside an unclosed `<think>` block. Otherwise reasoning bleeds into the content channel. [ollama#14493](https://github.com/ollama/ollama/issues/14493), [Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475) |
| **Q5** | Unwrap the OpenAI tool envelope to the inner function spec (gated) | Harnesses send tool defs wrapped in `{"type":"function","function":{...}}`. Upstream passes the whole wrapper through `tool \| tojson`, wasting ~12 tokens per tool. Qwen's own [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/chat_template.jinja) unwraps it (lines 35-37); this backports that. |
| **Q6** | Strengthened IMPORTANT instructions block (gated) | Adds three bullets: do not omit the opening `<tool_call>` tag ([Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475)), keep `<tool_call>`/`<function>` at the start of a line with no indentation (defensive), and do not nest `<tool_call>` blocks. Targets the content-channel XML-leak failure mode in [goose#6883](https://github.com/block/goose/issues/6883). |
| **Q7** | Mid-conversation `system`/`developer` rendered inline (gated) | Upstream hard-crashes on any `system`/`developer` message after index 0 — the same failure Q2 fixed, just at a later index. Agents inject steering messages mid-session. Q7 renders them as a valid `<\|im_start\|>system` frame instead of killing the loop. |
| **Q8** | Single `tool_call` mapping normalized to a one-item list | Some adapters hand back a single tool_call object instead of a list. Upstream's `is not mapping` guards then silently drop the call, desyncing the following `tool` message. Q8 wraps a lone mapping into a one-element list. |

## Gated kwargs and recovering upstream behavior

Three patches are toggle-gated and default to the agentic-friendly setting:

| Kwarg | Default | Recover upstream |
|-------|---------|------------------|
| `preserve_thinking` (Q1) | `true` | `false` |
| `unwrap_tool_envelope` (Q5) | `true` | `false` |
| `verbose_tool_instructions` (Q6) | `true` | `false` |

A fourth gate, `strict_system_position` (Q7), defaults `false` so mid-conversation
system/developer messages render inline. Set it `true` to restore upstream's hard
raise. Q2, Q3, Q4, and Q8 are not gated — they only fire on inputs that hit a
documented bug surface, so they never change well-formed output.

Pass kwargs through the request:

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": true,
      "preserve_thinking": false,
      "unwrap_tool_envelope": false,
      "verbose_tool_instructions": false
    }
  }
}
```

## Developer role and string arguments

**Developer role (Q2).** Both `system` and `developer` are valid roles. At index 0
the content folds into the system block. After index 0 it renders as its own system
frame (Q7). No harness change needed.

**String arguments (Q3).** `tool_call.arguments` must be a JSON object (a mapping).
If your adapter hands it back as a JSON-encoded string, the template raises a clear
error naming the bug and pointing at [pi#3325](https://github.com/earendil-works/pi/issues/3325).
Fix it on the harness side: deserialize the arguments string exactly once on ingest
and store the resulting dict. Do not pre-stringify.

## Serving

SGLang:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-27B \
  --chat-template /path/to/custom_pub_chat_template_qwen36.jinja \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

vLLM:

```bash
vllm serve Qwen/Qwen3.6-27B \
  --chat-template /path/to/custom_pub_chat_template_qwen36.jinja \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

`--tool-call-parser qwen3_coder` parses the `<tool_call>`/`<function>` XML the
template emits. `--reasoning-parser qwen3` splits the `<think>` channel.

## opencode / pi integration

No changes for the common case. Defaults are tuned for agentic coding out of the box.

- **opencode**: the kwargs map to `chat_template_args` in the model config. Use it
  only if you need to recover upstream defaults.
- **pi**: set `compat.thinkingFormat="qwen-chat-template"` and pi injects the kwargs
  correctly.

Both harnesses must deserialize tool-call arguments once on ingest (see Q3).

## Byte-identity invariant

With `preserve_thinking=false`, `unwrap_tool_envelope=false`, and
`verbose_tool_instructions=false`, and on inputs that don't exercise Q2, Q3, Q4, Q7,
or Q8, the fork renders **byte-for-byte identical** to upstream. This protects the
prefix cache. Any drift means a patch leaked into a path it shouldn't have.

Q4 has a tighter contract: for plain `</think>` inputs — including pathological
bodies (multiple `</think>`, embedded literal `<think>`, nested openers) — Q4
reproduces upstream's exact last-opener / first-closer / last-closer split. Only
`</thinking>`, whitespace variants, and unclosed `<think>` diverge, and only in the
strictly safer direction.

## Conformance suite

39 collected tests at `../tests/test_custom_pub_chat_template_qwen36.py`. They render
both templates with Jinja2 and assert three contracts: byte-identity to upstream,
strict-where-upstream-silent behavior, and the documented agentic bug shapes (Sxx).

```bash
python3 -m pytest tests/test_custom_pub_chat_template_qwen36.py -v
```

Run from the `qwen/` directory. Requires `jinja2` and `pytest`.

## Testing

Full validation notes are in [`TESTING.md`](TESTING.md). Three layers:

- **Offline conformance suite** — 39 tests at
  [`../tests/test_custom_pub_chat_template_qwen36.py`](../tests/test_custom_pub_chat_template_qwen36.py).
  Renders both templates with Jinja2 and asserts byte-identity, strict-safety,
  and the documented agentic bug shapes. Run with
  `uv run --with jinja2 --with pytest python -m pytest tests/test_custom_pub_chat_template_qwen36.py -v`.
- **Live agentic probe** — [`live_agentic_probe.py`](live_agentic_probe.py).
  Drives a live SGLang/vLLM endpoint to check the message shapes end-to-end and
  run the Q1 multi-turn degradation A/B. Point it at your own endpoint
  (`--endpoint`, defaults to `$ENDPOINT` or `http://localhost:8000`).
- **A/B coding benchmark** — [`results/`](results/). Single-user serving numbers
  for the fork vs the stock template, showing the fork is performance-neutral.
