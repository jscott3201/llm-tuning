# Testing the Gemma 4 chat-template fork

This covers how the fork is tested: what the five patches fix, the offline
conformance suite you can run today, and the live A/B procedure you run
yourself against your own endpoints.

Two kinds of result live here. Keep them separate.

- **Reproducible right now.** The conformance suite is 20 offline Jinja tests.
  Clone, run, get 20 passing. No GPU, no network, no model.
- **You run your own.** The A/B is a procedure, not a result. This fork was
  never deployed live, so there are no benchmark numbers to quote. The
  procedure below tells you how to get your own.

If you see a tool-call accuracy number or a win rate anywhere in this repo that
is not the test count, it is wrong. There are none.

## What the patches fix

Five patches, P1 through P5, sit on top of the upstream template pinned at
`fcf2302760ae9c6e528a8dbba9dd636e56848237`. Each one targets a spot where the
upstream template is silent or wrong on inputs an agentic harness actually
sends. The full comment for each patch sits next to its patch site in
`custom_pub_chat_template_gemma4.jinja`.

### P1 — `None` renders as JSON `null`, not the string `"None"`

**Symptom.** A tool call with an optional argument set to null renders the DSL
token `after:None` instead of `after:null`. The model sees a bare `None` token
it was never trained on.

**Root cause.** The upstream `format_argument` macro has no `is none` branch.
Python `None` falls through to the final else branch, which stringifies it:
`str(None)` is `"None"`. Optional fields are everywhere in coding tools
(`language=null`, `after=null`, `pattern=null`).

**Fix.** Add an `is none` branch first, before `is string` / `is mapping` /
`is sequence`. Emit `null`. Ordering matters: `None` matches none of the other
type tests, so without the early branch it always reaches the stringifying
else.

### P2 — `enable_thinking` defaults to `True`

**Symptom.** Thinking ends up permanently off. Tool-call accuracy suffers and
there is no failure signal.

**Root cause.** Upstream defaults `enable_thinking` to false. SGLang has no
server-side default for chat-template kwargs
([sgl-project/sglang#5635](https://github.com/sgl-project/sglang/issues/5635)),
so the only way to turn thinking on is a per-request override. Many
OpenAI-compatible adapters drop unknown request fields, so that override never
arrives ([anomalyco/opencode#24264](https://github.com/anomalyco/opencode/issues/24264)).
Google's own model card says thinking enhances function-calling accuracy, and
tool calling is the core contract a coding harness uses the model for.

**Fix.** Flip the server-side default to true. Callers that want chat-only
behavior pass `chat_template_kwargs.enable_thinking=false` per request.

### P3 — string `arguments` raises instead of corrupting the prompt

**Symptom.** A tool call renders as `call:fn{{"city":"Tokyo"}}` — nested
braces, JSON colons, quoted keys. None of that is valid Gemma 4 DSL. The model
usually still answers, which hides the bug behind what looks like a model
quality problem.

**Root cause.** Some adapters (the Vercel AI SDK used by opencode is the common
one) hand `tool_calls[].function.arguments` back as a JSON-encoded string
rather than the deserialized object. Upstream's `is string` fallback emits that
string verbatim inside its own braces. Upstream does not raise.

**Fix.** Replace the silent string fallback with `raise_exception(...)`. A
contract violation now fails loudly at the render step. `arguments=None` stays
allowed (it renders an empty `{}`); every other non-mapping form raises. The
harness should still deserialize on ingest and check `is_object()` at its own
boundary; the template guard is a backstop, not a substitute. See
[earendil-works/pi#3325](https://github.com/earendil-works/pi/issues/3325).

### P4 — `preserve_thinking` keeps prior reasoning, defaults to `True`

**Symptom.** After two or three turns of a tool loop, tool calls collapse to
`arguments: {}` even though the model's prior reasoning had correctly worked
out the parameters.

**Root cause.** Upstream re-emits a prior turn's `<|channel>thought` block only
when the assistant message carries reasoning, has `tool_calls`, *and* sits
after the last user message. That third clause drops earlier-turn reasoning, so
the model loses the chain it would have imitated. Google's model card says
"historical model output should only include the final response," which is
right for plain chat and harmful for multi-step tool calling.

**Fix.** Add a `preserve_thinking` kwarg, default true. When true, drop the
"after last user message" clause so prior `<|channel>` blocks survive across
turns. The `tool_calls` clause stays — re-emitting a channel on a finalized
text-only turn is out of distribution. Set `preserve_thinking=false` to recover
upstream behavior byte-for-byte. Same fix as the Qwen analogue:
[earendil-works/pi#3325](https://github.com/earendil-works/pi/issues/3325).

### P5 — symmetric turn-tag close for two consecutive assistant messages

**Symptom.** Two back-to-back text-only assistant messages render as
`<|turn>model\npart 1<turn|>\npart 2<turn|>\n` — one open, two closes. The
model reads it as a truncated and re-opened turn, which destabilizes long
agentic histories that accumulate consecutive assistant messages.

**Root cause.** Upstream suppresses the *open* of a continuation (it sees the
previous non-tool message was also assistant) but emits the *close*
unconditionally. Open and close use different state, so they disagree. Google
staff confirmed this in
[HF discussion #62](https://huggingface.co/google/gemma-4-31B-it/discussions/62).

**Fix.** Forward-scan for the next non-tool message. If it is another assistant
and the current message is text-only (no `tool_calls`, no `tool_responses`),
suppress this close and emit a single `\n` so the two contents do not glue
together. The tool-call + tool-response chain is excluded and closes normally,
so the model still sees a balanced frame around the `<|tool_response>` block.

## Offline conformance suite

The suite is 20 tests at `gemma4/tests/test_custom_chat_template.py`. It renders
the Jinja directly with `jinja2`. No model, no server, no GPU. This is the only
number in this repo you can reproduce by running it: 20 tests, all passing.

It enforces two contracts.

**Byte-identity when the new behavior is disabled.** With both
`enable_thinking=False` and `preserve_thinking=False` passed explicitly, the
fork must render byte-for-byte identical to upstream on every input upstream
handles correctly. This is the prefix-cache invariant. Any drift means a patch
leaked into a path it should not touch. Tests T0, T2, T6, T7, T8, T9b, T10, T15,
T16, T17, T18, T19, and T4 cover this group, across bare prompts, system
prompts, tool declarations, full tool round-trips, current-turn reasoning,
booleans and integers, nested objects and arrays, multimodal content parts,
multi-tool fanout, and `arguments=None`.

**Strict where upstream is silent.** P1, P3, and P5 cannot match upstream,
because upstream produces corrupt or malformed output at those sites. That is
the point of the patch.

- T3 — string `arguments` raises a template error.
- T5 — `None` in arguments renders as `null` in the fork and `None` in upstream.
- T9 — `preserve_thinking=true` re-emits a prior turn's reasoning block.
- T11 — two consecutive text-only assistants merge into one balanced turn.
- T13 — a tool chain followed by two text assistants closes the tool chain
  normally and merges only the trailing text pair.
- T14 — default kwargs (no overrides) fire both new behaviors.
- T1 — default kwargs diverge from upstream (the companion to T0's byte-identity).

### The byte-identity-when-disabled invariant

This is the load-bearing invariant for cache economics. The fork shares the
upstream model's pinned revision, so the prompt prefix is stable across turns
only if the rendered bytes match. If the fork drifts from upstream on a path
upstream got right, the prefix cache misses and cost per turn climbs. The
byte-identity group is the regression test for that. Run it on every template
change.

### Running it

The suite needs `jinja2` and `pytest`. `uv` pulls both for the run without
touching your environment:

```
uv run --with jinja2 --with pytest pytest tests/test_custom_chat_template.py -v
```

Run from the `gemma4/` directory. Expected output ends with `20 passed`.

## Live A/B procedure

The patches change defaults and fix bug sites. The offline suite proves the
render is correct. It does not prove the model behaves better, because that
depends on weights, sampling, and the parser. To measure behavior you run an
A/B yourself. This fork was never deployed, so there are no numbers here to
copy. The steps below are what you run to get your own.

The shape is two endpoints behind a coin-flip router. Server A serves the
upstream template. Server B serves the fork. Everything else is identical.

1. **Stand up two SGLang servers, identical except the template.** Same model
   weights, same pinned revision, same CLI flags (`--reasoning-parser gemma4`,
   `--tool-call-parser gemma4`, `--attention-backend triton`, same
   `--kv-cache-dtype`, same sampling). The only difference:

   - Server A: `--chat-template <path>/gemma4_upstream.jinja`
   - Server B: `--chat-template <path>/custom_pub_chat_template_gemma4.jinja`

   Confirm each server actually loaded the template you intended before you
   trust anything. Hit `/get_server_info` and check the `chat_template` field
   points at the file you passed, not `None`.

2. **Put a coin-flip router in front of both.** Each request goes to A or B at
   random. Tag every downstream log line and metric with the variant so you can
   split results later. Send the same traffic distribution to both.

3. **Run the same agentic workload against the router.** The probe in this
   directory, `live_agentic_probe.py`, drives a representative set: a
   thinking-on default, a 3-turn tool loop that watches whether arguments
   collapse, a tool call with an optional null argument (the P1 bug site), and
   a two-tool parallel call. Point it at each variant in turn:

   ```
   uv run --with openai python live_agentic_probe.py --endpoint $ENDPOINT
   ```

   `--endpoint` defaults to `http://localhost:8000`. Set `ENDPOINT` or pass the
   flag to point at server A, then server B.

4. **Compare two things between variants.**

   - **Tool-call success.** Count responses where `arguments` parses as JSON,
     is an object, and carries the required keys with non-empty values. P1, P3,
     and P4 should move this. The 3-turn loop is where P4 shows up: watch
     whether arguments stay healthy past turn 2 or collapse to `{}`.
   - **Reasoning preservation.** With `preserve_thinking=true`, prior-turn
     reasoning should survive into the next prompt. Measure how much of the
     emitted `<|channel>thought` text reappears in the next turn's context. It
     should be near-identical. With `preserve_thinking=false` it should be
     dropped, matching upstream.

Treat the multi-turn argument-collapse check as a diagnostic, not a hard pass.
The degradation is probabilistic. The point is to see whether the fix moves the
turn at which arguments first go bad, A versus B, on your stack.

## Maintenance contract

The fork is pinned to one upstream revision. When that revision bumps
(`HF_REVISION` changes), the fork is stale until you re-check it. Do this:

1. Re-fetch the upstream `chat_template.jinja` at the new revision into
   `gemma4_upstream.jinja`.
2. Diff against the prior pinned revision. Look for upstream fixes that overlap
   a patch. If Google merges the discussion #62 fix, P5 is redundant and should
   be removed. If upstream adds a `null` branch to `format_argument`, P1 is
   redundant.
3. Re-apply the patches on the new base. Each patch site is marked with a
   `P# (public fork):` comment. Keep the boundaries clear.
4. Re-run the conformance suite. It must be 20/20 green. A byte-identity test
   failing means an upstream change moved a path the patch sits next to;
   reconcile before shipping.
5. If you run live serving, re-run the A/B against the new revision. Defaults
   and bug behavior can shift between model revisions even when the template
   text does not.

Bump the pin recorded in the README when the revision changes.
