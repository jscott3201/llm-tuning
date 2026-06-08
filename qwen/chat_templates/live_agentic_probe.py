"""Live agentic-shape probe for the Qwen3.6 chat template (upstream vs fork).

This complements the OFFLINE conformance suite (test_custom_pub_chat_template_qwen36.py,
which renders Jinja directly) by validating the template END-TO-END through a
running SGLang/vLLM OpenAI endpoint: tokenization + parsers + the actual model.

Two groups of checks:

  singleshot  — message shapes the public fork (Q1/Q2/Q3/Q5/Q7) handles and
                that upstream rejects/garbles. Each shape "passes" if the
                request returns 200 with usable output (no template crash).

  degradation — the controlled A/B for Q1 (preserve_thinking). Runs the SAME
                multi-turn tool-calling loop twice — once with
                preserve_thinking=True (the fork default), once False — and
                watches whether tool-call `arguments` collapse to `{}` after a
                few turns. This is the exact failure mode pi#3325 documents
                ("multi-turn tool calls degrade to empty {} arguments" when
                prior <think> blocks are dropped). It is a DIAGNOSTIC, not a
                hard assertion: the degradation is probabilistic, so the test
                surfaces the per-turn argument health for both settings so a
                human can see whether the fix matters on this stack.

Requires a live endpoint and the `openai` package. Model-agnostic — point it at
either deployment:

    uv run --with openai python live_agentic_probe.py \
        --endpoint http://your-endpoint:8000 --model qwen3.6-27b
    uv run --with openai python live_agentic_probe.py \
        --endpoint http://your-endpoint:8000 --model qwen3.6-35b-a3b --scenario degradation

The endpoint defaults to the ENDPOINT env var, or http://localhost:8000.

Tip: verify the endpoint is actually serving the FORK before trusting results:
    curl http://localhost:8000/get_server_info | grep chat_template
(should point at the baked custom_pub_chat_template_qwen36.jinja, not None).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI, APIError, BadRequestError


# ─────────────────────────────────────────────────────────────────────
# Tooling for the multi-turn coding-agent loop
# ─────────────────────────────────────────────────────────────────────

CODING_TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files in a directory.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Directory path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a source file's contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "apply_patch",
        "description": "Replace a snippet in a file with a fix.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "find": {"type": "string", "description": "Exact text to replace"},
                                      "replace": {"type": "string", "description": "Replacement text"}},
                       "required": ["path", "find", "replace"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the test suite for a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]

# Required keys per tool — used to judge whether emitted arguments are "healthy"
# (i.e. not collapsed to {} or missing everything).
_REQUIRED = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "apply_patch": ["path", "find", "replace"],
    "run_tests": ["path"],
}

_BUGGY_SRC = '''def divide(a, b):
    return a / b          # crashes on b == 0
'''


def synthetic_tool_result(name: str, args: dict) -> str:
    """Plausible canned results so the loop can keep going deterministically."""
    if name == "list_dir":
        return "calc.py\ntest_calc.py\nREADME.md"
    if name == "read_file":
        return _BUGGY_SRC
    if name == "apply_patch":
        return f"Patch applied to {args.get('path', 'calc.py')}. 1 hunk changed."
    if name == "run_tests":
        return "collected 3 items\n\ntest_calc.py ...   [100%]\n\n3 passed in 0.04s"
    return "ok"


def arg_health(name: str, raw_args) -> tuple[bool, str]:
    """A tool call's arguments are healthy iff they parse to a dict that
    carries at least one of the tool's required keys with a non-empty value.
    The pi#3325 degradation manifests as arguments == {} (or missing keys)."""
    if raw_args is None:
        return False, "arguments=None"
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if s in ("", "{}"):
            return False, f"empty args ({s or 'blank'})"
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False, f"unparseable: {s[:60]!r}"
    elif isinstance(raw_args, dict):
        obj = raw_args
    else:
        return False, f"unexpected type {type(raw_args).__name__}"
    if not isinstance(obj, dict) or not obj:
        return False, "empty/non-dict object"
    req = _REQUIRED.get(name, [])
    present = [k for k in req if str(obj.get(k, "")).strip()]
    if req and not present:
        return False, f"missing all required keys {req}; got {list(obj)}"
    return True, f"ok ({', '.join(present) or list(obj)})"


# ─────────────────────────────────────────────────────────────────────
# Single-shot agentic shapes (Q1/Q2/Q3/Q5/Q7)
# ─────────────────────────────────────────────────────────────────────

def _call(client, model, messages, *, tools=None, max_tokens=1500, enable_thinking=True,
          preserve_thinking=True):
    kw = dict(model=model, messages=messages, max_tokens=max_tokens,
              temperature=0.6, top_p=0.95,
              extra_body={"top_k": 20, "min_p": 0.0,
                          "chat_template_kwargs": {"enable_thinking": enable_thinking,
                                                   "preserve_thinking": preserve_thinking}})
    if tools:
        kw["tools"] = tools
    return client.chat.completions.create(**kw)


def run_singleshot(client, model, label):
    results = []

    def rec(name, ok, detail):
        results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"\n=== single-shot agentic shapes against {label} ===")

    # Q2 — developer role at index 0 (upstream: 400)
    try:
        r = _call(client, model, [
            {"role": "developer", "content": "You are terse. Answer in one word."},
            {"role": "user", "content": "What is the capital of France?"}])
        c = (r.choices[0].message.content or "").strip()
        rec("Q2_developer_index0", r.choices[0].finish_reason != "length" and bool(c),
            f"finish={r.choices[0].finish_reason} content={c[:80]!r}")
    except (BadRequestError, APIError) as e:
        rec("Q2_developer_index0", False, f"{type(e).__name__}: {str(e)[:160]}")

    # Q7 — developer role mid-conversation (upstream: 400). Thinking off so a
    # short answer doesn't get truncated inside <think>.
    try:
        r = _call(client, model, [
            {"role": "user", "content": "Help me write Python."},
            {"role": "assistant", "content": "Sure, what do you need?"},
            {"role": "developer", "content": "From now on, always include a docstring."},
            {"role": "user", "content": "Write a one-line function add(a,b) returning their sum, with a docstring."}],
            max_tokens=400, enable_thinking=False)
        c = (r.choices[0].message.content or "").strip()
        rec("Q7_developer_midconv", r.choices[0].finish_reason == "stop" and bool(c),
            f"finish={r.choices[0].finish_reason} content={c[:80]!r}")
    except (BadRequestError, APIError) as e:
        rec("Q7_developer_midconv", False, f"{type(e).__name__}: {str(e)[:160]}")

    # Q1 — multi-turn + preserve_thinking (no crash, coherent)
    try:
        r = _call(client, model, [
            {"role": "user", "content": "Pick a number 1-10, reply only the number."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Multiply it by 3; reply only the result."}])
        c = (r.choices[0].message.content or "").strip()
        rec("Q1_multiturn_preserve_thinking", r.choices[0].finish_reason != "length" and bool(c),
            f"finish={r.choices[0].finish_reason} content={c[:80]!r}")
    except (BadRequestError, APIError) as e:
        rec("Q1_multiturn_preserve_thinking", False, f"{type(e).__name__}: {str(e)[:160]}")

    # Q3/Q5 — single tool round-trip (envelope + args wire format)
    try:
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        msgs = [{"role": "user", "content": "What's the weather in Paris? Use get_weather."}]
        r1 = _call(client, model, msgs, tools=tools)
        tcs = r1.choices[0].message.tool_calls or []
        if tcs:
            tc = tcs[0]
            msgs += [
                {"role": "assistant", "content": r1.choices[0].message.content or "",
                 "tool_calls": [{"id": tc.id or "c1", "type": "function",
                                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]},
                {"role": "tool", "tool_call_id": tc.id or "c1",
                 "content": json.dumps({"city": "Paris", "temp_c": 18, "sky": "clear"})}]
            r2 = _call(client, model, msgs, tools=tools)
            c2 = (r2.choices[0].message.content or "").strip()
            rec("Q3Q5_tool_roundtrip", r2.choices[0].finish_reason != "length" and bool(c2),
                f"call={tc.function.name}{tc.function.arguments} -> {c2[:80]!r}")
        else:
            rec("Q3Q5_tool_roundtrip", r1.choices[0].finish_reason != "length",
                "no tool_call emitted (no crash)")
    except (BadRequestError, APIError) as e:
        rec("Q3Q5_tool_roundtrip", False, f"{type(e).__name__}: {str(e)[:160]}")

    n_ok = sum(1 for _, ok in results if ok)
    print(f"  -> {n_ok}/{len(results)} single-shot shapes passed")
    return results


# ─────────────────────────────────────────────────────────────────────
# Multi-turn tool-degradation A/B  (the pi#3325 reproduction for Q1)
# ─────────────────────────────────────────────────────────────────────

SYSTEM = ("You are a meticulous coding agent. Investigate and fix bugs using the "
          "provided tools. Think step by step. Make exactly ONE tool call per turn, "
          "then wait for its result before the next step. When the fix is verified, "
          "reply with a short summary and no tool call.")
TASK = ("calc.py has a bug: divide(a, b) crashes when b == 0. Investigate with the "
        "tools, apply a fix, and run the tests to confirm. One tool call per turn.")


def _drive_loop(client, model, *, preserve_thinking: bool, turns: int):
    """Run the agent loop for up to `turns` steps; return per-turn arg health."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    per_turn = []
    for t in range(turns):
        try:
            r = _call(client, model, messages, tools=CODING_TOOLS, max_tokens=2000,
                      enable_thinking=True, preserve_thinking=preserve_thinking)
        except (BadRequestError, APIError) as e:
            per_turn.append({"turn": t, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            break
        msg = r.choices[0].message
        fr = r.choices[0].finish_reason
        reasoning = getattr(msg, "reasoning_content", None) or ""
        tcs = msg.tool_calls or []
        if not tcs:
            per_turn.append({"turn": t, "tools": [], "finish": fr,
                             "reasoning_toks": len(reasoning.split()),
                             "final": (msg.content or "").strip()[:80]})
            break
        healths = [(tc.function.name, *arg_health(tc.function.name, tc.function.arguments),
                    tc.function.arguments) for tc in tcs]
        per_turn.append({"turn": t, "finish": fr, "reasoning_toks": len(reasoning.split()),
                         "tools": [(h[0], h[1], h[2], (h[3] or "")[:70]) for h in healths]})
        # Reconstruct the assistant turn — carry reasoning_content so the ONLY
        # difference between the two runs is whether the template re-emits it.
        a = {"role": "assistant", "content": msg.content or ""}
        if reasoning:
            a["reasoning_content"] = reasoning
        a["tool_calls"] = [{"id": tc.id or f"c{t}_{i}", "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for i, tc in enumerate(tcs)]
        messages.append(a)
        for i, tc in enumerate(tcs):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": tc.id or f"c{t}_{i}",
                             "content": synthetic_tool_result(tc.function.name, args)})
    return per_turn


def run_degradation(client, model, label, turns):
    print(f"\n=== multi-turn tool-degradation A/B against {label} (up to {turns} turns) ===")
    print("    Watching whether tool-call arguments collapse to {} across turns")
    print("    (pi#3325). Same loop, preserve_thinking True vs False.\n")
    summary = {}
    for preserve in (True, False):
        tag = f"preserve_thinking={preserve}"
        print(f"  --- {tag} ---")
        per = _drive_loop(client, model, preserve_thinking=preserve, turns=turns)
        first_bad = None
        for step in per:
            if "error" in step:
                print(f"    turn {step['turn']}: ERROR {step['error']}")
                continue
            if not step["tools"]:
                print(f"    turn {step['turn']}: (final answer, no tool) finish={step['finish']} "
                      f"reasoning≈{step['reasoning_toks']}w :: {step.get('final','')!r}")
                continue
            for (name, ok, reason, args) in step["tools"]:
                flag = "ok " if ok else "BAD"
                print(f"    turn {step['turn']}: {flag} {name}({args}) [{reason}] "
                      f"reasoning≈{step['reasoning_toks']}w finish={step['finish']}")
                if not ok and first_bad is None:
                    first_bad = step["turn"]
        verdict = ("all tool calls healthy" if first_bad is None
                   else f"FIRST DEGRADED at turn {first_bad}")
        summary[tag] = verdict
        print(f"    => {verdict}\n")
    print("  SUMMARY (the fix matters if False degrades earlier/more than True):")
    for tag, v in summary.items():
        print(f"    {tag:28} {v}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:8000"),
                    help="e.g. http://your-endpoint:8000 (defaults to $ENDPOINT or http://localhost:8000)")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--label", default=None)
    ap.add_argument("--scenario", choices=("all", "singleshot", "degradation"), default="all")
    ap.add_argument("--turns", type=int, default=5, help="max turns for the degradation loop")
    args = ap.parse_args()
    label = args.label or args.endpoint
    client = OpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key="EMPTY", timeout=180)

    if args.scenario in ("all", "singleshot"):
        run_singleshot(client, args.model, label)
    if args.scenario in ("all", "degradation"):
        run_degradation(client, args.model, label, args.turns)


if __name__ == "__main__":
    main()
