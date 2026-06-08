"""Live agentic-shape probe for the Gemma 4 chat-template fork.

This complements the OFFLINE conformance suite
(tests/test_custom_chat_template.py, which renders the Jinja directly) by
driving the template END-TO-END through a running SGLang or vLLM
OpenAI-compatible endpoint: tokenization, the gemma4 parsers, and the actual
model. The offline suite proves the render is correct. This probe shows how the
model behaves on the shapes the fork targets.

Model-agnostic. Point it at any OpenAI-compatible endpoint serving a Gemma 4
model with the fork loaded:

    uv run --with openai python live_agentic_probe.py
    uv run --with openai python live_agentic_probe.py --endpoint http://localhost:8000
    ENDPOINT=http://localhost:8000 uv run --with openai python live_agentic_probe.py

--endpoint defaults to os.environ["ENDPOINT"] or http://localhost:8000.

Scenarios, each prints PASS or FAIL:

  thinking_default   enable_thinking defaults to ON in the fork (P2). A plain
                     request with no chat_template_kwargs should come back with
                     reasoning_content populated.

  preserve_thinking  preserve_thinking keeps prior reasoning across a 3-turn
                     tool-call loop (P4). The failure mode it guards against is
                     tool-call arguments collapsing to {} after a couple of
                     turns. This is a DIAGNOSTIC: the collapse is probabilistic,
                     so the scenario surfaces per-turn argument health and only
                     fails hard if every tool call after turn 0 is empty.

  null_argument      an optional/null argument round-trips without leaking the
                     bare string "None" (P1). A tool with a nullable field is
                     offered; whatever the model emits, the args must parse as
                     JSON and carry no literal "None" token.

  parallel_tools     a basic two-tool parallel call lands in tool_calls, not in
                     content.

Before trusting results, confirm the endpoint actually loaded the fork:
    curl http://localhost:8000/get_server_info | grep chat_template
(should point at custom_pub_chat_template_gemma4.jinja, not None).
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

# Required keys per tool — used to judge whether emitted arguments are healthy
# (i.e. not collapsed to {} or missing everything).
_REQUIRED = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "apply_patch": ["path", "find", "replace"],
    "run_tests": ["path"],
}

# A tool with an OPTIONAL / nullable field. Gemma 4's template has an explicit
# `nullable` handler; declare a single concrete type plus "nullable": true
# rather than a union. This is the P1 bug site: if the model sends the optional
# field as null, the upstream template would emit the bare token `None`.
SEARCH_TOOL = [
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search source files for a pattern, optionally filtered by language.",
        "parameters": {"type": "object",
                       "properties": {
                           "pattern": {"type": "string", "description": "Substring or regex to find"},
                           "language": {"type": "string", "nullable": True,
                                        "description": "Restrict to one language, or null for all"}},
                       "required": ["pattern"]}}},
]

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
    if name == "search_code":
        return "calc.py:2: return a / b"
    return "ok"


def parse_args(raw_args) -> tuple[bool, dict, str]:
    """Parse tool-call arguments into a dict. Returns (ok, obj, detail)."""
    if raw_args is None:
        return True, {}, "None -> {}"
    if isinstance(raw_args, dict):
        return True, raw_args, "already a dict"
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if s in ("", "{}"):
            return True, {}, "empty"
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False, {}, f"unparseable: {s[:60]!r}"
        if not isinstance(obj, dict):
            return False, {}, f"not an object: {type(obj).__name__}"
        return True, obj, "ok"
    return False, {}, f"unexpected type {type(raw_args).__name__}"


def arg_health(name: str, raw_args) -> tuple[bool, str]:
    """Arguments are healthy iff they parse to a dict carrying at least one of
    the tool's required keys with a non-empty value. The P4 degradation
    manifests as arguments == {} after a few turns."""
    ok, obj, detail = parse_args(raw_args)
    if not ok:
        return False, detail
    if not obj:
        return False, "empty/{}"
    req = _REQUIRED.get(name, [])
    present = [k for k in req if str(obj.get(k, "")).strip()]
    if req and not present:
        return False, f"missing all required keys {req}; got {list(obj)}"
    return True, f"ok ({', '.join(present) or list(obj)})"


# ─────────────────────────────────────────────────────────────────────
# Request helper
# ─────────────────────────────────────────────────────────────────────

def _call(client, model, messages, *, tools=None, max_tokens=4000,
          enable_thinking=None, preserve_thinking=None, omit_kwargs=False):
    """One chat completion. By default sends NO chat_template_kwargs so the
    fork's baked-in defaults (enable_thinking=true, preserve_thinking=true)
    apply — which is exactly what we want to probe for the default scenario.
    Pass explicit bools to override; omit_kwargs forces the no-kwargs path."""
    kw = dict(model=model, messages=messages, max_tokens=max_tokens,
              temperature=1.0, top_p=0.95,
              extra_body={"top_k": 64})
    if not omit_kwargs and (enable_thinking is not None or preserve_thinking is not None):
        ctk = {}
        if enable_thinking is not None:
            ctk["enable_thinking"] = enable_thinking
        if preserve_thinking is not None:
            ctk["preserve_thinking"] = preserve_thinking
        kw["extra_body"]["chat_template_kwargs"] = ctk
    if tools:
        kw["tools"] = tools
    return client.chat.completions.create(**kw)


def _reasoning(msg) -> str:
    return getattr(msg, "reasoning_content", None) or ""


# ─────────────────────────────────────────────────────────────────────
# Scenario (a): enable_thinking default is ON
# ─────────────────────────────────────────────────────────────────────

def scenario_thinking_default(client, model):
    """P2: with no chat_template_kwargs, the fork defaults enable_thinking to
    true, so the response should carry reasoning_content."""
    print("\n=== scenario thinking_default (P2: enable_thinking defaults ON) ===")
    try:
        r = _call(client, model,
                  [{"role": "user", "content": "What is 17 * 23? Think it through."}],
                  omit_kwargs=True, max_tokens=2000)
    except (BadRequestError, APIError) as e:
        print(f"  [FAIL] thinking_default: {type(e).__name__}: {str(e)[:160]}")
        return False
    msg = r.choices[0].message
    reasoning = _reasoning(msg)
    ok = bool(reasoning.strip())
    detail = (f"reasoning_content {'present' if ok else 'ABSENT'} "
              f"({len(reasoning.split())}w); content={(msg.content or '')[:60]!r}")
    if not ok:
        detail += "  (absent => server fell back to upstream template, default OFF)"
    print(f"  [{'PASS' if ok else 'FAIL'}] thinking_default: {detail}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Scenario (b): preserve_thinking across a 3-turn tool loop
# ─────────────────────────────────────────────────────────────────────

SYSTEM = ("You are a meticulous coding agent. Investigate and fix bugs using the "
          "provided tools. Think step by step. Make exactly ONE tool call per turn, "
          "then wait for its result before the next step. When the fix is verified, "
          "reply with a short summary and no tool call.")
TASK = ("calc.py has a bug: divide(a, b) crashes when b == 0. Investigate with the "
        "tools, apply a fix, and run the tests to confirm. One tool call per turn.")


def _drive_loop(client, model, *, preserve_thinking: bool, turns: int):
    """Run the agent loop for up to `turns` steps; return per-turn arg health.
    reasoning_content is carried back onto each assistant turn so the ONLY
    difference between settings is whether the template re-emits it."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    per_turn = []
    for t in range(turns):
        try:
            r = _call(client, model, messages, tools=CODING_TOOLS, max_tokens=4000,
                      enable_thinking=True, preserve_thinking=preserve_thinking)
        except (BadRequestError, APIError) as e:
            per_turn.append({"turn": t, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            break
        msg = r.choices[0].message
        fr = r.choices[0].finish_reason
        reasoning = _reasoning(msg)
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
        a = {"role": "assistant", "content": msg.content or ""}
        if reasoning:
            a["reasoning_content"] = reasoning
        a["tool_calls"] = [{"id": tc.id or f"c{t}_{i}", "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for i, tc in enumerate(tcs)]
        messages.append(a)
        for i, tc in enumerate(tcs):
            _, args, _ = parse_args(tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id or f"c{t}_{i}",
                             "content": synthetic_tool_result(tc.function.name, args)})
    return per_turn


def scenario_preserve_thinking(client, model, turns):
    """P4: across a multi-turn tool loop, arguments should not collapse to {}.
    Diagnostic — runs preserve_thinking True and False and prints both. Passes
    if the True run keeps at least one healthy tool call after turn 0."""
    print(f"\n=== scenario preserve_thinking (P4: 3-turn tool loop, up to {turns} turns) ===")
    print("    Watching whether tool-call arguments collapse to {} across turns.")
    results = {}
    for preserve in (True, False):
        tag = f"preserve_thinking={preserve}"
        print(f"  --- {tag} ---")
        per = _drive_loop(client, model, preserve_thinking=preserve, turns=turns)
        first_bad = None
        healthy_after_0 = 0
        for step in per:
            if "error" in step:
                print(f"    turn {step['turn']}: ERROR {step['error']}")
                continue
            if not step["tools"]:
                print(f"    turn {step['turn']}: (final, no tool) finish={step['finish']} "
                      f"reasoning~{step['reasoning_toks']}w :: {step.get('final','')!r}")
                continue
            for (name, ok, reason, args) in step["tools"]:
                flag = "ok " if ok else "BAD"
                print(f"    turn {step['turn']}: {flag} {name}({args}) [{reason}] "
                      f"reasoning~{step['reasoning_toks']}w finish={step['finish']}")
                if ok and step["turn"] > 0:
                    healthy_after_0 += 1
                if not ok and first_bad is None:
                    first_bad = step["turn"]
        verdict = ("all tool calls healthy" if first_bad is None
                   else f"first degraded at turn {first_bad}")
        results[preserve] = (verdict, healthy_after_0)
        print(f"    => {verdict}\n")
    print("  compare (the fix matters if False degrades earlier/more than True):")
    for preserve, (verdict, _) in results.items():
        print(f"    preserve_thinking={str(preserve):5} {verdict}")
    # Pass: the default (True) run keeps tool calls healthy past the first turn,
    # or finishes cleanly without ever degrading.
    true_verdict, true_healthy = results[True]
    ok = (true_verdict == "all tool calls healthy") or true_healthy > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] preserve_thinking: "
          f"default run {'kept arguments healthy' if ok else 'degraded immediately'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Scenario (c): null / optional argument does not leak "None"  (P1)
# ─────────────────────────────────────────────────────────────────────

def scenario_null_argument(client, model):
    """P1: a tool with a nullable optional field. Whatever the model emits, the
    args must parse as JSON and carry no literal "None" token. The round-trip
    (re-send the call, get a coherent follow-up) exercises the render path that
    would otherwise emit the bare `None` DSL token."""
    print("\n=== scenario null_argument (P1: optional/null arg, no 'None' leak) ===")
    msgs = [{"role": "user",
             "content": "Search the code for the pattern 'divide' across all languages "
                        "using search_code. Leave the language filter unset."}]
    try:
        r1 = _call(client, model, msgs, tools=SEARCH_TOOL, max_tokens=3000)
    except (BadRequestError, APIError) as e:
        print(f"  [FAIL] null_argument: {type(e).__name__}: {str(e)[:160]}")
        return False
    tcs = r1.choices[0].message.tool_calls or []
    if not tcs:
        print(f"  [FAIL] null_argument: no tool_call emitted "
              f"(finish={r1.choices[0].finish_reason})")
        return False
    tc = tcs[0]
    raw = tc.function.arguments
    ok_parse, obj, detail = parse_args(raw)
    if not ok_parse:
        print(f"  [FAIL] null_argument: arguments did not parse [{detail}] raw={raw!r}")
        return False
    # The bug we guard against: a literal "None" leaking into the rendered DSL.
    # On the response side it would show up as the string value "None".
    leaked = any(str(v).strip() == "None" for v in obj.values())
    # Round-trip: feed the call + a result back and confirm a coherent follow-up.
    msgs.append({"role": "assistant", "content": r1.choices[0].message.content or "",
                 "tool_calls": [{"id": tc.id or "c1", "type": "function",
                                 "function": {"name": tc.function.name, "arguments": raw}}]})
    _, args, _ = parse_args(raw)
    msgs.append({"role": "tool", "tool_call_id": tc.id or "c1",
                 "content": synthetic_tool_result(tc.function.name, args)})
    try:
        r2 = _call(client, model, msgs, tools=SEARCH_TOOL, max_tokens=3000)
        follow = (r2.choices[0].message.content or "").strip()
        round_trip_ok = r2.choices[0].finish_reason != "length"
    except (BadRequestError, APIError) as e:
        print(f"  [FAIL] null_argument: round-trip render error "
              f"(likely a 'None' leak) {type(e).__name__}: {str(e)[:160]}")
        return False
    ok = (not leaked) and round_trip_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] null_argument: "
          f"call={tc.function.name}({raw}) keys={list(obj)} "
          f"none_leak={leaked} round_trip={'ok' if round_trip_ok else 'truncated'} "
          f":: {follow[:60]!r}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Scenario (d): basic two-tool parallel call
# ─────────────────────────────────────────────────────────────────────

PARALLEL_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Current local time for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def scenario_parallel_tools(client, model):
    """Two distinct tools in one turn. The calls must land in tool_calls (not in
    content). Non-streaming, per the harness rule for >=2 expected calls."""
    print("\n=== scenario parallel_tools (two-tool parallel call) ===")
    msgs = [{"role": "user",
             "content": "For Tokyo, get BOTH the current weather and the current local "
                        "time. Call get_weather and get_time."}]
    try:
        r = _call(client, model, msgs, tools=PARALLEL_TOOLS, max_tokens=3000)
    except (BadRequestError, APIError) as e:
        print(f"  [FAIL] parallel_tools: {type(e).__name__}: {str(e)[:160]}")
        return False
    tcs = r.choices[0].message.tool_calls or []
    names = [tc.function.name for tc in tcs]
    # Healthy: at least two calls, each with a parsable city arg. We do not hard
    # require exactly two — some stacks split parallel calls across turns — but
    # we do require the calls land in tool_calls, not free text.
    healthy = []
    for tc in tcs:
        ok_h, reason = arg_health(tc.function.name, tc.function.arguments)
        healthy.append((tc.function.name, ok_h, reason, tc.function.arguments))
    n_ok = sum(1 for _, ok_h, _, _ in healthy if ok_h)
    ok = len(tcs) >= 2 and n_ok >= 2
    for (name, ok_h, reason, raw) in healthy:
        print(f"    {'ok ' if ok_h else 'BAD'} {name}({raw}) [{reason}]")
    if not tcs:
        print(f"    no tool_calls; content={(r.choices[0].message.content or '')[:80]!r}")
    print(f"  [{'PASS' if ok else 'FAIL'}] parallel_tools: "
          f"{len(tcs)} call(s) {names}, {n_ok} healthy")
    return ok


# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "thinking_default": lambda c, m, a: scenario_thinking_default(c, m),
    "preserve_thinking": lambda c, m, a: scenario_preserve_thinking(c, m, a.turns),
    "null_argument": lambda c, m, a: scenario_null_argument(c, m),
    "parallel_tools": lambda c, m, a: scenario_parallel_tools(c, m),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:8000"),
                    help="OpenAI-compatible base (default: $ENDPOINT or http://localhost:8000)")
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    ap.add_argument("--turns", type=int, default=5, help="max turns for the preserve_thinking loop")
    args = ap.parse_args()

    client = OpenAI(base_url=args.endpoint.rstrip("/") + "/v1",
                    api_key=args.api_key, timeout=180)
    print(f"endpoint: {args.endpoint}   model: {args.model}")

    to_run = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    results = {}
    for name, fn in to_run.items():
        results[name] = fn(client, args.model, args)

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_ok = sum(1 for ok in results.values() if ok)
    print(f"  -> {n_ok}/{len(results)} scenarios passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
