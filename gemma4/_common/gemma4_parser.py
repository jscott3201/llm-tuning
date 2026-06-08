"""Parser for Gemma 4's native tool-call output format.

Gemma 4 emits tool calls inline in the assistant's content using a custom
token format that is NOT JSON:

    <|tool_call>call:fn_name{key:<|"|>value<|"|>, num:42}<tool_call|>

There are two reasons to parse it client-side instead of relying on the
serving runtime's ``--tool-call-parser gemma4``:

1. **Concurrency safety.** Some runtime parser builds hold shared mutable
   state across requests; under concurrent tool-call workloads this can
   leak ``<pad>`` tokens into the response. Disabling the server-side
   parser and parsing client-side dodges the bug entirely.
2. **Train/serve shape parity.** If you fine-tune on exactly these raw
   tokens, parsing them client-side at eval time keeps the serving and
   training surfaces identical — no parser discrepancy can mask a real
   model regression.

The token shape is documented in the Gemma 4 chat template; the
delimiters ``<|"|>`` for string values and ``<|tool_call>`` /
``<tool_call|>`` for the call boundary are model-specific. Don't try to use
a generic JSON parser — the inner format isn't JSON.
"""

from __future__ import annotations

import re
from typing import Any


# Outer call boundary. The trailing `|$` lets us recover partial calls
# at end-of-stream when generation stops mid-token without a closing tag.
TOOL_CALL_RE = re.compile(
    r"<\|tool_call>\s*call:\s*(\w+)\s*\{(.*?)\}\s*(?:<tool_call\|>|$)",
    re.DOTALL,
)

# Inner string-value delimiter. Gemma 4 uses `<|"|>...<|"|>` to wrap
# string values so the parser can distinguish them from numbers, bools,
# and nested objects.
QUOTED_STR_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)


def parse_value(raw: str) -> Any:
    """Parse a single Gemma 4 tool-call value into a Python object.

    Strings, booleans, null, integers, floats, lists, and nested dicts
    are all supported. Anything that doesn't match a recognised shape
    falls through as a raw string — the surrounding scoring code is
    permissive about that on purpose, so a malformed call from the
    model still scores against expectations rather than aborting the
    whole eval run.
    """
    raw = raw.strip()
    if raw.startswith('<|"|>'):
        m = QUOTED_STR_RE.match(raw)
        if m:
            return m.group(1)
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        return parse_array(raw[1:-1])
    if raw.startswith("{") and raw.endswith("}"):
        return parse_args(raw[1:-1])
    return raw


def parse_array(raw: str) -> list:
    """Parse a Gemma 4 array body (the contents between `[` and `]`).

    Walks character by character so nested objects/arrays don't break
    the comma-split — `{a: 1, b: 2}` inside an array is one element,
    not two.
    """
    items = []
    depth = 0
    current = ""
    for c in raw:
        if c in ("{", "["):
            depth += 1
            current += c
        elif c in ("}", "]"):
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            if current.strip():
                items.append(parse_value(current.strip()))
            current = ""
        else:
            current += c
    if current.strip():
        items.append(parse_value(current.strip()))
    return items


def parse_args(raw: str) -> dict[str, Any]:
    """Parse a Gemma 4 tool-call argument body (between `{` and `}`).

    Handles three subtleties the format presents:

    1. String values are delimited by `<|"|>...<|"|>`, NOT JSON quotes,
       so we can't lean on a JSON parser. Detect the marker explicitly.
    2. Comma is the separator at depth 0 only — nested objects and
       arrays must not be split.
    3. Author errors (extra commas, missing values) shouldn't abort the
       whole eval. Be permissive: skip empty keys, take the last value
       for duplicate keys, recover at the next comma.
    """
    args = {}
    i = 0
    while i < len(raw):
        # Skip whitespace and stray commas between entries.
        while i < len(raw) and raw[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= len(raw):
            break

        key_start = i
        while i < len(raw) and raw[i] != ":":
            i += 1
        if i >= len(raw):
            break
        key = raw[key_start:i].strip()
        i += 1

        # Walk to the next depth-0 comma or close brace, treating
        # `<|"|>...<|"|>` as one atomic span.
        value_start = i
        depth = 0
        in_quote = False
        while i < len(raw):
            if raw[i: i + 5] == '<|"|>':
                in_quote = not in_quote
                i += 5
                continue
            if in_quote:
                i += 1
                continue
            if raw[i] in ("{", "["):
                depth += 1
            elif raw[i] in ("}", "]"):
                depth -= 1
                if depth < 0:
                    break
            elif raw[i] == "," and depth == 0:
                break
            i += 1

        value_str = raw[value_start:i].strip()
        if key:
            args[key] = parse_value(value_str)

    return args


def parse_model_output(
    raw_text: str,
) -> tuple[list[str], list[tuple[str, dict[str, Any]]], str]:
    """Split a Gemma 4 generation into (tool_names, tool_args, content).

    ``tool_args`` is returned as a list of ``(name, args)`` tuples so that
    multiple calls to the same tool are preserved. A dict keyed on tool
    name would silently drop the second call; tool-use evals frequently
    have agents that loop the same tool with different arguments, so
    we keep the order explicit.

    The content text has all tool-call, tool-response, thinking, and
    turn-marker tokens stripped — it's what the user-facing assistant
    message would contain.
    """
    tool_names: list[str] = []
    tool_args: list[tuple[str, dict[str, Any]]] = []

    for match in TOOL_CALL_RE.finditer(raw_text):
        fn_name = match.group(1)
        args_raw = match.group(2)
        tool_names.append(fn_name)
        try:
            args = parse_args(args_raw)
        except Exception:
            # Permissive parser failure — record an empty arg dict so
            # downstream scoring sees the tool was called but with no
            # parseable args (worse than wrong args, but not fatal).
            args = {}
        tool_args.append((fn_name, args))

    text = TOOL_CALL_RE.sub("", raw_text)
    text = re.sub(
        r"<\|tool_response>.*?<tool_response\|>", "", text, flags=re.DOTALL
    )
    text = re.sub(
        r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<turn\|>", "", text)
    text = re.sub(r"<\|turn>\w*\n?", "", text)
    text = text.strip()

    return tool_names, tool_args, text
