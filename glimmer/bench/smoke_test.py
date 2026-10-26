#!/usr/bin/env python3
"""Smoke test for a Muse Glimmer SGLang endpoint.

Checks what an agent harness depends on before you point one at it:

  1. plain chat: an answer in ``content``, reasoning split out into
     ``reasoning_content`` by the ``muse`` reasoning parser
  2. a tool call parsed into ``message.tool_calls`` by the ``muse`` parser
  3. a full tool round trip: the tool result goes back in and the model
     answers from it (this exercises SGLang's JSON-string → dict argument
     normalisation that the chat template requires)
  4. a streamed tool call that reassembles into valid JSON
  5. image input (skip with ``--no-image`` for a text-only deployment)
  6. OpenAI's ``reasoning_effort`` field is accepted

Run:

    uv run python bench/smoke_test.py --base-url https://<your-app>.modal.run/v1

``--base-url`` defaults to ``$GLIMMER_ENDPOINT`` (``/v1`` appended) or
``http://localhost:8000/v1``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import zlib

from openai import OpenAI


def _default_base_url() -> str:
    endpoint = os.environ.get("GLIMMER_ENDPOINT", "http://localhost:8000").rstrip("/")
    return endpoint if endpoint.endswith("/v1") else endpoint + "/v1"


# Model-card sampling. top_k is an SGLang extension, so it goes in extra_body.
SAMPLING = {"temperature": 1.0, "top_p": 0.95}
EXTRA = {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "low"}}
# Reasoning tokens count against max_tokens; leave room for them.
MAX_TOKENS = 4096

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
    },
}
WEATHER_Q = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]


def _png_two_halves(w: int = 224, h: int = 224) -> str:
    """Left half red, right half blue, as a base64 PNG (stdlib only)."""
    red, blue = b"\xdc\x1e\x1e", b"\x1e\x3c\xdc"
    row = b"\x00" + red * (w // 2) + blue * (w - w // 2)
    raw = row * h

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _short(text: str | None, n: int = 160) -> str:
    if not text:
        return repr(text)
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _chat(client: OpenAI, model: str, messages, **kw):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        extra_body=kw.pop("extra_body", EXTRA),
        **SAMPLING,
        **kw,
    )


def t_plain(client, model) -> str | None:
    r = _chat(client, model, [{"role": "user", "content": "What is 17 * 23? Answer with the number."}])
    m = r.choices[0].message
    print(f"  content   : {_short(m.content)}")
    print(f"  reasoning : {len(getattr(m, 'reasoning_content', None) or '')} chars")
    if not m.content or "391" not in m.content:
        return "expected 391 in content"
    if not getattr(m, "reasoning_content", None):
        return "no reasoning_content (is --reasoning-parser muse set?)"
    return None


def _first_call(client, model):
    r = _chat(client, model, WEATHER_Q, tools=[WEATHER_TOOL])
    return r.choices[0].message


def t_tool_call(client, model) -> str | None:
    m = _first_call(client, model)
    calls = m.tool_calls or []
    print(f"  tool_calls: {[(c.function.name, c.function.arguments) for c in calls]}")
    if not calls or calls[0].function.name != "get_weather":
        return "no get_weather call"
    if "city" not in json.loads(calls[0].function.arguments):
        return "arguments missing 'city'"
    return None


def t_round_trip(client, model) -> str | None:
    m = _first_call(client, model)
    if not m.tool_calls:
        return "no tool call to answer"
    call = m.tool_calls[0]
    history = WEATHER_Q + [
        {
            "role": "assistant",
            "content": m.content,
            # As the OpenAI SDK returns it: arguments is a JSON *string*.
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "light rain"}),
        },
    ]
    r = _chat(client, model, history, tools=[WEATHER_TOOL])
    content = r.choices[0].message.content
    print(f"  answer    : {_short(content)}")
    if not content or "17" not in content:
        return "final answer does not use the tool result (17°C)"
    return None


def t_stream_tool(client, model) -> str | None:
    stream = _chat(client, model, WEATHER_Q, tools=[WEATHER_TOOL], stream=True)
    name, args = None, ""
    for chunk in stream:
        if not chunk.choices:
            continue
        for tc in chunk.choices[0].delta.tool_calls or []:
            if tc.function and tc.function.name:
                name = tc.function.name
            if tc.function and tc.function.arguments:
                args += tc.function.arguments
    print(f"  streamed  : {name} {args}")
    if name != "get_weather":
        return "no streamed get_weather call"
    try:
        json.loads(args)
    except json.JSONDecodeError:
        return f"streamed arguments are not valid JSON: {args!r}"
    return None


def t_image(client, model) -> str | None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_two_halves()}},
                {"type": "text", "text": "This image has two colored halves. Name the color of the left half and the right half."},
            ],
        }
    ]
    r = _chat(client, model, msgs)
    content = (r.choices[0].message.content or "").lower()
    print(f"  content   : {_short(content)}")
    if "red" not in content or "blue" not in content:
        return "expected both 'red' and 'blue'"
    return None


def t_reasoning_effort(client, model) -> str | None:
    # No chat_template_kwargs here: the custom template maps reasoning_effort
    # onto Glimmer's reasoning strength.
    r = _chat(
        client, model, [{"role": "user", "content": "Say hello."}],
        reasoning_effort="low", extra_body={"top_k": 64},
    )
    content = r.choices[0].message.content
    print(f"  content   : {_short(content)}")
    return None if content else "empty content"


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test a Muse Glimmer endpoint.")
    ap.add_argument("--base-url", default=_default_base_url())
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    ap.add_argument("--no-image", action="store_true", help="skip the image test")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    tests = [
        ("plain chat + reasoning split", t_plain),
        ("tool call", t_tool_call),
        ("tool round trip", t_round_trip),
        ("streamed tool call", t_stream_tool),
        ("reasoning_effort accepted", t_reasoning_effort),
    ]
    if not args.no_image:
        tests.insert(4, ("image input", t_image))

    print(f"endpoint: {args.base_url}   model: {args.model}\n")
    failed = 0
    for i, (label, fn) in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {label}")
        try:
            err = fn(client, args.model)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        print(f"  -> {'PASS' if err is None else 'FAIL: ' + err}\n")
        failed += err is not None

    if failed:
        print(f"{failed}/{len(tests)} failed. Check the URL, that the server finished booting, and the server logs.")
        return 1
    print(f"all {len(tests)} passed — safe to point an agent harness at this endpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
