#!/usr/bin/env python3
"""Smoke test for the Qwen3.6-27B (SGLang on Modal) OpenAI-compatible endpoint.

Confirms the endpoint (a) answers a plain chat completion and (b) emits a
well-formed tool call, BEFORE you point an agentic harness at it. If the
tool-call test fails here, a tool-using client will fail too.

Run (no install needed):

    uv run --with openai python smoke_test.py --base-url http://localhost:8000/v1

The default --base-url is read from the QWEN_ENDPOINT environment variable
(with a trailing /v1 appended if missing) and otherwise falls back to
http://localhost:8000/v1. For a Modal deployment, pass the public URL
printed by `modal deploy` (a *.modal.run host) with /v1 appended, e.g.:

    uv run --with openai python smoke_test.py \\
        --base-url https://<the-url-printed-by-modal-deploy>/v1

Defaults match the deployment: model qwen3.6-27b, the Qwen3.6 "precise
coding" sampling recipe baked in, non-streaming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("FAIL: the `openai` package is not available.")
    print("      Run this as:  uv run --with openai python smoke_test.py ...")
    sys.exit(2)


# --- Defaults ----------------------------------------------------------------
def _default_base_url() -> str:
    """Default base URL: $QWEN_ENDPOINT (with /v1 appended) else localhost."""
    endpoint = os.environ.get("QWEN_ENDPOINT", "http://localhost:8000").rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    return endpoint


DEFAULT_BASE_URL = _default_base_url()
DEFAULT_MODEL = "qwen3.6-27b"
# Auth is the user's choice — this server ships OPEN, so the key is unused
# by default. If your endpoint is gated, pass --api-key (or set $API_KEY).
# Prefer Modal's own endpoint protection:
#   https://modal.com/docs/guide/webhooks#security
DEFAULT_API_KEY = os.environ.get("API_KEY", "EMPTY")

# Qwen3.6 "precise coding" sampling recipe.
#   - temperature/top_p are standard OpenAI fields.
#   - top_k/min_p are SGLang EXTENSIONS -> must go in extra_body, never as
#     top-level kwargs (the OpenAI client would reject them).
#   - chat_template_kwargs (enable_thinking/preserve_thinking) likewise goes in
#     extra_body and is read by SGLang as a top-level body field.
TEMPERATURE = 0.6
TOP_P = 0.95
EXTRA_BODY = {
    "top_k": 20,
    "min_p": 0.0,
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": True,
    },
}

# max_tokens FOOTGUN: enable_thinking burns 800-4000+ reasoning tokens inside
# <think> BEFORE any answer content. An undersized cap returns
# finish_reason="length" with content=None (truncated mid-think), NOT a real
# answer. The smoke-test prompts are tiny so a modest cap is fine here; real
# agent work should use >= 16384.
MAX_TOKENS = 4096


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for.",
                },
            },
            "required": ["city"],
        },
    },
}


def _truncate(text: str, limit: int = 200) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def test_plain_completion(client: OpenAI, model: str) -> bool:
    print("=" * 70)
    print("TEST 1: plain chat completion (non-streaming)")
    print("-" * 70)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly the word: OK"},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stream=False,
            extra_body=EXTRA_BODY,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  request error: {exc!r}")
        print("RESULT: FAIL (request raised an exception)")
        return False

    choice = resp.choices[0]
    msg = choice.message
    finish = choice.finish_reason
    content = msg.content
    reasoning = getattr(msg, "reasoning_content", None)
    usage = resp.usage

    print(f"  finish_reason     : {finish}")
    print(f"  content           : {_truncate(content) if content else content!r}")
    if reasoning:
        print(f"  reasoning_content : present ({len(reasoning)} chars, SGLang ext)")
    else:
        print("  reasoning_content : (none)")
    if usage:
        print(
            f"  usage             : prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )

    if finish == "length" and not content:
        print("RESULT: FAIL (finish_reason=length with empty content -> "
              "truncated inside <think>; raise max_tokens)")
        return False
    if not content or not content.strip():
        print("RESULT: FAIL (empty content)")
        return False

    print("RESULT: PASS")
    return True


def test_tool_call(client: OpenAI, model: str) -> bool:
    print("=" * 70)
    print("TEST 2: single tool call -> get_weather (non-streaming)")
    print("-" * 70)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the get_weather tool.",
                },
            ],
            tools=[WEATHER_TOOL],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stream=False,
            extra_body=EXTRA_BODY,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  request error: {exc!r}")
        print("RESULT: FAIL (request raised an exception)")
        return False

    choice = resp.choices[0]
    msg = choice.message
    finish = choice.finish_reason
    tool_calls = msg.tool_calls or []

    print(f"  finish_reason : {finish}")
    print(f"  tool_calls    : {len(tool_calls)}")

    if not tool_calls:
        print("  (model returned no tool calls)")
        if msg.content:
            print(f"  content       : {_truncate(msg.content)}")
        print("RESULT: FAIL (endpoint did not emit a tool call)")
        return False

    call = tool_calls[0]
    name = call.function.name
    raw_args = call.function.arguments  # qwen3_coder -> JSON STRING
    print(f"  call[0].name  : {name}")
    print(f"  raw arguments : {raw_args!r}  (JSON string, qwen3_coder format)")

    if name != "get_weather":
        print(f"RESULT: FAIL (called '{name}', expected 'get_weather')")
        return False

    # qwen3_coder returns arguments as a JSON-encoded STRING (OpenAI wire
    # format). The harness must json.loads it once on ingest.
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"  json.loads error: {exc!r}")
        print("RESULT: FAIL (arguments are not valid JSON)")
        return False

    print(f"  parsed args   : {parsed}")
    if not isinstance(parsed, dict) or not parsed.get("city"):
        print("RESULT: FAIL (parsed arguments missing 'city')")
        return False

    print("RESULT: PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test the Qwen3.6-27B SGLang OpenAI endpoint "
                    "(plain completion + tool call)."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"OpenAI base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                        help="API key (default: EMPTY; the server ships open)")
    args = parser.parse_args()

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Sampling : temperature={TEMPERATURE} top_p={TOP_P} "
          f"extra_body={EXTRA_BODY}")
    print(f"max_tokens={MAX_TOKENS}  stream=False")
    print("")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results = []
    results.append(test_plain_completion(client, args.model))
    print("")
    results.append(test_tool_call(client, args.model))
    print("")

    print("=" * 70)
    passed = sum(1 for r in results if r)
    total = len(results)
    if all(results):
        print(f"OVERALL: PASS ({passed}/{total}) -- endpoint speaks tools. "
              "Safe to point an agentic harness at it.")
        return 0
    print(f"OVERALL: FAIL ({passed}/{total}) -- do NOT launch the harness yet. "
          "Check the endpoint URL is correct and the server is fully booted.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
