"""Capture one full response per benchmark prompt for quality grading.

The throughput bench (benchmark.py) only records timing + token counts —
it does NOT save the response text. This script sends one non-streaming
request per prompt in each profile and writes the full content (plus
reasoning_content for thinking-mode responses) to JSON for human review.

The default endpoint is read from the ``QWEN_ENDPOINT`` environment
variable, falling back to ``http://localhost:8000``. For a Modal
deployment, pass the public URL printed by ``modal deploy`` via
``--endpoint``.

Usage:
    uv run --with openai python qwen/bench/capture_samples.py \\
        --endpoint https://<the-url-printed-by-modal-deploy> \\
        --out /tmp/qwen-bench/samples.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import sys

from openai import AsyncOpenAI

# benchmark.py lives next to this script; let it be importable regardless
# of where this is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import (  # noqa: E402
    AGENTIC_TOOLS,
    AGENTIC_USER_TURNS,
    CODING_PROMPTS,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    _agentic_messages_for,
)


async def capture(
    client: AsyncOpenAI,
    *,
    model: str,
    label: str,
    messages: list[dict],
    max_tokens: int,
    tools: list[dict] | None = None,
) -> dict:
    t0 = time.perf_counter()
    try:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            stream=False,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                },
            },
        )
        if tools:
            kwargs["tools"] = tools
        resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        return {
            "label": label,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.perf_counter() - t0,
        }

    elapsed = time.perf_counter() - t0
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = [
        {
            "name": c.function.name,
            "arguments": c.function.arguments,  # qwen3_coder -> JSON string
        }
        for c in (msg.tool_calls or [])
    ]
    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "finish_reason": choice.finish_reason,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "reasoning_content": getattr(msg, "reasoning_content", None),
        "content": msg.content,
        "tool_calls": tool_calls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=(
            "Base URL of the OpenAI-compatible server "
            f"(default: $QWEN_ENDPOINT or {DEFAULT_ENDPOINT})."
        ),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # Auth is the user's choice — off by default. If your endpoint is
    # gated, set $API_KEY (or pass --api-key). Prefer Modal's own endpoint
    # protection: https://modal.com/docs/guide/webhooks#security
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    ap.add_argument("--max-tokens-coding", type=int, default=1024)
    ap.add_argument("--max-tokens-agentic", type=int, default=2048)
    ap.add_argument(
        "--out", default="/tmp/qwen-bench/samples.json",
        help="Where to dump the full samples (default: /tmp/qwen-bench/samples.json)",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    client = AsyncOpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key=args.api_key)

    async def _run() -> dict:
        results: dict = {"coding": [], "agentic": []}

        # Coding profile — short single-message prompts
        print("\n── coding profile (single-message prompts) ──")
        for i, prompt in enumerate(CODING_PROMPTS):
            label = f"coding-{i:02d}"
            print(f"  {label}: {prompt[:70]}...")
            r = await capture(
                client,
                model=args.model,
                label=label,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens_coding,
            )
            r["prompt"] = prompt
            print(f"    -> {r.get('completion_tokens')} tokens in {r.get('elapsed_s')}s")
            results["coding"].append(r)

        # Agentic profile — stable system+tools prefix + task turns
        print("\n── agentic profile (stable system+tools prefix + task turns) ──")
        for i, turn in enumerate(AGENTIC_USER_TURNS):
            label = f"agentic-{i:02d}"
            print(f"  {label}: {turn[:70]}...")
            r = await capture(
                client,
                model=args.model,
                label=label,
                messages=_agentic_messages_for(i),
                max_tokens=args.max_tokens_agentic,
                tools=AGENTIC_TOOLS,
            )
            r["user_turn"] = turn
            print(f"    -> {r.get('completion_tokens')} tokens in {r.get('elapsed_s')}s")
            results["agentic"].append(r)

        return results

    out = asyncio.run(_run())

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
