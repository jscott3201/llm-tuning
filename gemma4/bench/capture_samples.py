"""Capture one full response per benchmark prompt for quality grading.

The throughput bench (benchmark.py) records only timing + token counts —
it does NOT save response text. This script sends one non-streaming
request per prompt in each profile and writes the full content (plus
`reasoning_content` for thinking-mode responses and any `tool_calls`) to
JSON for human review.

Targets `$ENDPOINT` (default http://localhost:8000). For a Modal
deployment, point it at the public *.modal.run URL printed by
`modal deploy`.

Usage:
    uv run --with openai python bench/capture_samples.py \\
        --out bench-results/samples.json

    ENDPOINT=https://<the-url-printed-by-modal-deploy> \\
        uv run --with openai python bench/capture_samples.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from openai import AsyncOpenAI

# benchmark.py lives next to this script; make it importable regardless
# of where this is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import (  # noqa: E402
    AGENTIC_TOOLS,
    AGENTIC_USER_TURNS,
    BUILTIN_PROMPTS,
    SAMPLING,
    _agentic_messages_for,
)


async def capture(
    client: AsyncOpenAI,
    *,
    model: str,
    label: str,
    messages: list[dict],
    max_tokens: int,
    sampling: dict,
    enable_thinking: bool,
    tools: list[dict] | None = None,
) -> dict:
    t0 = time.perf_counter()
    try:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            stream=False,
            extra_body={
                "top_k": sampling["top_k"],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
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
    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "finish_reason": choice.finish_reason,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        # SGLang's gemma4 reasoning parser surfaces the <|channel>thought
        # block here.
        "reasoning_content": getattr(msg, "reasoning_content", None),
        "content": msg.content,
        # Surface any tool calls so the capture doubles as a quick
        # tool-calling smoke check.
        "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--endpoint",
        default=os.environ.get("ENDPOINT", "http://localhost:8000"),
        help="OpenAI-compatible base (default: $ENDPOINT or "
        "http://localhost:8000). For a Modal deployment, use the URL "
        "printed by `modal deploy`.",
    )
    ap.add_argument("--model", default="gemma-4-31b-it")
    # Optional bearer token; OFF by default. To secure a Modal web endpoint
    # instead, see Modal's endpoint-security docs (proxy auth tokens).
    ap.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", "EMPTY"),
        help="Optional bearer token if the endpoint is api-key-gated "
        "(default: $API_KEY or 'EMPTY' = no auth)",
    )
    ap.add_argument(
        "--sampling",
        choices=tuple(SAMPLING.keys()),
        default="official",
        help="Sampling preset (see benchmark.py)",
    )
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--max-tokens-coding", type=int, default=4096)
    ap.add_argument(
        "--max-tokens-agentic",
        type=int,
        default=3072,
        help="Agentic cap. Gemma 4 thinking can be token-hungry — size "
        "generously or responses truncate inside the thought block.",
    )
    ap.add_argument("--out", default="bench-results/samples.json")
    args = ap.parse_args()

    sampling = SAMPLING[args.sampling]
    enable_thinking = not args.no_thinking
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    client = AsyncOpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key=args.api_key)

    async def _run() -> dict:
        results: dict = {
            "sampling": {"preset": args.sampling, **sampling},
            "enable_thinking": enable_thinking,
            "coding": [],
            "agentic": [],
        }

        print("\n── coding profile (single-message prompts) ──")
        for i, prompt in enumerate(BUILTIN_PROMPTS):
            label = f"coding-{i:02d}"
            print(f"  {label}: {prompt[:70]}...")
            r = await capture(
                client,
                model=args.model,
                label=label,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens_coding,
                sampling=sampling,
                enable_thinking=enable_thinking,
            )
            r["prompt"] = prompt
            print(f"    -> {r.get('completion_tokens')} tokens in {r.get('elapsed_s')}s")
            results["coding"].append(r)

        print("\n── agentic profile (stable prefix + example tools) ──")
        for i, turn in enumerate(AGENTIC_USER_TURNS):
            label = f"agentic-{i:02d}"
            print(f"  {label}: {turn[:70]}...")
            r = await capture(
                client,
                model=args.model,
                label=label,
                messages=_agentic_messages_for(i),
                max_tokens=args.max_tokens_agentic,
                sampling=sampling,
                enable_thinking=enable_thinking,
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
