#!/usr/bin/env python3
"""Async throughput + latency benchmark for an OpenAI-compatible endpoint.

Sends a fixed pool of prompts at a target concurrency and reports:

  - total wall time
  - aggregate output-tokens/second
  - per-request latency P50 / P95 / mean
  - mean output token count

Designed for apples-to-apples comparison of vLLM / SGLang serve
configurations (MTP on/off, different sizes, different sampling params).
Same prompt pool + same concurrency + same `max_tokens` ceiling on every
run; only the endpoint URL changes.

The endpoint is the public `*.modal.run` URL printed by `modal deploy`
for one of the serve scripts (the server binds `0.0.0.0`, so Modal's
ingress publishes it). Point `--endpoint` at that base URL (no `/v1`
suffix).

Run:

    uv run python scripts/bench_endpoint.py \\
        --endpoint <the URL printed by modal deploy> \\
        --model google/gemma-4-31B-it \\
        --concurrency 4 \\
        --requests 20 \\
        --max-tokens 256 \\
        --label "31B-mtp-off-bf16" \\
        --out scripts/bench-results/31b-no-mtp.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx


# Fixed prompt pool — short, mid, and longer prompts. Designed to elicit
# 100-300 token responses so each request exercises decode rather than
# being dominated by prefill. Chosen to be neutral / non-task-specific
# so the benchmark stays comparable across model checkpoints.
PROMPTS: list[str] = [
    "Explain in three sentences why low-rank adaptation is more parameter-efficient than full fine-tuning.",
    "Write a short Python function that returns the n-th Fibonacci number using memoization.",
    "Summarise the trade-offs between sparse mixture-of-experts and dense Transformer architectures in one paragraph.",
    "List five practical strategies for reducing GPU memory usage during large-language-model training.",
    "Describe what speculative decoding is and how a draft-and-verify cycle preserves output quality.",
    "Compare and contrast prefix caching and KV-cache fp8 quantization as inference optimizations.",
    "What is a chat template, and why does masking only assistant tokens matter for SFT?",
    "Outline a four-step plan for benchmarking a new language model on a tool-use rubric.",
    "Explain the difference between greedy decoding and nucleus sampling in two short paragraphs.",
    "Sketch the architecture of a Retrieval-Augmented Generation system that grounds a chat agent on a SQL database.",
]


@dataclass
class RequestResult:
    """Per-request metric capture."""

    index: int
    latency_s: float
    output_tokens: int | None
    completion_text_len: int
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Aggregate results, suitable for JSON dump."""

    label: str
    endpoint: str
    model: str
    concurrency: int
    requests: int
    max_tokens: int
    timestamp: str
    wall_time_s: float
    total_output_tokens: int
    throughput_tokens_per_s: float
    successful_requests: int
    failed_requests: int
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    output_tokens_mean: float
    per_request: list[dict]


async def _one_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    api_key: str | None,
    semaphore: asyncio.Semaphore,
    index: int,
) -> RequestResult:
    """Issue one chat completion. Captures latency from request-send to
    response-fully-received (NOT time-to-first-token; we'd need
    streaming for that). Output token count comes from the server's
    `usage.completion_tokens` — vLLM populates it correctly."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Disable thinking explicitly so the comparison measures raw
        # decode throughput, not thinking-token overhead. Per-request
        # via chat_template_kwargs; server default is irrelevant once
        # this is set.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, headers=headers, json=body, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - t0
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            output_tokens = usage.get("completion_tokens")
            return RequestResult(
                index=index,
                latency_s=latency,
                output_tokens=int(output_tokens) if output_tokens is not None else None,
                completion_text_len=len(text),
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            return RequestResult(
                index=index,
                latency_s=latency,
                output_tokens=None,
                completion_text_len=0,
                error=f"{type(e).__name__}: {e}",
            )


async def benchmark(args: argparse.Namespace) -> BenchmarkReport:
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    semaphore = asyncio.Semaphore(args.concurrency)

    # Warm-up pass: a single synchronous request first so the comparison
    # isn't dominated by the initial Triton/torch.compile compile cost.
    print(f"[bench] warming up the endpoint with one request...", flush=True)
    async with httpx.AsyncClient() as client:
        warm = await _one_request(
            client, args.endpoint, args.model, PROMPTS[0],
            max_tokens=64, api_key=api_key, semaphore=semaphore, index=-1,
        )
        if warm.error:
            print(f"[bench] warm-up FAILED: {warm.error}", file=sys.stderr)
            sys.exit(2)
        print(f"[bench] warm-up ok ({warm.latency_s:.2f}s, "
              f"{warm.output_tokens} tokens)", flush=True)

        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
        print(f"[bench] running {args.requests} requests at concurrency={args.concurrency}...",
              flush=True)
        t0 = time.perf_counter()
        tasks = [
            _one_request(
                client, args.endpoint, args.model, p,
                max_tokens=args.max_tokens, api_key=api_key,
                semaphore=semaphore, index=i,
            )
            for i, p in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    if failed:
        print(f"[bench] WARNING: {len(failed)}/{len(results)} requests failed",
              file=sys.stderr)
        for r in failed[:3]:
            print(f"  err {r.index}: {r.error}", file=sys.stderr)

    if not successful:
        print("[bench] ALL REQUESTS FAILED — aborting", file=sys.stderr)
        sys.exit(2)

    latencies = sorted(r.latency_s for r in successful)
    output_tokens_list = [
        r.output_tokens for r in successful if r.output_tokens is not None
    ]
    total_output_tokens = sum(output_tokens_list) if output_tokens_list else 0
    mean_output = (
        statistics.mean(output_tokens_list) if output_tokens_list else 0.0
    )

    def _pct(values: list[float], pct: float) -> float:
        # Inclusive nearest-rank percentile (small N → full sort already done).
        if not values:
            return 0.0
        k = max(0, min(len(values) - 1, int(round((pct / 100.0) * len(values))) - 1))
        return values[k]

    report = BenchmarkReport(
        label=args.label,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        requests=args.requests,
        max_tokens=args.max_tokens,
        timestamp=datetime.now(timezone.utc).isoformat(),
        wall_time_s=round(wall, 3),
        total_output_tokens=total_output_tokens,
        throughput_tokens_per_s=round(total_output_tokens / wall, 2) if wall > 0 else 0.0,
        successful_requests=len(successful),
        failed_requests=len(failed),
        latency_mean_s=round(statistics.mean(latencies), 3) if latencies else 0.0,
        latency_p50_s=round(_pct(latencies, 50), 3),
        latency_p95_s=round(_pct(latencies, 95), 3),
        output_tokens_mean=round(mean_output, 1),
        per_request=[asdict(r) for r in results],
    )

    print(f"\n=== {args.label} ===")
    print(f"  endpoint:        {args.endpoint}")
    print(f"  model:           {args.model}")
    print(f"  concurrency:     {args.concurrency}")
    print(f"  requests:        {args.requests} ({len(successful)} ok, {len(failed)} err)")
    print(f"  wall time:       {wall:.2f}s")
    print(f"  total tokens:    {total_output_tokens}")
    print(f"  throughput:      {report.throughput_tokens_per_s:.1f} tok/s")
    print(f"  latency mean:    {report.latency_mean_s:.2f}s")
    print(f"  latency P50:     {report.latency_p50_s:.2f}s")
    print(f"  latency P95:     {report.latency_p95_s:.2f}s")
    print(f"  mean out tokens: {report.output_tokens_mean:.1f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
        print(f"\n[bench] wrote {out_path}")

    return report


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True,
                   help="Base URL of the served endpoint, no /v1 suffix "
                        "(the *.modal.run URL printed by `modal deploy`).")
    p.add_argument("--model", required=True,
                   help="Model name (sent in the request body).")
    p.add_argument("--label", required=True,
                   help="Free-form tag, included in the JSON report (e.g. '31B-mtp-on').")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the JSON report to this path.")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=256)
    # Optional bearer token, OFF by default. Modal endpoints are public
    # by default and this benchmark assumes that; only set this if you
    # have locked the endpoint down with proxy auth — see
    # modal.com/docs/guide/webhook-proxy-auth. No auth is implemented here.
    p.add_argument("--api-key-env", default=None,
                   help="Optional env var holding a bearer token, if the "
                        "endpoint is gated. Off by default.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(benchmark(_cli()))
