"""Throughput + metrics benchmark for the Qwen3.6-27B SGLang endpoint.

Targets the OpenAI-compatible ``/v1`` endpoint published by the Modal
deployment (``deployments/27b/solo/serve.py`` or
``deployments/27b/concurrent/serve.py``). The bench fires N requests at a
chosen concurrency (default sweep: 1, 2, 3 — to validate the
``max_running_requests=3`` fanout shape of the solo deployment), streams
responses to measure TTFT and decode tokens/sec per request, and scrapes
the SGLang ``/metrics`` endpoint (Prometheus format) for MTP acceptance
length, prefix-cache hit rate, and KV-cache occupancy.

Endpoint
--------
The default endpoint is read from the ``QWEN_ENDPOINT`` environment
variable, falling back to ``http://localhost:8000``. For a Modal
deployment, pass the public URL printed by ``modal deploy`` (a
``*.modal.run`` host) via ``--endpoint``; for a local SGLang server, the
``localhost`` default works as-is. The ``/metrics`` scrape is best-effort
and silently degrades when the endpoint does not expose it.

Usage
-----
    # Default: 1, 2, 3 concurrency sweep with built-in coding prompts.
    uv run --with openai python qwen/bench/benchmark.py

    # Single concurrency level, more requests, against a Modal URL.
    uv run --with openai python qwen/bench/benchmark.py \\
        --endpoint https://<the-url-printed-by-modal-deploy> \\
        --concurrency 3 \\
        --requests 20 \\
        --max-tokens 1024

    # Agentic profile: long stable system prefix + a per-request task,
    # exercises RadixAttention prefix-cache reuse.
    uv run --with openai python qwen/bench/benchmark.py --profile agentic

    # Save raw per-request samples for follow-up analysis.
    uv run --with openai python qwen/bench/benchmark.py --out bench.json

Output
------
Per-concurrency-level summary including:
  - TTFT (time-to-first-token) p50 / p95 / max
  - Decode tokens/sec per request (p50 / p95)
  - Total wall time + aggregate tokens/sec
  - MTP acceptance length (avg accepted draft tokens per target forward)
  - Prefix-cache hit rate
  - Peak KV-cache usage
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# Use the OpenAI-compatible client; SGLang exposes /v1/chat/completions.
from openai import AsyncOpenAI

# Default endpoint: env var, else localhost. Never a private host/IP.
DEFAULT_ENDPOINT = os.environ.get("QWEN_ENDPOINT", "http://localhost:8000")
DEFAULT_MODEL = "qwen3.6-27b"


# ─────────────────────────────────────────────────────────────────────
# `coding` profile — short single-message agentic-coding prompts
# ─────────────────────────────────────────────────────────────────────
# Each prompt is short by design (the bench measures decode speed, not
# prefill — we don't want long prefill dominating the wall time). For
# prefill-heavy benchmarks, point --prompts-file at a file with longer
# inputs, or use --profile agentic.

CODING_PROMPTS: list[str] = [
    "Write a Python function that returns the n-th Fibonacci number using memoization. Include doctests.",
    "Implement a Rust struct `RingBuffer<T>` with push_back, pop_front, and an Iterator impl. Use VecDeque internally.",
    "Write a TypeScript type-safe debounce helper that preserves argument and return types via generics.",
    "Refactor this function to use early returns: def f(x):\\n  if x > 0:\\n    if x < 100:\\n      return x * 2\\n    else:\\n      return 0\\n  else:\\n    return -1",
    "Explain when to use `tokio::spawn` vs `tokio::task::spawn_blocking` in Rust async code.",
    "Write a Go function `Parallel(fns []func() error) []error` that runs all fns concurrently and returns errors in input order.",
    "Implement an LRU cache decorator in Python without importing functools.lru_cache. Include type hints with ParamSpec.",
    "Walk through the time complexity of inserting into a B-tree of order m with n existing keys.",
]


# ─────────────────────────────────────────────────────────────────────
# `agentic` profile — long stable system prefix + a per-request task
# ─────────────────────────────────────────────────────────────────────
# A generic coding-assistant system prompt plus a public example tool set
# stands in for a real agent harness's stable context. The system prompt
# repeats across every request in the rotation, so RadixAttention's prefix
# cache should kick in after the first request; the bench reports decode
# TPS that benefits from this, just like a real agentic workload will.
# Nothing here is account- or workload-specific — swap in your own system
# prompt and tools to mirror your harness.

AGENTIC_SYSTEM_PROMPT = """You are a coding assistant operating inside an agentic harness. You
work on a software project by reading code, running commands, and editing
files through the tools available to you. Follow these guidelines:

1. Prefer reading the relevant code before proposing a change. Do not
   guess at APIs you have not inspected.
2. Make the smallest change that satisfies the request. Keep diffs tight
   and explain non-obvious decisions briefly.
3. When a task needs information you do not have, call a tool to get it
   rather than fabricating an answer.
4. State-changing actions (writing files, running commands) should be
   clearly justified before you take them.
5. Lead with the answer or the plan, then the supporting detail. Use
   plain, direct language and format code in fenced blocks.

You have access to the following tools: read_file, write_file,
run_command, search_code, list_directory. Plan your tool use, prefer one
well-formed call to several vague ones, and report results concretely.
"""

# A public example tool set — the kind of generic developer tools an
# agentic coding harness exposes. These are descriptive only; they exist
# to give the prefill a realistic, stable shape, not to be executed.
AGENTIC_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or repo-relative file path.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a file with the given contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "contents": {
                        "type": "string",
                        "description": "Full file contents to write.",
                    },
                },
                "required": ["path", "contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the project working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the project for a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to scope the search to.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the entries in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# A few realistic agent task turns — vary the question type so the bench
# exercises both short lookups and longer synthesis turns. The stable
# system prompt + tools above form the cacheable prefix.
AGENTIC_USER_TURNS: list[str] = [
    "The test suite is failing with an ImportError on `utils.config`. Find the cause and propose a fix.",
    (
        "Add a `--dry-run` flag to the CLI in `cli.py` that prints the actions it "
        "would take without executing them. Walk me through your plan before editing."
    ),
    (
        "Review the error handling in the HTTP client module. Where could an "
        "unhandled exception crash the worker, and how would you make it resilient?"
    ),
]


def _agentic_messages_for(turn_idx: int) -> list[dict]:
    """Build the messages list for an `agentic`-profile request.

    The stable prefix (system prompt) repeats across all prompts in the
    rotation, so RadixAttention's prefix cache should engage after the
    first request.
    """
    return [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": AGENTIC_USER_TURNS[turn_idx % len(AGENTIC_USER_TURNS)],
        },
    ]


# ─────────────────────────────────────────────────────────────────────
# Per-request sample + aggregate summary
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    prompt_idx: int
    ttft_s: float
    e2e_s: float
    output_tokens: int
    prompt_tokens: int | None
    decode_tps: float  # output_tokens / (e2e - ttft)
    error: str | None = None


@dataclass
class Summary:
    concurrency: int
    requests: int
    wall_s: float
    ttft_p50: float
    ttft_p95: float
    ttft_max: float
    decode_tps_p50: float
    decode_tps_p95: float
    aggregate_tps: float
    output_tokens_total: int
    errors: int
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ─────────────────────────────────────────────────────────────────────
# Streaming request — captures TTFT + decode rate
# ─────────────────────────────────────────────────────────────────────


async def run_one(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict],
    prompt_idx: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> Sample:
    t_start = time.perf_counter()
    t_first: float | None = None
    output_tokens = 0
    prompt_tokens: int | None = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={
                # top_k / min_p are SGLang extensions, not OpenAI fields,
                # so they go in extra_body. chat_template_kwargs is read by
                # SGLang as a top-level body field.
                "top_k": 20,
                "min_p": 0.0,
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                    "preserve_thinking": True,
                },
            },
        )

        async for chunk in stream:
            # Final chunk carries usage stats when include_usage=True.
            if chunk.usage is not None:
                output_tokens = chunk.usage.completion_tokens or output_tokens
                prompt_tokens = chunk.usage.prompt_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and (delta.content or getattr(delta, "reasoning_content", None)):
                if t_first is None:
                    t_first = time.perf_counter()
                # Crude per-chunk token count fallback in case usage isn't
                # reported. SGLang DOES report usage when
                # stream_options.include_usage is set, but be defensive.
                output_tokens += 1
    except Exception as e:
        return Sample(
            prompt_idx=prompt_idx,
            ttft_s=0.0,
            e2e_s=time.perf_counter() - t_start,
            output_tokens=0,
            prompt_tokens=None,
            decode_tps=0.0,
            error=f"{type(e).__name__}: {e}",
        )

    t_end = time.perf_counter()
    ttft = (t_first or t_end) - t_start
    e2e = t_end - t_start
    decode_window = max(e2e - ttft, 1e-6)
    decode_tps = output_tokens / decode_window if output_tokens else 0.0

    return Sample(
        prompt_idx=prompt_idx,
        ttft_s=ttft,
        e2e_s=e2e,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        decode_tps=decode_tps,
    )


# ─────────────────────────────────────────────────────────────────────
# /metrics scrape
# ─────────────────────────────────────────────────────────────────────


async def fetch_metrics(endpoint: str) -> dict[str, Any]:
    """Pull the SGLang Prometheus endpoint and extract the metrics we care
    about. We don't drag in prometheus_client — minimal text parsing. The
    scrape is best-effort: if the endpoint doesn't serve /metrics we just
    report the error and move on.
    """
    import urllib.error
    import urllib.request

    url = endpoint.rstrip("/") + "/metrics"
    keys_of_interest = (
        # MTP / speculative
        "sglang:spec_decode_acceptance_length",
        "sglang:spec_decode_acceptance_rate",
        "sglang:num_accepted_tokens",
        "sglang:num_draft_tokens",
        # Prefix cache
        "sglang:cache_hit_rate",
        "sglang:cached_tokens",
        # KV / scheduler
        "sglang:num_running_reqs",
        "sglang:num_queue_reqs",
        "sglang:token_usage",
        "sglang:gen_throughput",
    )
    snapshot: dict[str, Any] = {}
    try:
        loop = asyncio.get_running_loop()

        def _fetch() -> str:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.read().decode("utf-8", errors="replace")

        body = await loop.run_in_executor(None, _fetch)
    except (urllib.error.URLError, TimeoutError) as e:
        return {"_error": f"could not scrape /metrics: {e}"}

    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Lines look like: metric_name{labels} value [timestamp]
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if not any(name.startswith(k) for k in keys_of_interest):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            snapshot.setdefault(name, []).append(float(parts[1]))
        except ValueError:
            continue
    return snapshot


# ─────────────────────────────────────────────────────────────────────
# Concurrency level driver
# ─────────────────────────────────────────────────────────────────────


async def run_level(
    *,
    endpoint: str,
    model: str,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    message_builder,
    n_prompts: int,
    api_key: str,
) -> tuple[Summary, list[Sample]]:
    base_url = endpoint.rstrip("/") + "/v1"
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    samples: list[Sample] = []

    async def driven(i: int) -> Sample:
        async with sem:
            idx = i % n_prompts
            return await run_one(
                client,
                model=model,
                messages=message_builder(idx),
                prompt_idx=idx,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )

    t_start = time.perf_counter()
    samples = await asyncio.gather(*[driven(i) for i in range(total_requests)])
    wall = time.perf_counter() - t_start

    metrics = await fetch_metrics(endpoint)
    ok = [s for s in samples if not s.error]

    ttfts = [s.ttft_s for s in ok]
    decode_tps = [s.decode_tps for s in ok if s.decode_tps > 0]
    total_out = sum(s.output_tokens for s in ok)
    aggregate_tps = total_out / wall if wall > 0 else 0.0

    summary = Summary(
        concurrency=concurrency,
        requests=total_requests,
        wall_s=wall,
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_max=max(ttfts) if ttfts else 0.0,
        decode_tps_p50=percentile(decode_tps, 50),
        decode_tps_p95=percentile(decode_tps, 95),
        aggregate_tps=aggregate_tps,
        output_tokens_total=total_out,
        errors=len(samples) - len(ok),
        metrics_snapshot=metrics,
    )
    return summary, samples


# ─────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────


def print_summary(s: Summary) -> None:
    print(f"\n┌─ concurrency={s.concurrency}  requests={s.requests}  errors={s.errors}")
    print(f"│  wall              {s.wall_s:>8.2f} s")
    print(f"│  ttft   p50 / p95  {s.ttft_p50:>8.3f} / {s.ttft_p95:.3f} s   max {s.ttft_max:.3f}")
    print(f"│  decode p50 / p95  {s.decode_tps_p50:>8.1f} / {s.decode_tps_p95:.1f} tok/s/req")
    print(f"│  aggregate         {s.aggregate_tps:>8.1f} tok/s  ({s.output_tokens_total} tokens)")
    m = s.metrics_snapshot
    if m and "_error" not in m:
        print("│  /metrics:")
        for k, v in sorted(m.items()):
            if v:
                last = v[-1]
                print(f"│    {k:<48} {last:.4f}")
    elif m and "_error" in m:
        print(f"│  /metrics: {m['_error']}")
    print("└─")


# ─────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=(
            "Base URL of the OpenAI-compatible server "
            f"(default: $QWEN_ENDPOINT or {DEFAULT_ENDPOINT}). For a Modal "
            "deployment, pass the URL printed by `modal deploy`."
        ),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--concurrency",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Concurrency levels to sweep (default 1 2 3 to match max_running_requests)",
    )
    ap.add_argument(
        "--requests",
        type=int,
        default=8,
        help="Requests per concurrency level (default 8 == one full prompt rotation)",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable <think> blocks (faster; less accurate for agentic coding)",
    )
    # Auth is the user's choice — this stays off by default. If your
    # endpoint is gated, set $API_KEY (or pass --api-key). Prefer Modal's
    # own endpoint protection: https://modal.com/docs/guide/webhooks#security
    ap.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", "EMPTY"),
        help="Bearer token if the endpoint is api-key-gated (default: read $API_KEY or 'EMPTY')",
    )
    ap.add_argument(
        "--profile",
        choices=("coding", "agentic"),
        default="coding",
        help=(
            "Workload shape. 'coding' (default): short single-message "
            "agentic-coding prompts — useful as a quick smoke test. "
            "'agentic': a long stable system prompt + tool set + per-request "
            "task turn — exercises prefix-cache reuse with a realistic "
            "agent-harness input shape."
        ),
    )
    ap.add_argument(
        "--prompts-file",
        help="Optional path to a JSON list of strings (coding profile only)",
    )
    ap.add_argument("--out", help="Optional path to dump full samples + summaries as JSON")
    args = ap.parse_args()

    if args.profile == "agentic":
        message_builder = _agentic_messages_for
        n_prompts = len(AGENTIC_USER_TURNS)
        profile_desc = (
            f"agentic ({n_prompts} task turns sharing a stable system+tools prefix)"
        )
    else:
        if args.prompts_file:
            with open(args.prompts_file) as f:
                prompts = json.load(f)
            assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)
        else:
            prompts = CODING_PROMPTS

        def message_builder(idx: int) -> list[dict]:
            return [{"role": "user", "content": prompts[idx]}]

        n_prompts = len(prompts)
        profile_desc = f"coding ({n_prompts} prompts)"

    print(f"endpoint: {args.endpoint}")
    print(f"model:    {args.model}")
    print(f"profile:  {profile_desc}")
    print(f"sweep:    {args.concurrency} × {args.requests} req each")

    results: dict[str, Any] = {"args": vars(args), "levels": []}

    async def _run() -> None:
        for c in args.concurrency:
            s, samples = await run_level(
                endpoint=args.endpoint,
                model=args.model,
                concurrency=c,
                total_requests=args.requests,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=not args.no_thinking,
                message_builder=message_builder,
                n_prompts=n_prompts,
                api_key=args.api_key,
            )
            print_summary(s)
            results["levels"].append(
                {"summary": asdict(s), "samples": [asdict(x) for x in samples]}
            )

    asyncio.run(_run())

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nfull data written to {args.out}")


if __name__ == "__main__":
    main()
