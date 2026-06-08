#!/usr/bin/env python3
"""Score the 4-axis subliminal-learning probe against an OpenAI-compat endpoint.

Run pre-SFT (baseline anchor) and post-SFT (compare vs anchor). The
two output files feed into `compare_anchor.py`, which applies the
2pt composite / 3pt per-axis pass thresholds.

This is a pure HTTP client — runs against any vLLM / SGLang /
OpenAI-compatible server. No model weights loaded locally. Point
`--endpoint` at the public `*.modal.run` URL the serve script prints
on `modal deploy`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MMLU_ANSWER_RE = re.compile(r"(?:answer\s*[:\-]\s*)?([ABCD])\b", re.IGNORECASE)
MMLU_FINAL_LETTER_RE = re.compile(r"\b([ABCD])\b")
INTEGER_RE = re.compile(r"-?\d[\d,]*")

AXIS_ORDER = ["mmlu_stem", "gsm8k", "ifeval", "humaneval"]


def load_probe(probe_dir: Path) -> dict[str, list[dict]]:
    probe_dir = Path(probe_dir)
    axis_files = {
        "mmlu_stem": probe_dir / "samples" / "mmlu_stem.jsonl",
        "gsm8k": probe_dir / "samples" / "gsm8k.jsonl",
        "ifeval": probe_dir / "samples" / "ifeval.jsonl",
        "humaneval": probe_dir / "samples" / "humaneval.jsonl",
    }
    items_by_axis: dict[str, list[dict]] = {}
    for axis, path in axis_files.items():
        if not path.exists():
            print(f"  WARNING: missing axis file {path}; axis will be skipped.",
                  file=sys.stderr)
            items_by_axis[axis] = []
            continue
        with path.open("r", encoding="utf-8") as f:
            items_by_axis[axis] = [json.loads(line) for line in f if line.strip()]
    return items_by_axis


def build_prompt(item: dict) -> str:
    axis = item["axis"]
    if axis == "mmlu_stem":
        choices = item["choices"]
        return (
            f"{item['prompt']}\n\n"
            f"A. {choices['A']}\n"
            f"B. {choices['B']}\n"
            f"C. {choices['C']}\n"
            f"D. {choices['D']}\n\n"
            "Answer with a single capital letter: A, B, C, or D.\n"
            "Answer:"
        )
    if axis == "gsm8k":
        return (
            f"{item['prompt']}\n\n"
            "Think step by step. Put the final integer answer on its own last line."
        )
    if axis == "ifeval":
        return item["prompt"]
    if axis == "humaneval":
        return (
            "Complete the following Python function. "
            "Respond with only the function definition (no surrounding prose, "
            "no markdown fences).\n\n"
            f"{item['prompt']}"
        )
    raise ValueError(f"unknown axis: {axis}")


# ─────────────────────────────────────────────────────────────────────
# Per-axis scoring
# ─────────────────────────────────────────────────────────────────────


def score_mmlu(item: dict, response: str) -> bool:
    text = response.strip()
    last_line = text.splitlines()[-1] if text else ""
    m = MMLU_ANSWER_RE.search(last_line) or MMLU_ANSWER_RE.search(text)
    if not m:
        m = MMLU_FINAL_LETTER_RE.search(last_line) or MMLU_FINAL_LETTER_RE.search(text)
    if not m:
        return False
    return m.group(1).upper() == item["correct"].upper()


def score_gsm8k(item: dict, response: str) -> bool:
    text = response.strip()
    if not text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        nums = INTEGER_RE.findall(line)
        if nums:
            try:
                parsed = int(nums[-1].replace(",", ""))
            except ValueError:
                continue
            return parsed == item["correct_number"]
    return False


def score_instruction(item: dict, response: str) -> tuple[bool | None, bool]:
    """Returns (correct_or_none, skipped). `None` means the constraint
    kind isn't supported and the item is skipped from the denominator."""
    kind = item["constraint"]["kind"]
    params = item["constraint"].get("params", {})
    text = response

    if kind == "length_constraints:number_sentences":
        want = params.get("num_sentences")
        relation = params.get("relation", "at least")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        count = len(sentences)
        if want is None:
            return (None, True)
        want = int(want)
        if relation == "at least":
            return (count >= want, False)
        if relation == "less than":
            return (count < want, False)
        if relation == "at most":
            return (count <= want, False)
        if relation == "more than":
            return (count > want, False)
        return (count == want, False)

    if kind == "startend:end_checker":
        ending = params.get("end_phrase")
        if ending is None:
            return (None, True)
        return (text.rstrip().endswith(ending), False)

    if kind == "punctuation:no_comma":
        return ("," not in text, False)

    if kind == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return (False, False)
        return (all(c.isupper() for c in letters), False)

    if kind == "keywords:existence":
        keywords = params.get("keywords") or []
        if not keywords:
            return (None, True)
        lowered = text.lower()
        return (all(kw.lower() in lowered for kw in keywords), False)

    return (None, True)


def _extract_python_code(response: str) -> str:
    text = response.strip()
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def score_code(item: dict, response: str, timeout_s: int = 10) -> bool:
    """Run the model's generated function against a single canonical
    HumanEval assert in a subprocess. Subprocess isolation keeps a
    runaway model output from cooking the scoring host."""
    code = _extract_python_code(response)
    entry_point = item["entry_point"]
    test_line = item["test"]

    if f"def {entry_point}" not in code:
        return False

    harness = (
        code
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + "    try:\n"
        + f"        {test_line}\n"
        + "        print('__SUBLIMINAL_PROBE_OK__')\n"
        + "    except Exception as exc:\n"
        + "        print('__SUBLIMINAL_PROBE_FAIL__', type(exc).__name__, exc)\n"
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8",
    ) as tf:
        tf.write(harness)
        tf_path = tf.name

    try:
        result = subprocess.run(
            [sys.executable, tf_path],
            capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return "__SUBLIMINAL_PROBE_OK__" in (result.stdout or "")
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tf_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────
# HTTP client + driver
# ─────────────────────────────────────────────────────────────────────


async def call_model(
    client, semaphore, model: str, item: dict,
    timeout: float, max_retries: int = 1,
) -> str:
    prompt = build_prompt(item)
    messages = [{"role": "user", "content": prompt}]
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 512,
                    },
                    timeout=timeout,
                )
            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            return content
        except Exception as exc:
            last_err = str(exc)
            if attempt >= max_retries:
                break
    return f"__SUBLIMINAL_PROBE_ERROR__ {last_err}"


async def run_axis(
    client, semaphore, model: str, axis: str, items: list[dict],
    timeout: float,
) -> dict:
    tasks = [call_model(client, semaphore, model, item, timeout) for item in items]
    responses = await asyncio.gather(*tasks)

    correct = 0
    skipped = 0
    errors = 0
    for item, response in zip(items, responses):
        if response.startswith("__SUBLIMINAL_PROBE_ERROR__"):
            errors += 1
            continue
        if axis == "mmlu_stem":
            if score_mmlu(item, response):
                correct += 1
        elif axis == "gsm8k":
            if score_gsm8k(item, response):
                correct += 1
        elif axis == "ifeval":
            ok, skip = score_instruction(item, response)
            if skip:
                skipped += 1
            elif ok:
                correct += 1
        elif axis == "humaneval":
            if score_code(item, response):
                correct += 1

    graded = len(items) - skipped - errors
    accuracy = (correct / graded) if graded > 0 else 0.0
    return {
        "accuracy": accuracy,
        "n": len(items),
        "correct": correct,
        "skipped": skipped,
        "errors": errors,
    }


async def amain(args) -> int:
    import httpx

    items_by_axis = load_probe(args.probe_dir)

    base_url = args.endpoint.rstrip("/")
    headers = {}
    # Optional bearer token, off unless `OPENAI_API_KEY` is set in the
    # environment. Modal endpoints are public by default and need no
    # key; if you locked yours down with proxy auth, set the matching
    # token here. See modal.com/docs/guide/webhook-proxy-auth.
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        axis_results: dict[str, dict] = {}
        for axis in AXIS_ORDER:
            items = items_by_axis.get(axis, [])
            if not items:
                continue
            print(f"[subliminal-probe] scoring {axis} (n={len(items)}) ...", flush=True)
            result = await run_axis(
                client, semaphore, args.model, axis, items, args.timeout,
            )
            axis_results[axis] = result
            print(
                f"  {axis:20s} acc={result['accuracy']:.4f} "
                f"({result['correct']}/{result['n']}, "
                f"skipped={result['skipped']}, errors={result['errors']})",
                flush=True,
            )

    per_axis_accs = [r["accuracy"] for r in axis_results.values()]
    composite = sum(per_axis_accs) / len(per_axis_accs) if per_axis_accs else 0.0

    output = {
        "label": args.label,
        "model": args.model,
        "endpoint": args.endpoint,
        "axes": axis_results,
        "composite": composite,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print()
    print(f"[subliminal-probe] composite = {composite:.4f}")
    print(f"  {'axis':20s} {'acc':>8s} {'n':>5s} {'correct':>8s} {'skipped':>8s} {'errors':>8s}")
    for axis, result in axis_results.items():
        print(
            f"  {axis:20s} {result['accuracy']:>8.4f} "
            f"{result['n']:>5d} {result['correct']:>8d} "
            f"{result['skipped']:>8d} {result['errors']:>8d}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"\n[subliminal-probe] wrote {out_path}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True,
                   help="OpenAI-compat base URL (no /v1 suffix needed).")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--probe-dir", type=Path, default=Path(__file__).parent,
        help="Directory containing samples/.",
    )
    p.add_argument("--label", default="unlabeled")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
