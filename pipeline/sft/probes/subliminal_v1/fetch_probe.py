#!/usr/bin/env python3
"""Fetch the four public-benchmark axes of the subliminal-learning probe.

Same-family distillation can transmit misalignment via unrelated
tasks; this probe establishes a 4-axis baseline that post-SFT models
must not regress on by more than 2pt composite / 3pt per-axis.

Axes fetched here:
  - mmlu_stem      (cais/mmlu, stem subjects, stratified across 7 subjects)
  - gsm8k          (openai/gsm8k main config, test split)
  - ifeval         (google/IFEval, filtered to machine-checkable constraints)
  - humaneval      (openai/openai_humaneval, canonical solution <= 20 real LOC)

All four datasets are public on the Hugging Face Hub. If you want an
extra axis tailored to your own domain, hand-author it and have
`score_probe.py` recognise the axis name.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


MMLU_STEM_SUBJECTS = [
    "high_school_physics",
    "college_chemistry",
    "high_school_biology",
    "college_computer_science",
    "astronomy",
    "college_mathematics",
    "high_school_mathematics",
]

# Instruction-following constraint kinds the scorer can check without
# human judgment. Anything outside this set is skipped at scoring time
# (which keeps the axis honest -- we don't grade on guesswork).
IFEVAL_SUPPORTED_CONSTRAINTS = {
    "length_constraints:number_sentences",
    "startend:end_checker",
    "punctuation:no_comma",
    "change_case:english_capital",
    "keywords:existence",
}


def _write_jsonl(path: Path, items: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def _count_code_lines(src: str) -> int:
    count = 0
    for raw in src.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        count += 1
    return count


def fetch_mmlu_stem(n: int, seed: int) -> list[dict]:
    """Stratify across the 7 STEM subjects so the axis isn't dominated
    by any single one. Remainder distribution is deterministic on the
    subject order."""
    from datasets import load_dataset

    rng = random.Random(seed)
    per_subject = max(1, n // len(MMLU_STEM_SUBJECTS))
    remainder = n - per_subject * len(MMLU_STEM_SUBJECTS)

    items: list[dict] = []
    for i, subject in enumerate(MMLU_STEM_SUBJECTS):
        take = per_subject + (1 if i < remainder else 0)
        ds = load_dataset("cais/mmlu", subject, split="test")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        picked = idxs[:take]
        for rank, idx in enumerate(picked):
            row = ds[int(idx)]
            choices = row["choices"]
            correct_letter = ["A", "B", "C", "D"][int(row["answer"])]
            items.append({
                "axis": "mmlu_stem",
                "id": f"mmlu-{subject}-{rank}",
                "prompt": row["question"],
                "choices": {
                    "A": choices[0], "B": choices[1],
                    "C": choices[2], "D": choices[3],
                },
                "correct": correct_letter,
            })
    return items[:n]


_GSM8K_ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,]*)")


def fetch_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    items: list[dict] = []
    for idx in idxs:
        row = ds[int(idx)]
        m = _GSM8K_ANS_RE.search(row["answer"])
        if not m:
            continue
        try:
            ans = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        items.append({
            "axis": "gsm8k",
            "id": f"gsm8k-{len(items)}",
            "prompt": row["question"],
            "correct_number": ans,
        })
        if len(items) >= n:
            break
    return items


def fetch_ifeval(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    # IFEval ships a single `train` split on google/IFEval.
    ds = load_dataset("google/IFEval", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    items: list[dict] = []
    for idx in idxs:
        row = ds[int(idx)]
        instruction_ids = row.get("instruction_id_list") or []
        kwargs_list = row.get("kwargs") or []
        if not instruction_ids or not kwargs_list:
            continue
        chosen = None
        for inst_id, kw in zip(instruction_ids, kwargs_list):
            if inst_id in IFEVAL_SUPPORTED_CONSTRAINTS:
                chosen = (inst_id, kw or {})
                break
        if chosen is None:
            continue
        inst_id, kw = chosen
        params = {k: v for k, v in kw.items() if v is not None}
        items.append({
            "axis": "ifeval",
            "id": f"ifeval-{len(items)}",
            "prompt": row["prompt"],
            "constraint": {"kind": inst_id, "params": params},
        })
        if len(items) >= n:
            break
    return items


def fetch_humaneval(n: int, seed: int, max_loc: int = 20) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("openai/openai_humaneval", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    items: list[dict] = []
    for idx in idxs:
        row = ds[int(idx)]
        solution = row.get("canonical_solution") or ""
        if _count_code_lines(solution) > max_loc:
            continue
        test_block = row["test"]
        entry_point = row["entry_point"]
        single_assert = _first_assert(test_block, entry_point)
        if single_assert is None:
            single_assert = test_block
        items.append({
            "axis": "humaneval",
            "id": f"humaneval-{row['task_id'].replace('/', '-')}",
            "prompt": row["prompt"],
            "test": single_assert,
            "entry_point": entry_point,
        })
        if len(items) >= n:
            break
    return items


def _first_assert(test_block: str, entry_point: str) -> str | None:
    """Pull the first `assert candidate(...) == ...` style line.

    Canonical HumanEval test harnesses wrap asserts in a `check(candidate)`
    function; substituting the entry point lets the scorer call it
    directly without setting up the `check` indirection.
    """
    for raw in test_block.splitlines():
        line = raw.strip()
        if line.startswith("assert ") and "candidate" in line:
            return line.replace("candidate", entry_point)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mmlu-n", type=int, default=50)
    p.add_argument("--gsm8k-n", type=int, default=50)
    p.add_argument("--ifeval-n", type=int, default=50)
    p.add_argument("--humaneval-n", type=int, default=50)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent / "samples",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("mmlu_stem.jsonl", fetch_mmlu_stem, args.mmlu_n),
        ("gsm8k.jsonl", fetch_gsm8k, args.gsm8k_n),
        ("ifeval.jsonl", fetch_ifeval, args.ifeval_n),
        ("humaneval.jsonl", fetch_humaneval, args.humaneval_n),
    ]

    totals: dict[str, int] = {}
    for filename, fn, n in tasks:
        print(f"[subliminal-probe] fetching {filename} (n={n}) ...", flush=True)
        items = fn(n, args.seed)
        if len(items) < n:
            print(
                f"  WARNING: got {len(items)} items for {filename}; "
                f"dataset may have fewer usable rows than requested.",
                file=sys.stderr,
            )
        written = _write_jsonl(out_dir / filename, items)
        totals[filename] = written
        print(f"  wrote {written} items -> {out_dir / filename}", flush=True)

    print()
    print("[subliminal-probe] done:")
    for k, v in totals.items():
        print(f"  {k:20s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
