#!/usr/bin/env python3
"""Compare a baseline `eval_results.json` against a post-SFT one.

Reads two JSON reports produced by the eval stage's scorer and prints a
per-axis diff with PASS/FAIL verdicts. Exit code is 0 when the
candidate improved (or held within `--tolerance`), non-zero when it
regressed.

The simplest "did SFT help?" gate. The subliminal probe under
`probes/subliminal_v1/` is the orthogonal capability-regression
guard.

Run:

    uv run sft/compare_eval.py \\
        --baseline /tmp/base-e4b-v1.json \\
        --candidate /tmp/sft-e4b-v1.json \\
        --tolerance 0.005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Mirrors the five scoring axes in `_common/eval_scoring.py` plus the
# weighted `overall`.
AXES = (
    "tool_selection",
    "argument_correctness",
    "semantic_arg_correctness",
    "sql_syntax",
    "safety",
    "overall",
)


def _fmt_delta(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:+.4f}"[1:] if delta == 0 else f"{sign}{delta:.4f}"


def compare(baseline: dict, candidate: dict, *, tolerance: float) -> dict:
    """Build a per-axis comparison record. The verdict is "PASS" if
    the candidate matched or beat the baseline within `tolerance` on
    every axis (the candidate's allowed to be slightly noisier)."""
    b = baseline.get("per_axis") or {}
    c = candidate.get("per_axis") or {}
    rows = []
    any_failed = False
    for axis in AXES:
        bv = b.get(axis)
        cv = c.get(axis)
        if bv is None or cv is None:
            rows.append({"axis": axis, "baseline": bv, "candidate": cv,
                         "delta": None, "passed": False,
                         "reason": "axis missing on one side"})
            any_failed = True
            continue
        delta = round(cv - bv, 4)
        # PASS rule: candidate >= baseline - tolerance. Allows tiny
        # regressions inside noise without flagging them.
        passed = delta >= -tolerance
        rows.append({
            "axis": axis,
            "baseline": bv,
            "candidate": cv,
            "delta": delta,
            "passed": passed,
            "reason": None if passed else f"regressed by {abs(delta):.4f}",
        })
        if not passed:
            any_failed = True
    return {
        "baseline_label": baseline.get("label"),
        "candidate_label": candidate.get("label"),
        "tolerance": tolerance,
        "rows": rows,
        "overall_passed": not any_failed,
    }


def render(report: dict) -> str:
    lines = [
        f"baseline:  {report['baseline_label']}",
        f"candidate: {report['candidate_label']}",
        f"tolerance: {report['tolerance']:.4f}",
        "",
        f"  {'axis':28s} {'baseline':>10s} {'candidate':>10s} {'delta':>10s}  verdict",
        f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}  -------",
    ]
    for row in report["rows"]:
        verdict = "PASS" if row["passed"] else "FAIL"
        bv = f"{row['baseline']:10.4f}" if row["baseline"] is not None else "       N/A"
        cv = f"{row['candidate']:10.4f}" if row["candidate"] is not None else "       N/A"
        if row["delta"] is None:
            dv = "       N/A"
        else:
            dv = f"{row['delta']:+10.4f}"
        extra = f"  ({row['reason']})" if row["reason"] else ""
        lines.append(f"  {row['axis']:28s} {bv} {cv} {dv}  [{verdict}]{extra}")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if report['overall_passed'] else 'FAIL'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Pre-SFT eval_results.json (base-model baseline).")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Post-SFT eval_results.json.")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Allowed per-axis regression before flagging FAIL.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path for the JSON comparison report.")
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    report = compare(baseline, candidate, tolerance=args.tolerance)
    print(render(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
