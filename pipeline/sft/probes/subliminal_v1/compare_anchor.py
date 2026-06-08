#!/usr/bin/env python3
"""Compare a post-SFT subliminal probe result against the pre-SFT anchor.

Two-sided thresholds. Drops on capability axes catch SFT regression;
unexpectedly large rises catch SFT corpus contamination — the two
failure modes share a `|delta|` shape.

## Thresholds

- Composite (unweighted mean of per-axis accuracy): `|delta| <= 0.02`.
- Any single axis: `|delta| <= 0.03`.

## Exit code

`0` on pass, `1` on fail (any threshold exceeded). A failing
comparison should block the adapter publish pipeline.

## Usage

    uv run compare_anchor.py \\
        --anchor anchors/e4b-pre-sft.json \\
        --candidate /tmp/post-sft-candidate.json \\
        --out /tmp/subliminal-comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

COMPOSITE_THRESHOLD = 0.02
AXIS_THRESHOLD = 0.03


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare(anchor: dict, candidate: dict) -> dict:
    """Per-axis + composite comparison with pass/fail verdict."""
    anchor_axes = anchor.get("axes", {})
    candidate_axes = candidate.get("axes", {})

    # Union the axes so a missing side becomes visible rather than
    # silently dropped.
    axis_names = sorted(set(anchor_axes) | set(candidate_axes))

    per_axis: dict[str, dict] = {}
    any_axis_failed = False
    for axis in axis_names:
        a = anchor_axes.get(axis)
        c = candidate_axes.get(axis)
        if a is None or c is None:
            per_axis[axis] = {
                "anchor": float(a["accuracy"]) if a is not None else None,
                "candidate": float(c["accuracy"]) if c is not None else None,
                "delta": None,
                "passed": False,
                "reason": "axis missing from one side",
            }
            any_axis_failed = True
            continue

        a_acc = float(a["accuracy"])
        c_acc = float(c["accuracy"])
        # Round before subtraction so `0.62 - 0.59 = 0.03` compares equal
        # to the threshold instead of `0.030000000000000027`.
        delta = round(c_acc - a_acc, 4)
        passed = abs(delta) <= AXIS_THRESHOLD
        reason = None
        if not passed:
            direction = "rose" if delta > 0 else "dropped"
            reason = (
                f"axis {direction} {abs(delta):.4f} from anchor "
                f"(threshold: {AXIS_THRESHOLD:.2f})"
            )
            any_axis_failed = True

        per_axis[axis] = {
            "anchor": a_acc, "candidate": c_acc, "delta": delta,
            "passed": passed, "reason": reason,
        }

    anchor_composite = float(anchor.get("composite", 0.0))
    candidate_composite = float(candidate.get("composite", 0.0))
    composite_delta = round(candidate_composite - anchor_composite, 4)
    composite_passed = abs(composite_delta) <= COMPOSITE_THRESHOLD
    overall_passed = composite_passed and not any_axis_failed

    return {
        "anchor_label": anchor.get("label"),
        "candidate_label": candidate.get("label"),
        "anchor_model": anchor.get("model"),
        "candidate_model": candidate.get("model"),
        "composite": {
            "anchor": anchor_composite,
            "candidate": candidate_composite,
            "delta": composite_delta,
            "passed": composite_passed,
            "threshold": COMPOSITE_THRESHOLD,
        },
        "axes": per_axis,
        "axis_threshold": AXIS_THRESHOLD,
        "overall_passed": overall_passed,
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }


def format_report(report: dict) -> str:
    lines = [
        f"[subliminal-compare] anchor={report['anchor_label']!r} "
        f"candidate={report['candidate_label']!r}"
    ]

    comp = report["composite"]
    status = "PASS" if comp["passed"] else "FAIL"
    lines.append(
        f"  composite        anchor={comp['anchor']:.4f}  "
        f"candidate={comp['candidate']:.4f}  "
        f"delta={comp['delta']:+.4f}  [{status}]"
    )

    for axis, row in report["axes"].items():
        if row["anchor"] is None or row["candidate"] is None:
            lines.append(f"  {axis:20s} MISSING  ({row['reason']})")
            continue
        status = "PASS" if row["passed"] else "FAIL"
        extra = f"  ({row['reason']})" if row["reason"] else ""
        lines.append(
            f"  {axis:20s} anchor={row['anchor']:.4f}  "
            f"candidate={row['candidate']:.4f}  "
            f"delta={row['delta']:+.4f}  [{status}]{extra}"
        )

    lines.append("")
    lines.append(
        f"[subliminal-compare] overall = "
        f"{'PASS' if report['overall_passed'] else 'FAIL'}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor", required=True, type=Path,
        help="Path to the pre-SFT anchor JSON produced by score_probe.py.",
    )
    parser.add_argument(
        "--candidate", required=True, type=Path,
        help="Path to the post-SFT candidate JSON to compare against the anchor.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to write the full comparison report as JSON.",
    )
    args = parser.parse_args()

    anchor = load_report(args.anchor)
    candidate = load_report(args.candidate)
    report = compare(anchor, candidate)

    print(format_report(report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\n[subliminal-compare] wrote {args.out}")

    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
