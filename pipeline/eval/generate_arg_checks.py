#!/usr/bin/env python3
"""Auto-generate `expected_arg_checks` for a chinook eval JSONL.

Reads a scenario file (OAI-messages shape), scans each scenario's
first-assistant-turn `tool_calls`, and emits a parallel
`expected_arg_checks` list populated from the patterns it recognises
with high confidence:

  * chinook table names (`Album`, `Track`, `Invoice`, …) → `equals` on
    the matching arg, plus a `matches` regex that anchors any SQL
    `FROM <Table>` clause.
  * SQL query openers (`SELECT`, `WITH`, `INSERT`, …) → `matches`
    anchored to the first table label.
  * Boolean flags → `equals` on `"true"`/`"false"`.
  * Numeric values on identifier-shaped keys (`limit`, `count`,
    `offset`, `top_k`, `record_id`, …) → `equals` on the Display form.
  * Nested dicts → recurse with dotted paths.

Free-form prose (notes, queries-as-natural-language, descriptions)
and ambiguous numerics (durations, ratios, raw floats) are skipped —
those need author judgment.

Scenarios that already declare `expected_arg_checks` at the top level
are preserved verbatim (idempotent — re-running doesn't overwrite
author-curated checks). Pass `--force-regenerate` to overwrite.

This is a plain local script with no cloud dependency — run it against
the scenario file in this directory and re-upload the result to your
eval-data volume:

    # 1. Generate checks locally
    python eval/generate_arg_checks.py \\
        --in eval/scenarios/chinook_eval_v1.jsonl \\
        --out eval/scenarios/chinook_eval_v1.enriched.jsonl

    # 2. Upload the enriched file to the eval-data Modal volume
    modal volume put gemma4-eval-data \\
        eval/scenarios/chinook_eval_v1.enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# Pattern recognisers — chinook
# ─────────────────────────────────────────────────────────────────────

# The 11 canonical chinook table names. Case-sensitive — chinook ships
# with mixed-case table names and SQLite is case-sensitive on table
# identifiers when they're double-quoted (which our manifest implies).
CHINOOK_TABLES: set[str] = {
    "Album", "Artist", "Customer", "Employee", "Genre",
    "Invoice", "InvoiceLine", "MediaType",
    "Playlist", "PlaylistTrack", "Track",
}

# SQL opener keywords. Anchors generated regexes when we see something
# that looks like a query string.
SQL_OPENER_RE = re.compile(
    r"^\s*(WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|EXPLAIN)\b",
    re.IGNORECASE,
)

# Pull the first `FROM <Table>` shape out of a SQL query. Multi-table
# joins still match the first table — that's the strongest single
# anchor for a `matches` check.
SQL_FROM_RE = re.compile(r"\bFROM\s+\"?(?P<table>[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

# Identifier-shaped keys where exact-numeric equality is meaningful.
# Free-form numerics (track lengths, dollar amounts) are deliberately
# excluded — a 10% drift on those isn't worth flagging.
NUMERIC_KEYS_WORTH_CHECKING: set[str] = {
    "limit", "count", "offset", "top_k", "topk", "n",
    "record_id", "id", "artist_id", "album_id", "track_id",
    "customer_id", "employee_id", "invoice_id",
}

# Keys whose string values are free-form prose or NL queries — don't
# emit equality checks on them. Authors can add `contains` checks
# manually for specific phrases worth pinning.
FREE_FORM_KEYS: set[str] = {
    "note", "reason", "description", "comment",
    # `query` on the natural-language search tools is intentionally
    # NOT here — we DO want a `contains` check on it (see below).
}

# Tools whose `query` arg is natural-language (not SQL). We emit a
# `contains` check on the user's literal noun rather than a SQL regex.
NATURAL_LANGUAGE_QUERY_TOOLS: set[str] = {
    "find_artist", "find_album", "find_track",
}


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _is_chinook_table(value: str) -> bool:
    return value in CHINOOK_TABLES


def _sql_anchor_table(value: str) -> str | None:
    """If `value` is a SQL query, return the first FROM-table; else None."""
    if not SQL_OPENER_RE.match(value):
        return None
    m = SQL_FROM_RE.search(value)
    return m.group("table") if m else None


def _emit_check(
    tool: str, path: str, value: Any, key_name: str,
) -> dict | None:
    """Decide which check (if any) fits this (tool, path, value, key).

    The generator is deliberately conservative: returning `None` is
    common, and `None` means the semantic axis stays neutral on this
    arg. Better to under-emit than to fabricate spurious checks.
    """
    if isinstance(value, bool):
        return {"type": "equals", "tool": tool, "path": path, "value": _stringify(value)}

    if isinstance(value, (int, float)):
        if key_name in NUMERIC_KEYS_WORTH_CHECKING:
            return {"type": "equals", "tool": tool, "path": path, "value": _stringify(value)}
        return None

    if not isinstance(value, str):
        return None

    if not value:
        return None

    if key_name in FREE_FORM_KEYS:
        return None

    # Natural-language query tool: emit `contains` on the literal value.
    if tool in NATURAL_LANGUAGE_QUERY_TOOLS and key_name == "query":
        return {"type": "contains", "tool": tool, "path": path, "needle": value}

    # SQL query: anchor a `matches` to the first FROM-table.
    anchor = _sql_anchor_table(value)
    if anchor is not None:
        return {
            "type": "matches",
            "tool": tool,
            "path": path,
            "regex": rf"(?i)\bFROM\s+\"?{re.escape(anchor)}\b",
        }

    # Bare chinook table name on a typed `table_name` arg.
    if _is_chinook_table(value):
        return {"type": "equals", "tool": tool, "path": path, "value": value}

    return None


def _walk_args(tool: str, path_prefix: str, args: Any, out: list[dict]) -> None:
    """Recursively walk `args`, emitting checks the recognisers fit.

    Nested dicts get dotted paths (`filter.id`); arrays aren't traversed
    so paths stay stable across regenerations.
    """
    if isinstance(args, dict):
        for key, value in args.items():
            if not isinstance(key, str):
                continue
            dotted = f"{path_prefix}.{key}" if path_prefix else key
            if isinstance(value, dict):
                _walk_args(tool, dotted, value, out)
                continue
            check = _emit_check(tool, dotted, value, key)
            if check is not None:
                out.append(check)


def generate_for_scenario(scenario: dict) -> list[dict]:
    """Produce `expected_arg_checks` for one scenario. Empty when no
    first-assistant-turn tool calls exist."""
    messages = scenario.get("messages")
    if not isinstance(messages, list):
        return []

    past_first_user = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            if past_first_user:
                break
            past_first_user = True
            continue
        if role == "assistant" and past_first_user:
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                break
            checks: list[dict] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(args, dict):
                    continue
                _walk_args(name, "", args, checks)
            return checks
    return []


def enrich_scenario(scenario: dict, *, force: bool = False) -> dict:
    """Return `scenario` (or a copy) with `expected_arg_checks` filled.

    Idempotent unless `force=True`: if checks already exist on the
    scenario, the input is returned unchanged.
    """
    if not force and isinstance(scenario.get("expected_arg_checks"), list):
        return scenario
    checks = generate_for_scenario(scenario)
    if not checks and not force:
        return scenario
    out = dict(scenario)
    out["expected_arg_checks"] = checks
    return out


def enrich_jsonl(in_path: Path, out_path: Path, *, force: bool) -> dict:
    stats = {
        "scenarios_read": 0,
        "scenarios_enriched": 0,
        "scenarios_skipped_preserved": 0,
        "scenarios_skipped_no_checks": 0,
        "total_checks_emitted": 0,
    }
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["scenarios_read"] += 1
            scenario = json.loads(line)
            enriched = enrich_scenario(scenario, force=force)
            if enriched is scenario:
                if isinstance(scenario.get("expected_arg_checks"), list):
                    stats["scenarios_skipped_preserved"] += 1
                else:
                    stats["scenarios_skipped_no_checks"] += 1
                fout.write(json.dumps(scenario) + "\n")
                continue
            stats["scenarios_enriched"] += 1
            stats["total_checks_emitted"] += len(enriched.get("expected_arg_checks", []))
            fout.write(json.dumps(enriched) + "\n")
    return stats


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", dest="in_path", type=Path, required=True,
                        help="Input JSONL (OAI-messages shape).")
    parser.add_argument("--out", dest="out_path", type=Path, required=True,
                        help="Output JSONL with expected_arg_checks populated.")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Overwrite expected_arg_checks on scenarios that already carry them.")
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        print(f"[gen-checks] input file not found: {args.in_path}", file=sys.stderr)
        return 2
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = enrich_jsonl(args.in_path, args.out_path, force=args.force_regenerate)
    print(f"[gen-checks] read: {stats['scenarios_read']} scenarios")
    print(f"[gen-checks] enriched: {stats['scenarios_enriched']} "
          f"(+{stats['total_checks_emitted']} checks)")
    print(f"[gen-checks] preserved (already had checks): {stats['scenarios_skipped_preserved']}")
    print(f"[gen-checks] skipped (no pattern fit): {stats['scenarios_skipped_no_checks']}")
    print(f"[gen-checks] wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
