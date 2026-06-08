#!/usr/bin/env python3
"""Scrub thinking-format pollution before SFT.

Three transformations on assistant messages:

  1. Strip `<think>...</think>` text blocks from `content`.
  2. Rename `thinking` → `reasoning` where the message has tool_calls
     (Gemma 4's chat template renders `reasoning` / `reasoning_content`,
     NOT `thinking`, and only when the message has tool_calls after
     the last user turn).
  3. Drop `thinking` field from messages without tool_calls (it would
     never render under the Gemma 4 template anyway).

Also normalises `role="model"` → `role="assistant"` for HF chat
compatibility — some upstream training adapters emit "model" instead.

Writes a new file alongside the input and prints an audit report that
exits non-zero if residual pollution remains.

Run:

    python corpus/scrub_corpus.py \\
        corpus/corpus_v1.jsonl \\
        corpus/corpus_v1.scrubbed.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
CHANNEL_BLOCK_RE = re.compile(r"<\|channel>thought.*?<channel\|>\s*", re.DOTALL)


def scrub_message(msg: dict, stats: dict) -> dict:
    if msg.get("role") == "model":
        msg["role"] = "assistant"
        stats["role_model_renamed"] += 1

    if msg.get("role") != "assistant":
        return msg

    stats["asst_messages"] += 1

    content = msg.get("content")
    if isinstance(content, str):
        original = content
        new_content = THINK_BLOCK_RE.sub("", content)
        new_content = CHANNEL_BLOCK_RE.sub("", new_content)
        if new_content != original:
            stats["content_stripped"] += 1
            msg["content"] = new_content.strip()

    if "thinking" in msg:
        thinking = msg.pop("thinking")
        has_tc = bool(msg.get("tool_calls"))
        if has_tc and thinking:
            msg["reasoning"] = thinking
            stats["thinking_renamed_to_reasoning"] += 1
        else:
            stats["thinking_dropped_no_tc"] += 1

    return msg


def scrub_file(in_path: Path, out_path: Path) -> dict:
    stats = {
        "total_sessions": 0,
        "asst_messages": 0,
        "role_model_renamed": 0,
        "content_stripped": 0,
        "thinking_renamed_to_reasoning": 0,
        "thinking_dropped_no_tc": 0,
    }
    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stats["total_sessions"] += 1
            messages = obj.get("messages", [])
            obj["messages"] = [scrub_message(m, stats) for m in messages]
            f_out.write(json.dumps(obj) + "\n")
    return stats


def audit(path: Path) -> dict:
    s = {
        "total_sessions": 0,
        "asst_messages": 0,
        "residual_think_tags": 0,
        "residual_channel_tags": 0,
        "residual_thinking_field": 0,
        "reasoning_field_present": 0,
        "role_model_remaining": 0,
    }
    think_check = re.compile(r"<think>", re.IGNORECASE)
    channel_check = re.compile(r"<\|channel>thought")
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            s["total_sessions"] += 1
            for m in obj.get("messages", []):
                if m.get("role") == "model":
                    s["role_model_remaining"] += 1
                if m.get("role") != "assistant":
                    continue
                s["asst_messages"] += 1
                content = m.get("content") or ""
                if think_check.search(content):
                    s["residual_think_tags"] += 1
                if channel_check.search(content):
                    s["residual_channel_tags"] += 1
                if "thinking" in m:
                    s["residual_thinking_field"] += 1
                if "reasoning" in m:
                    s["reasoning_field_present"] += 1
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.output.exists():
        print(
            f"ERROR: output already exists (refuse to overwrite): {args.output}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n=== Scrubbing {args.input.name} → {args.output.name} ===")
    scrub_stats = scrub_file(args.input, args.output)
    print("Transforms:")
    for k, v in scrub_stats.items():
        print(f"  {k:35s} {v}")

    print("\nPost-scrub audit:")
    audit_stats = audit(args.output)
    for k, v in audit_stats.items():
        print(f"  {k:35s} {v}")

    residual = (
        audit_stats["residual_think_tags"]
        + audit_stats["residual_channel_tags"]
        + audit_stats["residual_thinking_field"]
        + audit_stats["role_model_remaining"]
    )
    if residual == 0:
        print("\n  OK — zero pollution remaining.")
    else:
        print(f"\n  WARN — {residual} residual items. Inspect output.")
        sys.exit(2)


if __name__ == "__main__":
    main()
