#!/usr/bin/env python3
"""Generate a synthetic SFT corpus by driving a served model as agent.

For each (seed, persona) pair, run an agent loop:

    user (from seed) → agent tool calls → tools execute against
    chinook → agent response → optional persona-driven follow-up →
    final agent response.

Each completed session gets written as one line to the output JSONL.
The shape mirrors the chat-template format the SFT stage consumes:

    {
      "id": "...",
      "persona": "analyst",
      "seed_id": "seed-003",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "", "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "name": "...",
         "content": "<json result>"},
        {"role": "assistant", "content": "..."}
      ]
    }

## Why this design

This generator points at any OpenAI-compatible serving endpoint (the
public `*.modal.run` URL printed by `modal deploy` for one of the serve
scripts; a high-concurrency teacher endpoint such as the
`gemma4-26b-concurrent` deployment is the cost-appropriate target). A
high-concurrency teacher endpoint typically runs with the server-side
tool-call parser DISABLED (see the serve stage's notes on vLLM #39392):
responses come back with raw `<|tool_call>...<tool_call|>` tokens in the
content, and we extract them with `_common.gemma4_parser`.

The same model serves both "user" (when generating a follow-up) and
"agent" (when responding to user turns). Two passes through the same
weights with different system prompts is cheaper than running a
separate user-persona model and gives essentially the same diversity
once you sweep across 5-10 personas + 50-100 seeds.

Concurrency is bounded by an `asyncio.Semaphore`; each session runs
in its own coroutine and the executor is sync (sqlite3 is fast and
cheap relative to model latency, so threading it is overkill).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# This file lives at <repo>/pipeline/corpus/generate_corpus.py. Its
# parent.parent is <repo>/pipeline, which is the import root that holds
# the `_common` package and the sibling `eval` stage. Putting it on
# sys.path lets `from _common.<module> import ...` resolve the same way
# it does inside a Modal container (where the serve scripts add the
# package via `add_local_python_source("_common")`).
_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from _common.gemma4_parser import parse_model_output  # noqa: E402

# The 15-tool chinook manifest is defined once in the eval stage and
# reused here so the corpus and the eval rubric share an identical tool
# surface. Add the eval stage dir to the path and import it; if the eval
# stage hasn't been set up yet, fail with a pointer rather than an
# opaque ImportError.
sys.path.insert(0, str(_PIPELINE / "eval"))
try:
    from tool_manifest import MANIFEST as CHINOOK_TOOLS  # noqa: E402
except ImportError as _e:  # pragma: no cover - environment guard
    raise ImportError(
        "could not import the chinook tool manifest from the eval stage "
        f"({_PIPELINE / 'eval' / 'tool_manifest.py'}). The corpus generator "
        "reuses the eval stage's tool_manifest.MANIFEST so the two stages "
        "share one tool surface."
    ) from _e

import chinook_tools  # noqa: E402  (sibling import inside corpus/)


DEFAULT_MAX_TOOL_TURNS = 6
"""Cap on tool-call rounds per agent turn. Stops a runaway loop where
the model keeps calling list_tables forever; in practice 6 is more
than enough for any chinook question."""

DEFAULT_USER_FOLLOWUPS = 1
"""How many additional user turns the persona injects after the
opening seed. Set to 0 for one-shot sessions."""


# ─────────────────────────────────────────────────────────────────────
# Persona + seed loading
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Persona:
    name: str
    system_prompt: str
    follow_up_style: str
    temperature: float


def load_personas(persona_dir: Path) -> list[Persona]:
    out: list[Persona] = []
    for p in sorted(persona_dir.glob("*.json")):
        data = json.loads(p.read_text())
        out.append(Persona(
            name=data["name"],
            system_prompt=data["system_prompt"],
            follow_up_style=data.get("follow_up_style", ""),
            temperature=float(data.get("temperature", 0.7)),
        ))
    if not out:
        raise FileNotFoundError(f"no *.json personas in {persona_dir}")
    return out


def load_seeds(path: Path) -> list[dict]:
    seeds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seeds.append(json.loads(line))
    if not seeds:
        raise ValueError(f"no seeds in {path}")
    return seeds


# ─────────────────────────────────────────────────────────────────────
# Model client (OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────


AGENT_SYSTEM_PROMPT = (
    "You are a SQL analyst with access to a chinook (digital music store) "
    "SQLite database via tools. Use the most specific tool for the question; "
    "only fall back to run_query when nothing else fits. Destructive tools "
    "(delete_record, drop_table, truncate_table) require operator_confirmed=true; "
    "if the user has not provided confirmation, refuse and explain why."
)


async def call_agent(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    enable_thinking: bool = True,
    api_key: str | None = None,
) -> str:
    """One round-trip to the served endpoint. Returns raw assistant
    content (still containing `<|tool_call>...` tokens — we parse them
    client-side)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": messages,
        "tools": CHINOOK_TOOLS,
        "temperature": temperature,
        "max_tokens": 1024,
        # Per-request thinking toggle. Keep enable_thinking=true: when
        # the server combines `--reasoning-parser gemma4` with
        # enable_thinking=false AND a response_format constraint it can
        # silently skip xgrammar (vLLM #39130). Per-request
        # chat_template_kwargs always wins over the server-side default.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    resp = await client.post(url, headers=headers, json=body, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


async def call_user_persona(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    persona: Persona,
    transcript: list[dict],
    *,
    api_key: str | None = None,
) -> str:
    """Have the same model produce a follow-up user message in the
    persona's voice. We strip the agent's tool calls from the
    transcript before showing it to the persona — it should react to
    the agent's *answer*, not its tool plumbing."""
    visible = [m for m in transcript
               if m["role"] in ("user", "assistant")
               and not m.get("tool_calls")]
    user_msgs = [{
        "role": "system",
        "content": (
            persona.system_prompt
            + "\n\nWrite your next message as the user. "
            + persona.follow_up_style
            + "\nReply with ONLY the user's next message — no quoting, no preamble."
        ),
    }, {
        "role": "user",
        "content": "Conversation so far:\n\n"
                   + "\n\n".join(
                       f"{m['role'].upper()}: {m.get('content', '')}"
                       for m in visible
                   ),
    }]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": user_msgs,
        "temperature": persona.temperature,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    resp = await client.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────


async def run_session(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    persona: Persona,
    seed: dict,
    *,
    chinook_db: str,
    max_tool_turns: int,
    user_followups: int,
    api_key: str | None = None,
) -> dict | None:
    """Drive one full session and return the recorded message log.

    Returns None on irrecoverable error (network, malformed response).
    Per-tool errors are turned into tool-response messages and the
    loop continues — that's the *correct* training signal for "what
    does the model do when a tool returns an error."
    """
    session_id = str(uuid.uuid4())
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": seed["user_prompt"]},
    ]

    user_turns_remaining = user_followups

    try:
        while True:
            for _ in range(max_tool_turns):
                raw = await call_agent(
                    client, endpoint, model, messages,
                    temperature=persona.temperature,
                    api_key=api_key,
                )
                tool_names, tool_args, text = parse_model_output(raw)

                if not tool_names:
                    # Final assistant message for this turn.
                    messages.append({"role": "assistant", "content": text})
                    break

                # Tool-calling assistant turn — record as OAI-shaped
                # tool_calls plus an empty content string.
                tool_calls = []
                tool_results = []
                for name, args in tool_args:
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    })
                    result = chinook_tools.execute(
                        name, args, db_path=chinook_db,
                    )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result,
                    })
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                })
                messages.extend(tool_results)
            else:
                # max_tool_turns hit without a non-tool reply — give the
                # model one last "wrap up" pass so the recorded session
                # ends with a real assistant message.
                raw = await call_agent(
                    client, endpoint, model, messages,
                    temperature=persona.temperature,
                    api_key=api_key,
                )
                _, _, text = parse_model_output(raw)
                messages.append({"role": "assistant", "content": text or ""})

            if user_turns_remaining <= 0:
                break

            follow_up = await call_user_persona(
                client, endpoint, model, persona, messages,
                api_key=api_key,
            )
            if not follow_up:
                break
            messages.append({"role": "user", "content": follow_up})
            user_turns_remaining -= 1
    except httpx.HTTPError as e:
        return {
            "id": session_id,
            "error": f"http: {type(e).__name__}: {e}",
            "persona": persona.name,
            "seed_id": seed["id"],
            "partial_messages": messages,
        }

    return {
        "id": session_id,
        "persona": persona.name,
        "seed_id": seed["id"],
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


async def generate(args: argparse.Namespace) -> None:
    persona_dir = Path(args.persona_dir)
    seeds_path = Path(args.seeds)
    out_path = Path(args.out)

    personas = load_personas(persona_dir)
    seeds = load_seeds(seeds_path)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    rng = random.Random(args.seed)
    pairings = [
        (rng.choice(personas), rng.choice(seeds))
        for _ in range(args.num_sessions)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    written = 0
    failed = 0
    started = time.time()

    async with httpx.AsyncClient() as client:
        async def _one(persona: Persona, seed: dict) -> dict | None:
            async with sem:
                return await run_session(
                    client, args.endpoint, args.model, persona, seed,
                    chinook_db=args.chinook,
                    max_tool_turns=args.max_tool_turns,
                    user_followups=args.user_followups,
                    api_key=api_key,
                )

        tasks = [asyncio.create_task(_one(p, s)) for p, s in pairings]
        with out_path.open("w") as fout:
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                if rec is None:
                    failed += 1
                    continue
                if "error" in rec:
                    failed += 1
                    print(f"[gen] session error: {rec['error']}", file=sys.stderr)
                    continue
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                written += 1
                if written % 5 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[gen] {written}/{args.num_sessions} "
                        f"sessions written ({failed} failed) "
                        f"in {elapsed:.0f}s",
                        flush=True,
                    )

    elapsed = time.time() - started
    print(
        f"[gen] DONE — {written} sessions written, {failed} failed "
        f"in {elapsed:.0f}s",
        flush=True,
    )


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True,
                   help="Base URL of the served endpoint (no /v1 suffix). "
                        "Use the public *.modal.run URL printed by `modal deploy`.")
    p.add_argument("--model", required=True,
                   help="Model name to send in the request body — the "
                        "served-model-name the endpoint advertises.")
    p.add_argument("--persona-dir", default="corpus/personas",
                   help="Directory of persona JSON files.")
    p.add_argument("--seeds", default="corpus/seeds/v1.jsonl",
                   help="Seeds JSONL path.")
    p.add_argument("--chinook", default="data/chinook.db",
                   help="Path to the chinook SQLite database.")
    p.add_argument("--out", required=True,
                   help="Output corpus JSONL path.")
    p.add_argument("--num-sessions", type=int, default=20,
                   help="Total number of sessions to generate.")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max in-flight sessions.")
    p.add_argument("--max-tool-turns", type=int, default=DEFAULT_MAX_TOOL_TURNS,
                   help="Max tool-call rounds per agent turn.")
    p.add_argument("--user-followups", type=int, default=DEFAULT_USER_FOLLOWUPS,
                   help="Number of persona-driven follow-up user messages per session.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for persona/seed pairing.")
    # Optional bearer token for a gated endpoint. Auth is your choice and
    # is OFF unless this env var is set — endpoints are public by default
    # and can instead be locked down at the ingress with proxy auth
    # (see modal.com/docs/guide/webhook-proxy-auth). The generator does
    # not require or implement auth.
    p.add_argument("--api-key-env", default=None,
                   help="Env var holding a bearer token for the endpoint, "
                        "if you chose to gate it. Optional; leave unset for "
                        "a public endpoint.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(generate(_cli()))
