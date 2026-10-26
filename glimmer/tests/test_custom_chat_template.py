"""Conformance tests for `custom_chat_template_muse_glimmer.jinja`.

Contracts:

1. **Byte-identity to upstream** whenever the request does not set
   ``reasoning_effort`` — across plain chat, system prompts, tools,
   multi-turn tool loops with reasoning history, and image parts. This is
   the prefix-cache invariant: a patch must not leak into a path it
   doesn't own.
2. **[G1] reasoning_effort fallback** — the OpenAI-style field maps onto
   Glimmer's reasoning strength, an explicit ``reasoning_strength`` still
   wins, and a directive already in the system prompt is not duplicated.
3. **Upstream behaviour the server relies on** — string tool arguments
   raise in the template (SGLang parses them to dicts before rendering),
   and tool results resolve their tool name from ``tool_call_id``.

Renders locally with jinja2's sandbox (the same engine HF transformers uses
for ``apply_chat_template``). No GPU, no SGLang.

    uv run --group dev pytest tests/ -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "chat_templates"
UPSTREAM_PATH = TEMPLATE_DIR / "muse_glimmer_upstream.jinja"
FORK_PATH = TEMPLATE_DIR / "custom_chat_template_muse_glimmer.jinja"

# sha256 of chat_template.jinja at meta-models/Muse-Glimmer-30B
# @ cec94fcfd6512f2cfe708138c32533b346ea4f69. Guards against hand edits to
# the vendored upstream copy; re-vendor and update this when bumping.
UPSTREAM_SHA256 = "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678"

BOS = "<|begin_of_text|>"


def _env() -> jinja2.Environment:
    # Mirrors transformers' chat-template environment.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg: str) -> None:
        raise jinja2.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


@pytest.fixture(scope="module")
def upstream() -> jinja2.Template:
    return _env().from_string(UPSTREAM_PATH.read_text())


@pytest.fixture(scope="module")
def fork() -> jinja2.Template:
    return _env().from_string(FORK_PATH.read_text())


def render(tpl: jinja2.Template, messages, **kw) -> str:
    kw.setdefault("add_generation_prompt", True)
    kw.setdefault("bos_token", BOS)
    # Fixed date so renders are deterministic (strftime_now is not defined).
    kw.setdefault("current_date", "2026-10-26")
    return tpl.render(messages=messages, **kw)


# ─────────────────────────── fixtures: conversations ───────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "env": {"type": "object"},
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_LOOP = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Why does the build fail?"},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should look at the build log first.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": {"command": "make", "timeout": 60, "env": {"CI": True}},
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "error: missing header foo.h"},
    {"role": "assistant", "content": "The build is missing `foo.h`."},
    {"role": "user", "content": "Fix it."},
]

SCENARIOS = {
    "plain": [{"role": "user", "content": "hi"}],
    "system": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ],
    "system_with_directive": [
        {"role": "system", "content": "Be terse.\nReasoning strength: low"},
        {"role": "user", "content": "hi"},
    ],
    "tool_loop": TOOL_LOOP,
    "image_parts": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this image?"},
            ],
        }
    ],
}


# ─────────────────────────── 1. byte identity ───────────────────────────


def test_vendored_upstream_is_pinned():
    digest = hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest()
    assert digest == UPSTREAM_SHA256


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("with_tools", [False, True])
def test_identical_without_reasoning_effort(upstream, fork, name, with_tools):
    kw = {"tools": TOOLS} if with_tools else {}
    msgs = SCENARIOS[name]
    assert render(fork, msgs, **kw) == render(upstream, msgs, **kw)


@pytest.mark.parametrize("strength", ["low", "medium", "high", "xhigh"])
def test_identical_with_explicit_reasoning_strength(upstream, fork, strength):
    msgs = SCENARIOS["tool_loop"]
    a = render(fork, msgs, tools=TOOLS, reasoning_strength=strength)
    b = render(upstream, msgs, tools=TOOLS, reasoning_strength=strength)
    assert a == b
    assert f"Reasoning strength: {strength}." in a


def test_default_strength_is_high(fork):
    out = render(fork, SCENARIOS["plain"])
    assert "Reasoning strength: high." in out


# ─────────────────────────── 2. [G1] reasoning_effort ───────────────────────────


@pytest.mark.parametrize(
    "effort, expected",
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("minimal", "low"),
        ("none", "low"),
        ("max", "xhigh"),
        ("HIGH", "high"),
        ("bogus", "high"),
    ],
)
def test_reasoning_effort_maps_to_strength(fork, effort, expected):
    out = render(fork, SCENARIOS["system"], reasoning_effort=effort)
    assert out.count("Reasoning strength:") == 1
    assert f"Reasoning strength: {expected}." in out


def test_upstream_ignores_reasoning_effort(upstream):
    """Documents the gap G1 closes."""
    out = render(upstream, SCENARIOS["system"], reasoning_effort="low")
    assert "Reasoning strength: high." in out


def test_reasoning_strength_beats_reasoning_effort(fork):
    out = render(fork, SCENARIOS["plain"], reasoning_strength="xhigh", reasoning_effort="low")
    assert "Reasoning strength: xhigh." in out
    assert "Reasoning strength: low." not in out


def test_system_prompt_directive_not_duplicated(fork):
    out = render(fork, SCENARIOS["system_with_directive"], reasoning_effort="high")
    assert out.lower().count("reasoning strength:") == 1
    assert "Reasoning strength: low" in out


def test_effort_applies_to_default_system_block(fork):
    out = render(fork, SCENARIOS["plain"], reasoning_effort="medium")
    assert "You are a helpful AI assistant." in out
    assert "Reasoning strength: medium." in out


# ─────────────────────────── 3. upstream contracts ───────────────────────────


def test_string_tool_arguments_raise(fork):
    """SGLang (serving_chat.normalize_assistant_tool_call_arguments) parses
    JSON-string arguments to dicts before rendering, so this path is never
    hit through the server. Clients templating locally must do the same."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
            ],
        },
    ]
    with pytest.raises(jinja2.TemplateError, match="arguments to be a dict"):
        render(fork, msgs, tools=TOOLS)


def test_tool_loop_rendering(fork):
    out = render(fork, TOOL_LOOP, tools=TOOLS)
    # Reasoning history goes to the self channel.
    assert "<|start|>assistant to=self<|message|>I should look at the build log first.<|eom|>" in out
    # Tool call rendered as ATEM with typed values.
    assert '<atem:invoke name="run_command">' in out
    assert '<atem:parameter name="timeout">60</atem:parameter>' in out
    assert '<atem:parameter name="env">{"CI": true}</atem:parameter>' in out
    # Tool result name resolved from tool_call_id.
    assert '<|start|>tool run_command<|message|><tool_output name="run_command">' in out
    assert out.endswith("<|start|>assistant")
    assert '# Valid recipients: "self", "read_file.*", "run_command.*", "user".' in out
