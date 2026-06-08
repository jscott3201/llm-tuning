"""Conformance tests for `custom_chat_template_gemma4.jinja`.

Two contracts:

1. **Byte-identity to upstream** when both new kwargs are explicitly
   disabled (`enable_thinking=False, preserve_thinking=False`). This is
   the prefix-cache invariant — any drift here means the patches
   leaked into a path they shouldn't have.

2. **Strict where upstream is silent**: string-typed `arguments`
   raises (P3), `None` values render as JSON `null` (P1), consecutive
   text-only assistants merge symmetrically (P5).

Run with: `python3 -m pytest tests/test_custom_chat_template.py -v`
"""
from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

_HERE = Path(__file__).resolve().parent
UPSTREAM_PATH = _HERE / ".." / "chat_templates" / "gemma4_upstream.jinja"
CUSTOM_PATH = _HERE / ".." / "chat_templates" / "custom_pub_chat_template_gemma4.jinja"


# ─────────────────────────── fixtures ───────────────────────────


def _make_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        keep_trailing_newline=False,
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )

    def raise_exception(msg: str) -> None:
        raise jinja2.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


@pytest.fixture(scope="module")
def upstream() -> jinja2.Template:
    return _make_env().from_string(UPSTREAM_PATH.read_text())


@pytest.fixture(scope="module")
def custom() -> jinja2.Template:
    return _make_env().from_string(CUSTOM_PATH.read_text())


def _render(tpl: jinja2.Template, **kwargs) -> str:
    """Render with sensible defaults; caller overrides as needed."""
    kwargs.setdefault("bos_token", "<bos>")
    kwargs.setdefault("add_generation_prompt", True)
    return tpl.render(**kwargs)


# Shared tool spec used across several tests.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "default": 1},
                },
                "required": ["city"],
            },
        },
    }
]


# ────────── group 1: byte-identity with new kwargs disabled ──────────


def test_T0_bare_user_matches_upstream(upstream, custom):
    msgs = [{"role": "user", "content": "hi"}]
    u = _render(upstream, messages=msgs, enable_thinking=False)
    c = _render(
        custom,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T2_system_plus_user_matches_upstream(upstream, custom):
    msgs = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]
    u = _render(upstream, messages=msgs, enable_thinking=False)
    c = _render(
        custom,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T6_tools_plus_thinking_off_matches_upstream(upstream, custom):
    msgs = [{"role": "user", "content": "Tokyo?"}]
    u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=False)
    c = _render(
        custom,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T7_full_tool_roundtrip_matches_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "weather in Tokyo?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo, JP", "days": 3},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"temp_c": 22, "condition": "clear"}',
        },
        {"role": "assistant", "content": "It's 22°C and clear in Tokyo."},
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=False,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    assert u == c


def test_T8_reasoning_current_turn_matches_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "reasoning_content": "I should call the get_weather tool.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        add_generation_prompt=False,
    )
    c = _render(
        custom,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    assert u == c
    assert "<|channel>thought\nI should call the get_weather tool." in c


def test_T9b_preserve_thinking_false_recovers_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "reasoning_content": "Calling the weather tool now.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 18}'},
        {"role": "user", "content": "and Berlin?"},
    ]
    u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=True)
    c = _render(
        custom,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
    )
    assert u == c


def test_T10_developer_role_accepted_in_both(upstream, custom):
    msgs = [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    u = _render(upstream, messages=msgs, enable_thinking=False)
    c = _render(
        custom,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c
    assert "<|turn>system\nBe concise." in c


def test_T15_empty_args_dict_matches_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "ping"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c",
                    "function": {"name": "ping", "arguments": {}},
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    c = _render(
        custom,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T16_booleans_and_integers_match_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "f",
                        "arguments": {
                            "flag": True,
                            "count": 42,
                            "off": False,
                            "name": "alice",
                        },
                    }
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    c = _render(
        custom,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T17_nested_objects_and_arrays_match_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "f",
                        "arguments": {
                            "loc": {"lat": 35.6, "lng": 139.7},
                            "tags": ["weather", "tokyo"],
                        },
                    }
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    c = _render(
        custom,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T18_multimodal_user_content_matches_upstream(upstream, custom):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image"},
            ],
        }
    ]
    u = _render(upstream, messages=msgs, enable_thinking=False)
    c = _render(
        custom,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T19_multi_tool_fanout_matches_upstream(upstream, custom):
    msgs = [
        {"role": "user", "content": "do two things"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "tool_a", "arguments": {"x": 1}},
                },
                {
                    "id": "c2",
                    "function": {"name": "tool_b", "arguments": {"y": "hello"}},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "A"},
        {"role": "tool", "tool_call_id": "c2", "content": "B"},
        {"role": "assistant", "content": "Done both."},
    ]
    u = _render(
        upstream,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    c = _render(
        custom,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


def test_T4_args_None_equals_empty_dict_in_upstream(upstream, custom):
    """`arguments=None` is one of the cases P3 explicitly allows
    (the only non-mapping form the harness will not be punished for)."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "fn", "arguments": None}}
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    c = _render(
        custom,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert u == c


# ────────── group 2: strict-where-upstream-silent contracts ──────────


def test_T3_custom_raises_on_string_arguments(custom):
    """P3: upstream silently nests braces; custom raises so the
    contract violation surfaces at the SGLang render step instead of
    silently corrupting the prompt."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "fn", "arguments": "stringy"}}
            ],
        },
    ]
    with pytest.raises(jinja2.TemplateError) as ei:
        _render(custom, messages=msgs, add_generation_prompt=False)
    assert "must be a JSON object" in str(ei.value)


def test_T5_None_in_args_renders_as_json_null(upstream, custom):
    """P1: upstream emits the bare string 'None'; custom emits 'null'."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "fn",
                        "arguments": {"a": 1, "b": None},
                    }
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    c = _render(
        custom,
        messages=msgs,
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert "b:None" in u
    assert "b:null" in c
    assert "b:None" not in c


def test_T9_preserve_thinking_retains_prior_reasoning(upstream, custom):
    """P4: with preserve_thinking=true the prior assistant turn's
    <|channel>thought block is re-emitted, even though it sits before
    the last user message (upstream's gate)."""
    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "reasoning_content": "Calling the weather tool now.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 18}'},
        {"role": "user", "content": "and Berlin?"},
    ]
    u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=True)
    c = _render(
        custom,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=True,
    )
    assert "Calling the weather tool now" not in u
    assert "Calling the weather tool now" in c


def test_T11_consecutive_text_assistants_merge_symmetrically(custom):
    """P5: HF discussion #62 fix. Upstream emits one open and two
    closes for two consecutive assistant messages (asymmetric);
    custom emits one open and one close, with the content joined by
    a single newline."""
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "part 1"},
        {"role": "assistant", "content": "part 2"},
    ]
    c = _render(
        custom,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    assert c.count("<|turn>model") == 1
    # 2 closes total: user-turn close + the single (merged) model-turn close.
    assert c.count("<turn|>") == 2
    assert "part 1\npart 2" in c


def test_T13_tool_chain_then_text_text_merges_only_trailing_pair(custom):
    """P5 narrowing: the tool-call+response chain MUST close normally
    so the model still sees a balanced turn frame around the
    <|tool_response> block. Only the trailing two text-only assistant
    messages should merge."""
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    c = _render(
        custom,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=False,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    assert "first\nsecond" in c
    assert c.endswith("<turn|>\n")
    # Tool chain still closes its own turn:
    assert "<|tool_response>response:get_weather{value:" in c


def test_T14_default_kwargs_enable_both_new_behaviours(custom):
    """With no explicit overrides, the custom template's new defaults
    should both fire: enable_thinking=true opens a system <|think|>
    block; preserve_thinking=true is set (verifiable indirectly by
    the existence of the system block on a no-system prompt)."""
    msgs = [{"role": "user", "content": "hi"}]
    c = _render(custom, messages=msgs)
    assert "<|think|>" in c
    assert "<|turn>system" in c


def test_T1_default_flip_diverges_from_upstream(upstream, custom):
    """Companion to T0: when the caller does NOT pass kwargs at all,
    the custom default flip means the custom output diverges from
    upstream — but only here. T0 confirms byte-identity when the
    caller explicitly opts out."""
    msgs = [{"role": "user", "content": "hi"}]
    u = _render(upstream, messages=msgs)
    c = _render(custom, messages=msgs)
    assert u != c
    # Specifically: custom opens a system block (thinking-on default).
    assert "<|turn>system" in c
    assert "<|turn>system" not in u
