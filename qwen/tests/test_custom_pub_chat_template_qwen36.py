"""Conformance tests for `custom_pub_chat_template_qwen36.jinja`.

This is the public-facing fork of the Qwen/Qwen3.6-27B chat template,
intended for open-source agentic coding harnesses (opencode, pi, openclaw,
etc.).

Three contracts are exercised:

1. **Byte-identity to upstream** when all three toggle-gated patches
   (Q1 preserve_thinking, Q5 unwrap_tool_envelope, Q6
   verbose_tool_instructions) are explicitly disabled, AND the input
   does not exercise the strict-improvement patches (Q2 developer role,
   Q3 string args, Q4 thinking variants). This is the prefix-cache
   invariant — any drift here means a patch leaked into a path it
   shouldn't have.

2. **Strict-where-upstream-silent**: developer role accepted (Q2),
   string-typed arguments raise debuggably (Q3), </thinking>/whitespace
   variants extract reasoning (Q4), unclosed-think+tool_call rescue (Q4),
   plain </think> byte-identity even for pathological bodies (Q4/T18),
   mid-conversation system/developer rendered inline not crashed (Q7),
   single-mapping tool_calls normalized not dropped (Q8).

3. **Agentic-coding scenario tests** (Sxx) reproducing the exact bug
   shapes documented in:
     - https://github.com/earendil-works/pi/issues/3325
     - https://github.com/anomalyco/opencode/issues/24264 (S04 only)
     - https://gist.github.com/sudoingX/c2facf7d8f7608c65c1024ef3b22d431

Run with: `python3 -m pytest tests/test_custom_pub_chat_template_qwen36.py -v`
Requires: jinja2, pytest
"""
from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "chat_templates"
UPSTREAM_PATH = TEMPLATE_DIR / "qwen36_upstream.jinja"
CUSTOM_PUB_PATH = TEMPLATE_DIR / "custom_pub_chat_template_qwen36.jinja"


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
def custom_pub() -> jinja2.Template:
    return _make_env().from_string(CUSTOM_PUB_PATH.read_text())


def _render(tpl: jinja2.Template, **kwargs) -> str:
    """Render with sensible defaults."""
    kwargs.setdefault("add_generation_prompt", True)
    return tpl.render(**kwargs)


# To force the public fork to match upstream exactly, pass these kwargs.
# Three toggles plus enable_thinking=true (the upstream default behavior
# for an undefined kwarg).
RECOVER_UPSTREAM = dict(
    enable_thinking=True,
    preserve_thinking=False,
    unwrap_tool_envelope=False,
    verbose_tool_instructions=False,
)


# Tool spec modelled after a coding agent's typical tools — find_files +
# read_file (the canonical opencode/pi shape).
CODING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "dir": {"type": "string", "default": "."},
                    "language": {"type": "string", "nullable": True},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "nullable": True},
                    "end_line": {"type": "integer", "nullable": True},
                },
                "required": ["path"],
            },
        },
    },
]

TOOLS = [CODING_TOOLS[0]]


# ════════════════════════════════════════════════════════════════════
#  Group 1: byte-identity with all toggle-gated patches disabled
# ════════════════════════════════════════════════════════════════════


def test_T0_bare_user_matches_upstream(upstream, custom_pub):
    msgs = [{"role": "user", "content": "hi"}]
    u = _render(upstream, messages=msgs, enable_thinking=True, preserve_thinking=False)
    c = _render(custom_pub, messages=msgs, **RECOVER_UPSTREAM)
    assert u == c


def test_T1_system_plus_user_matches_upstream(upstream, custom_pub):
    msgs = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]
    u = _render(upstream, messages=msgs, enable_thinking=True, preserve_thinking=False)
    c = _render(custom_pub, messages=msgs, **RECOVER_UPSTREAM)
    assert u == c


def test_T2_tools_plus_user_matches_upstream(upstream, custom_pub):
    """The envelope-unwrap toggle (Q5) must be off to claim byte-identity
    on the tools block."""
    msgs = [{"role": "user", "content": "find python files"}]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
    )
    c = _render(custom_pub, messages=msgs, tools=TOOLS, **RECOVER_UPSTREAM)
    assert u == c


def test_T3_full_tool_roundtrip_matches_upstream(upstream, custom_pub):
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.py", "dir": "src"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '["src/main.py", "src/utils.py"]',
        },
        {"role": "assistant", "content": "Found 2 Python files."},
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        **RECOVER_UPSTREAM,
        add_generation_prompt=False,
    )
    assert u == c


def test_T4_reasoning_current_turn_matches_upstream(upstream, custom_pub):
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "reasoning_content": "I should call find_files with pattern '*.py'.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.py"},
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
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        **RECOVER_UPSTREAM,
        add_generation_prompt=False,
    )
    assert u == c
    assert "<think>\nI should call find_files with pattern '*.py'." in c


def test_T5_preserve_thinking_true_byte_identity(upstream, custom_pub):
    """With preserve_thinking=true explicitly on both sides, prior <think>
    blocks survive and the templates produce byte-identical output."""
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "reasoning_content": "thinking A",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.py"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "user", "content": "now look for *.ts"},
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=True,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=True,
        unwrap_tool_envelope=False,
        verbose_tool_instructions=False,
    )
    assert u == c
    assert "thinking A" in c


def test_T6_enable_thinking_false_matches_upstream(upstream, custom_pub):
    """enable_thinking=false produces the pre-filled empty <think></think>
    sequence in the generation prompt. Byte-identical to upstream."""
    msgs = [{"role": "user", "content": "hi"}]
    u = _render(
        upstream,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        enable_thinking=False,
        preserve_thinking=False,
        unwrap_tool_envelope=False,
        verbose_tool_instructions=False,
    )
    assert u == c
    assert "<think>\n\n</think>\n\n" in c


def test_T7_multi_tool_fanout_matches_upstream(upstream, custom_pub):
    msgs = [
        {"role": "user", "content": "find py and ts files"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.py"},
                    },
                },
                {
                    "id": "c2",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.ts"},
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[a.py]"},
        {"role": "tool", "tool_call_id": "c2", "content": "[b.ts]"},
        {"role": "assistant", "content": "Found both."},
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        **RECOVER_UPSTREAM,
        add_generation_prompt=False,
    )
    assert u == c


def test_T8_booleans_nones_ints_match_upstream(upstream, custom_pub):
    """Upstream's `tojson | safe` already correctly emits JSON keywords
    `true`/`false`/`null` for booleans and None. The public fork
    deliberately keeps the same code path. Lock it in."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "find_files",
                        "arguments": {
                            "recursive": True,
                            "case_insensitive": False,
                            "language": None,
                            "depth": 3,
                        },
                    }
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        **RECOVER_UPSTREAM,
        add_generation_prompt=False,
    )
    assert u == c
    # JSON keywords used everywhere — no Python `True`/`False`/`None`.
    assert "\ntrue\n" in c and "\nfalse\n" in c and "\nnull\n" in c
    assert "\nTrue\n" not in c and "\nFalse\n" not in c and "\nNone\n" not in c


def test_T9_nested_objects_and_arrays_match_upstream(upstream, custom_pub):
    msgs = [
        {"role": "user", "content": "complex"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "find_files",
                        "arguments": {
                            "filters": {"language": "py", "min_size": 100},
                            "extensions": [".py", ".pyi"],
                        },
                    }
                }
            ],
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        **RECOVER_UPSTREAM,
        add_generation_prompt=False,
    )
    assert u == c


def test_T10_multimodal_user_content_matches_upstream(upstream, custom_pub):
    """Image content parts render identically."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this screenshot?"},
                {"type": "image"},
            ],
        }
    ]
    u = _render(upstream, messages=msgs, enable_thinking=True, preserve_thinking=False)
    c = _render(custom_pub, messages=msgs, **RECOVER_UPSTREAM)
    assert u == c


def test_T11_default_kwargs_diverge_from_upstream(upstream, custom_pub):
    """Companion to T2: with no kwargs at all, public fork's new defaults
    (preserve_thinking=true, unwrap_tool_envelope=true,
    verbose_tool_instructions=true) cause divergence from upstream. This
    is intentional — the default flip IS the patch."""
    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "reasoning_content": "step one",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.py"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "user", "content": "q2"},
    ]
    u = _render(upstream, messages=msgs, tools=TOOLS)
    c = _render(custom_pub, messages=msgs, tools=TOOLS)
    # Diverges in at least three ways: prior reasoning retained, envelope
    # unwrapped, IMPORTANT block has additional bullets.
    assert u != c
    assert "step one" in c  # preserve_thinking=true default
    assert "step one" not in u
    assert "Do NOT omit the opening <tool_call>" in c  # Q6 verbose by default
    assert "Do NOT omit the opening <tool_call>" not in u


# ════════════════════════════════════════════════════════════════════
#  Group 2: strict-where-upstream-silent
# ════════════════════════════════════════════════════════════════════


def test_T12_Q2_developer_role_accepted(upstream, custom_pub):
    """Q2: upstream raises on the `developer` role; the public fork
    accepts it as an alias for `system` (OpenAI Responses API
    convention used by opencode, Claude Code, openclaw, Continue)."""
    msgs = [
        {"role": "developer", "content": "You are a careful coder."},
        {"role": "user", "content": "Hi"},
    ]
    with pytest.raises(jinja2.TemplateError):
        _render(
            upstream, messages=msgs, enable_thinking=True, preserve_thinking=False
        )
    c = _render(custom_pub, messages=msgs, **RECOVER_UPSTREAM)
    assert "<|im_start|>system\nYou are a careful coder" in c


def test_T13_Q3_string_arguments_raise_debuggably(custom_pub):
    """Q3: upstream raises Jinja's natural `Can only get item pairs from
    a mapping` when arguments is a string — impossible to debug. The
    public fork raises a clear error that names the bug surface and
    links to the canonical discussion."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "find_files",
                        # JSON-encoded STRING, NOT a dict. This is exactly
                        # what the Vercel AI SDK hands back in some flows.
                        "arguments": '{"pattern": "*.py"}',
                    }
                }
            ],
        },
    ]
    with pytest.raises(jinja2.TemplateError) as ei:
        _render(custom_pub, messages=msgs, tools=TOOLS, add_generation_prompt=False)
    err = str(ei.value)
    assert "JSON object" in err
    assert "deserialize" in err.lower()
    assert "pi/issues/3325" in err


def test_T14_Q4_thinking_long_form_extracts_reasoning(upstream, custom_pub):
    """Q4: <think>...</thinking> (long-form close) is extracted as
    reasoning by the public fork. Upstream leaks the tags as literal
    content."""
    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "<think>\ntaking notes\n</thinking>\nthe answer",
        },
        {"role": "user", "content": "q2"},
    ]
    u = _render(
        upstream,
        messages=msgs,
        enable_thinking=True,
        preserve_thinking=True,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        enable_thinking=True,
        preserve_thinking=True,
        unwrap_tool_envelope=False,
        verbose_tool_instructions=False,
    )
    # Upstream leaves the literal tag in content (no extraction).
    assert "</thinking>" in u
    # Public extracts the reasoning into a proper <think>...</think> block
    # AND removes the literal </thinking> tag from the assistant content.
    assert "</thinking>" not in c
    assert "<think>\ntaking notes\n</think>" in c
    assert "the answer" in c


def test_T15_Q4_unclosed_think_with_tool_call_rescue(custom_pub):
    """Q4: <think> not closed, but followed by <tool_call>. Public
    fork rescues by treating everything up to the <tool_call> as
    reasoning content. Pattern from ollama#14493."""
    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": (
                "<think>\nplanning to call find_files\n"
                "<tool_call>\n<function=find_files>\n"
                "<parameter=pattern>\n*.py\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
        },
        {"role": "user", "content": "q2"},
    ]
    c = _render(
        custom_pub,
        messages=msgs,
        enable_thinking=True,
        preserve_thinking=True,
        unwrap_tool_envelope=False,
        verbose_tool_instructions=False,
    )
    # Rescued reasoning shows up in a proper <think>...</think> block.
    assert "<think>\nplanning to call find_files\n</think>" in c
    # And the tool call is preserved in content.
    assert "<tool_call>\n<function=find_files>" in c


@pytest.mark.parametrize(
    "label,assistant_content",
    [
        # Multiple </think> close tags in the body.
        ("multiple_close", "<think>\na</think>b</think>c"),
        # A second <think> opener nested inside the reasoning body.
        ("nested_opener", "<think>\nouter<think>inner\n</think>final"),
        # The realistic agent-editing-a-chat-template case: the reasoning
        # body literally mentions the token <think>. Upstream's last-opener
        # split mangles this; Q4 must reproduce upstream BYTE-FOR-BYTE here
        # (it is a plain </think> input, NOT a variant), even though both
        # produce the same arguably-lossy result.
        (
            "literal_think_in_body",
            "<think>\nI need to add a <think> tag handler...\n</think>\nDone.",
        ),
        # Ordinary single well-formed pair.
        ("plain", "<think>\nreasoning here\n</think>\nthe answer"),
    ],
)
def test_T18_Q4_plain_close_tag_byte_identity(
    upstream, custom_pub, label, assistant_content
):
    """Q4 byte-identity regression guard.

    The STRICT-EQUIVALENCE invariant carves out only `</thinking>`,
    whitespace `</think>` variants, and unclosed `<think>`. For *plain*
    `</think>` inputs — including pathological bodies (multiple `</think>`,
    embedded literal `<think>`, nested openers) — Q4 must reproduce
    upstream's exact last-opener / first-closer / last-closer split,
    byte-for-byte. An earlier Q4 revision diverged on these (it kept inner
    `<think>` literals and split on the FIRST closer), which would silently
    perturb the prefix cache on a coding agent editing chat templates.
    """
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "q2"},
    ]
    u = _render(
        upstream, messages=msgs, enable_thinking=True, preserve_thinking=True
    )
    c = _render(
        custom_pub,
        messages=msgs,
        enable_thinking=True,
        preserve_thinking=True,
        unwrap_tool_envelope=False,
        verbose_tool_instructions=False,
    )
    assert u == c, f"Q4 diverged from upstream on plain </think> case {label!r}"


def test_T16_Q5_envelope_unwrap_default_diverges(upstream, custom_pub):
    """Q5: by default, the public fork unwraps the OpenAI envelope to
    match Qwen3-Coder-Next's canonical pattern. With
    unwrap_tool_envelope=false, the upstream behavior is recovered."""
    msgs = [{"role": "user", "content": "find python"}]
    u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=True,
                preserve_thinking=False)
    c_default = _render(custom_pub, messages=msgs, tools=TOOLS)
    c_off = _render(custom_pub, messages=msgs, tools=TOOLS, **RECOVER_UPSTREAM)

    # Default-on: the envelope `"type": "function"` is NOT present in the
    # tools block (it's been stripped).
    assert '"type": "function"' not in c_default.split("</tools>")[0]
    # Off: matches upstream byte-for-byte in the tools block AND overall.
    assert c_off == u


def test_T17_Q6_verbose_important_block_gated(upstream, custom_pub):
    """Q6: the strengthened IMPORTANT block adds 3 bullets targeting
    documented Qwen3-Coder failure modes. Gated by
    verbose_tool_instructions, defaults true."""
    msgs = [{"role": "user", "content": "find python"}]
    c_default = _render(custom_pub, messages=msgs, tools=TOOLS)
    c_off = _render(custom_pub, messages=msgs, tools=TOOLS, **RECOVER_UPSTREAM)

    # Default: the strengthened bullet "Do NOT omit the opening <tool_call> tag"
    # IS present.
    assert "Do NOT omit the opening <tool_call> tag" in c_default
    # Off: it's GONE.
    assert "Do NOT omit the opening <tool_call> tag" not in c_off
    # And c_off matches upstream's IMPORTANT block verbatim.
    u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=True,
                preserve_thinking=False)
    assert c_off == u


# ════════════════════════════════════════════════════════════════════
#  Group 3: AGENTIC CODING SCENARIO TESTS
#
#  Each Sxx reproduces a documented public bug shape end-to-end and
#  verifies the public fork prevents the failure mode.
# ════════════════════════════════════════════════════════════════════


def test_S01_pi_3325_multi_turn_keeps_prior_reasoning(custom_pub):
    """Scenario: the canonical pi#3325 bug — after 2-3 turns of the
    same tool being called, the model emits arguments: {} despite the
    prior reasoning correctly identifying the parameters.

    Root cause: upstream defaults preserve_thinking=false, dropping
    prior-turn <think> blocks. The model loses its trace of how it
    chose arguments last time.

    Fix verification: with default kwargs, both prior-turn reasoning
    blocks are present in the rendered prompt.

    Ref: https://github.com/earendil-works/pi/issues/3325
    """
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "reasoning_content": "Turn 1: pattern is '*.py', dir is '.'",
            "tool_calls": [{
                "id": "c1",
                "function": {
                    "name": "find_files",
                    "arguments": {"pattern": "*.py", "dir": "."},
                },
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[a.py]"},
        {"role": "user", "content": "now ts files"},
        {
            "role": "assistant",
            "reasoning_content": "Turn 2: pattern '*.ts', same dir.",
            "tool_calls": [{
                "id": "c2",
                "function": {
                    "name": "find_files",
                    "arguments": {"pattern": "*.ts", "dir": "."},
                },
            }],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "[]"},
        {"role": "user", "content": "and rust files in src/"},
    ]
    c = _render(custom_pub, messages=msgs, tools=TOOLS)
    assert "Turn 1: pattern is '*.py'" in c
    assert "Turn 2: pattern '*.ts'" in c
    # Generation prompt opens cleanly.
    assert c.rstrip().endswith("<think>")


def test_S02_pi_3325_preserve_thinking_false_recovers_breakage(
    upstream, custom_pub
):
    """Companion to S01: explicitly setting preserve_thinking=false on
    the public template reproduces the upstream bug behavior. Proves
    our fix is a fix and not a coincidence."""
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "reasoning_content": "Turn 1: pattern is '*.py'",
            "tool_calls": [{
                "id": "c1",
                "function": {"name": "find_files",
                             "arguments": {"pattern": "*.py"}},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "user", "content": "now *.ts"},
    ]
    u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=True,
                preserve_thinking=False)
    c = _render(custom_pub, messages=msgs, tools=TOOLS, **RECOVER_UPSTREAM)
    assert u == c
    assert "Turn 1" not in u  # both drop prior reasoning
    assert "Turn 1" not in c


def test_S03_opencode_developer_role_no_crash(custom_pub):
    """Scenario: opencode (and Claude Code, openclaw, Continue) send a
    `developer` role for reasoning-capable models. Upstream raises
    'Unexpected message role' — crashing the request before any
    response is generated.

    Fix verification: the request renders successfully and the
    developer content lands in a system block.

    Ref: https://gist.github.com/sudoingX/c2facf7d8f7608c65c1024ef3b22d431
    """
    msgs = [
        {"role": "developer",
         "content": "You are a careful Python coding assistant."},
        {"role": "user", "content": "Find the entry point."},
    ]
    c = _render(custom_pub, messages=msgs, tools=TOOLS)
    assert "<|im_start|>system\n" in c
    assert "You are a careful Python coding assistant." in c


def test_S04_opencode_24264_stripped_kwargs_still_get_thinking(custom_pub):
    """Scenario: the Vercel AI SDK (used by opencode) strips unknown
    request fields, so the harness's attempt to pass
    chat_template_kwargs.enable_thinking=true gets dropped silently.
    With upstream defaults this means thinking is effectively off,
    degrading tool accuracy.

    With the public template's defaults, the SERVER-SIDE default is
    thinking-on (matches upstream's effective default since
    enable_thinking only fires the empty-think branch when explicitly
    false). And the strengthened IMPORTANT block is enabled by default,
    AND envelope-unwrap is enabled by default. The agentic happy-path
    works without any harness-side configuration.

    Ref: https://github.com/anomalyco/opencode/issues/24264
    """
    msgs = [{"role": "user", "content": "find files"}]
    # No kwargs at all — simulates the AI SDK stripping everything.
    c = _render(custom_pub, messages=msgs, tools=TOOLS)
    # Thinking prompt is present.
    assert "<think>\n" in c
    assert "<think>\n\n</think>\n\n" not in c  # NOT the thinking-off form
    # Verbose IMPORTANT block is present (Q6 default).
    assert "Do NOT omit the opening <tool_call> tag" in c
    # Envelope is unwrapped (Q5 default) — no '"type": "function"' wrapper
    # in the tools block.
    tools_block = c.split("</tools>")[0]
    assert '"type": "function"' not in tools_block


def test_S05_vercel_ai_sdk_string_arguments_surfaces_clearly(custom_pub):
    """Scenario: a harness adapter (Vercel AI SDK, OpenAI compat bridge,
    etc.) hands tool_call.arguments back as a JSON-encoded STRING
    instead of the deserialized object. Upstream raises Jinja's
    natural `Can only get item pairs from a mapping` — undebuggable.
    Public fork raises with a clear, debuggable error message.

    Ref: https://github.com/earendil-works/pi/issues/3325
    """
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "c1",
                "function": {
                    "name": "find_files",
                    "arguments": '{"pattern": "*.py", "dir": "src"}',
                },
            }],
        },
    ]
    with pytest.raises(jinja2.TemplateError) as ei:
        _render(
            custom_pub,
            messages=msgs,
            tools=TOOLS,
            add_generation_prompt=False,
        )
    msg = str(ei.value)
    assert "JSON object" in msg
    assert "deserialize" in msg.lower()
    assert "pi/issues/3325" in msg


def test_S06_long_multi_step_history_remains_coherent(custom_pub):
    """End-to-end scenario: a realistic multi-step coding agent history
    with `developer` system, reasoning, multiple tools, and several
    turns. The kind of session opencode/pi produce in real use. The
    render must NOT raise, MUST preserve all reasoning, MUST accept
    the developer role, AND MUST emit balanced turn frames."""
    msgs = [
        {"role": "developer",
         "content": "You are a careful Python coding assistant."},
        {"role": "user",
         "content": "Find all Python files, then read main.py."},
        {
            "role": "assistant",
            "reasoning_content": "I'll start with find_files for the pattern.",
            "tool_calls": [{
                "id": "c1",
                "function": {
                    "name": "find_files",
                    "arguments": {"pattern": "*.py", "dir": "."},
                },
            }],
        },
        {"role": "tool", "tool_call_id": "c1",
         "content": '["main.py", "utils.py"]'},
        {
            "role": "assistant",
            "reasoning_content": "Now read main.py.",
            "tool_calls": [{
                "id": "c2",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "main.py"},
                },
            }],
        },
        {"role": "tool", "tool_call_id": "c2",
         "content": "def main():\n    print('hello')"},
        {"role": "assistant",
         "content": "Found 2 Python files; main.py prints 'hello'."},
    ]
    c = _render(custom_pub, messages=msgs, tools=CODING_TOOLS)

    # Developer role accepted as system.
    assert "<|im_start|>system\n" in c
    assert "You are a careful Python coding assistant." in c

    # All reasoning blocks survive.
    assert "I'll start with find_files" in c
    assert "Now read main.py" in c

    # Both tool calls and responses present.
    assert "<tool_call>\n<function=find_files>" in c
    assert "<tool_call>\n<function=read_file>" in c
    assert c.count("<tool_response>") == 2

    # Generation prompt opens cleanly.
    assert c.rstrip().endswith("<think>")


def test_S07_Q7_developer_mid_conversation_does_not_crash(custom_pub):
    """Q7: opencode / Continue / Claude-Code-style harnesses inject a
    `developer` (or `system`) steering message MID-session. Q2 only
    aliased it at index 0; mid-conversation upstream's guard still
    hard-crashed the request (the exact failure class Q2 was meant to
    prevent, just at a later index). The fork now renders it as an inline
    system frame instead of killing the live agent loop."""
    msgs = [
        {"role": "developer", "content": "You are a careful coder."},
        {"role": "user", "content": "find the bug"},
        {"role": "assistant", "content": "Looking now."},
        {"role": "developer", "content": "Prefer minimal diffs."},
        {"role": "user", "content": "go"},
    ]
    c = _render(custom_pub, messages=msgs)  # must not raise
    # The mid-conversation developer message is emitted as a system frame.
    assert "<|im_start|>system\nPrefer minimal diffs.<|im_end|>" in c
    # The index-0 developer still becomes the leading system block.
    assert c.startswith("<|im_start|>system\nYou are a careful coder.")


def test_S07b_Q7_mid_system_also_handled(custom_pub):
    """Q7 covers a mid-conversation `system` role too (not just
    `developer`), since upstream crashed on both."""
    msgs = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late steering"},
        {"role": "user", "content": "go"},
    ]
    c = _render(custom_pub, messages=msgs)  # must not raise
    assert "<|im_start|>system\nlate steering<|im_end|>" in c


def test_S07c_Q7_strict_system_position_recovers_raise(custom_pub):
    """Q7: passing strict_system_position=true restores upstream's hard
    raise on a mid-conversation system/developer message. (Byte-identity
    is unaffected because upstream raises on this shape regardless.)"""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    with pytest.raises(jinja2.TemplateError):
        _render(custom_pub, messages=msgs, strict_system_position=True)


def test_S07d_Q7_upstream_still_crashes_mid_system(upstream):
    """Document the upstream behavior Q7 diverges from: upstream raises on
    any system message after index 0. Establishes that Q7 is a
    strict-improvement on a shape upstream rejects (no byte-identity claim
    on this input)."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    with pytest.raises(jinja2.TemplateError):
        _render(upstream, messages=msgs, enable_thinking=True,
                preserve_thinking=False)


def test_S08_Q8_single_tool_call_mapping_not_dropped(custom_pub):
    """Q8: some OpenAI-compat adapters hand back a single tool_call OBJECT
    (a mapping) rather than a one-element list. Upstream's
    `... is not mapping` guards then SILENTLY DROP the call, producing an
    empty assistant turn that desyncs the following tool message. The fork
    normalizes a single mapping into a one-item list, so the <tool_call>
    block is still emitted."""
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": {
                "id": "c1",
                "function": {
                    "name": "find_files",
                    "arguments": {"pattern": "*.py"},
                },
            },
        },
    ]
    c = _render(
        custom_pub, messages=msgs, tools=TOOLS, add_generation_prompt=False
    )
    assert "<tool_call>\n<function=find_files>" in c
    assert "<parameter=pattern>\n*.py" in c


def test_S08b_Q8_upstream_drops_single_mapping(upstream):
    """Document the upstream behavior Q8 fixes: a single tool_call mapping
    is silently dropped (no <tool_call> emitted). Establishes that Q8 is a
    strict-improvement on a shape upstream silently mishandles."""
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": {
                "id": "c1",
                "function": {
                    "name": "find_files",
                    "arguments": {"pattern": "*.py"},
                },
            },
        },
    ]
    u = _render(
        upstream,
        messages=msgs,
        tools=TOOLS,
        enable_thinking=True,
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    # Upstream's guard treats the mapping as non-iterable-tool_calls and
    # emits no real function call. (The literal "<tool_call>" example
    # string still appears inside the IMPORTANT instructions block, so we
    # assert on the actual call marker instead.)
    assert "<function=find_files>" not in u
    # The assistant turn itself is empty (just an empty think block).
    assistant_turn = u.split("<|im_start|>assistant", 1)[1]
    assert "<tool_call>" not in assistant_turn


def test_S08c_Q8_well_formed_list_still_byte_identical(upstream, custom_pub):
    """Q8 regression guard: the normalization must NOT perturb the
    well-formed list case. With gates recovered, a normal list-of-tool_calls
    assistant turn stays byte-identical to upstream."""
    msgs = [
        {"role": "user", "content": "find python files"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "find_files",
                        "arguments": {"pattern": "*.py", "dir": "src"},
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
        preserve_thinking=False,
        add_generation_prompt=False,
    )
    c = _render(
        custom_pub,
        messages=msgs,
        tools=TOOLS,
        **RECOVER_UPSTREAM,
        add_generation_prompt=False,
    )
    assert u == c


# ════════════════════════════════════════════════════════════════════
#  Group 4: brainstorm validation lock-ins (2026-05-31). Additive and
#  byte-identity-safe — they pin behaviors confirmed by offline render-
#  tests so a future refactor can't silently regress them. No template
#  change accompanied these; they only assert on existing behavior.
# ════════════════════════════════════════════════════════════════════


def test_T7b_three_mixed_parallel_calls_well_formed(upstream, custom_pub):
    """Solo deployment fans out up to 3 parallel tool calls per turn. A list
    of 3 calls — mixing the {function:{...}} envelope with the flat
    {name,arguments} form — must render as 3 separate, fully-closed,
    non-nested <tool_call> blocks. The all-envelope variant stays
    byte-identical to upstream with the gates off."""
    mixed = [
        {"role": "user", "content": "scan the repo"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "find_files", "arguments": {"pattern": "*.py"}}},
            {"id": "c2", "name": "read_file", "arguments": {"path": "main.py"}},  # flat form
            {"id": "c3", "function": {"name": "find_files", "arguments": {"pattern": "*.md"}}},
        ]},
    ]
    c = _render(custom_pub, messages=mixed, tools=CODING_TOOLS,
                **RECOVER_UPSTREAM, add_generation_prompt=False)
    asst = c.split("<|im_start|>assistant", 1)[1].split("<|im_end|>", 1)[0]
    assert asst.count("<tool_call>") == 3 == asst.count("</tool_call>")
    assert asst.count("</function>") == 3
    assert "<tool_call>\n<tool_call>" not in asst  # no nesting

    env = [
        {"role": "user", "content": "scan the repo"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "find_files", "arguments": {"pattern": "*.py"}}},
            {"id": "c2", "function": {"name": "read_file", "arguments": {"path": "main.py"}}},
            {"id": "c3", "function": {"name": "find_files", "arguments": {"pattern": "*.md"}}},
        ]},
    ]
    u = _render(upstream, messages=env, tools=TOOLS, enable_thinking=True,
                preserve_thinking=False, add_generation_prompt=False)
    cc = _render(custom_pub, messages=env, tools=TOOLS,
                 **RECOVER_UPSTREAM, add_generation_prompt=False)
    assert u == cc  # all-envelope multi-call list: byte-identical to upstream


def test_T19_message_name_is_inert_and_byte_identical(upstream, custom_pub):
    """OpenAI permits a per-message `name` (multi-agent routing / function
    identity). Qwen pairs tool calls/responses positionally by id order, so
    the template intentionally IGNORES `name`: it must never appear in the
    output, and the render stays byte-identical to upstream."""
    msgs = [
        {"role": "user", "content": "find python files", "name": "router_alpha"},
        {"role": "assistant", "content": "LGTM", "name": "code_reviewer"},
        {"role": "user", "content": "thanks"},
    ]
    u = _render(upstream, messages=msgs, enable_thinking=True, preserve_thinking=False)
    c = _render(custom_pub, messages=msgs, **RECOVER_UPSTREAM)
    assert u == c
    assert "router_alpha" not in c and "code_reviewer" not in c


def test_S07e_multiple_leading_steering_messages_render_as_ordered_frames(custom_pub):
    """Contract: index-0 leads; an ADDITIONAL leading system/developer (e.g. a
    system followed by a developer at the very start) is emitted as its own
    inline <|im_start|>system frame, in original order — never merged."""
    msgs = [
        {"role": "system", "content": "SYS-LEAD"},
        {"role": "developer", "content": "DEV-SECOND"},
        {"role": "user", "content": "go"},
    ]
    c = _render(custom_pub, messages=msgs, add_generation_prompt=False)
    assert c.count("<|im_start|>system\n") == 2          # two distinct frames
    assert c.index("SYS-LEAD") < c.index("DEV-SECOND")   # original order
    assert "<|im_start|>system\nSYS-LEAD<|im_end|>" in c
    assert "<|im_start|>system\nDEV-SECOND<|im_end|>" in c


def test_S09_nonstring_tool_content_contract(upstream, custom_pub):
    """`tool` content contract (sibling of Q3, but byte-identical to upstream —
    no defensive patch added): string -> verbatim; text-part list ->
    concatenated; bare dict -> clean raise. All must match upstream."""
    pre = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "find_files", "arguments": {"pattern": "*.py"}}}]},
    ]

    def both_match(tool_content):
        msgs = pre + [{"role": "tool", "tool_call_id": "c1", "content": tool_content}]
        u = _render(upstream, messages=msgs, tools=TOOLS, enable_thinking=True,
                    preserve_thinking=False, add_generation_prompt=False)
        c = _render(custom_pub, messages=msgs, tools=TOOLS,
                    **RECOVER_UPSTREAM, add_generation_prompt=False)
        return u == c

    # (a) string -> verbatim; (b) text-part list -> concatenated. Both byte-identical.
    assert both_match("file contents")
    assert both_match([{"type": "text", "text": "part-one"}, {"type": "text", "text": "part-two"}])

    # (c) bare dict -> both raise (non-standard wire shape; out of scope, not patched).
    bad = pre + [{"role": "tool", "tool_call_id": "c1", "content": {"files": ["a.py"]}}]
    with pytest.raises(jinja2.TemplateError):
        _render(custom_pub, messages=bad, tools=TOOLS, **RECOVER_UPSTREAM, add_generation_prompt=False)
    with pytest.raises(jinja2.TemplateError):
        _render(upstream, messages=bad, tools=TOOLS, enable_thinking=True,
                preserve_thinking=False, add_generation_prompt=False)
