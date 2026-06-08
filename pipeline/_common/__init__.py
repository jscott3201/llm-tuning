"""Shared library for the Gemma 4 serving + evaluation pipeline.

Each stage's scripts import from here rather than re-implementing the
common pieces:

- `vllm_common`   — Modal image + `vllm serve` command builder + health-check poll
- `sglang_common` — Modal image + `sglang.launch_server` command builder (re-uses
                    `vllm_common.wait_for_health`)
- `gemma4_parser` — raw `<|tool_call>...<tool_call|>` token parser
- `eval_scoring`  — 5-axis scoring core (selection / args / semantic / SQL / safety)
- `model_registry`— canonical map from short name (`e2b`, `e4b`, `12b`, `26b`,
                    `31b`) to HF repo and recommended Modal GPU class

The serve scripts add this package to their Modal image via
`make_vllm_image().add_local_python_source("_common")` (or the SGLang
equivalent), then import with `from _common.<module> import <name>`.

If you're skimming the project, you don't need to read this folder
top-to-bottom — each stage's README links to the helpers it actually
relies on, in the order they show up.
"""
