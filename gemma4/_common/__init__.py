"""Shared Modal building blocks for Gemma 4 family deployments.

Each shape's serve script imports from here rather than re-implementing
the common pieces:

- ``model_registry`` — canonical map from short name (``e2b``, ``e4b``,
  ``12b``, ``26b``, ``31b``) to HF repo, recommended Modal GPU class,
  native context length, and MTP drafter.
- ``sglang_common`` — Modal SGLang image + ``sglang.launch_server`` command
  builder + the NEXTN MTP speculative profiles.
- ``health`` — /health poll + Modal Memory Snapshot warmup/release/resume
  helpers.
- ``gemma4_parser`` — client-side parser for Gemma 4's raw
  ``<|tool_call>...<tool_call|>`` token format.

The project makes this package importable on Modal by appending
``add_local_python_source("_common")`` as the LAST build step of the
deployment image, so serve scripts use ``from _common.<module> import
<name>``.
"""
