"""Shared Modal building blocks for Qwen3.6 deployments.

This package is added to the Modal image as the final build step via
``image.add_local_python_source("_common")`` in each serve script, so
deployment code imports its helpers as ``from _common.<module> import
<name>``.

Public modules:
  - ``model_registry`` — model spec table (27B dense-hybrid, 35B-A3B MoE).
  - ``sglang_common`` — SGLang image builder, MTP profiles, serve-cmd builder.
  - ``health``        — /health polling helper for serve entrypoints.
"""
