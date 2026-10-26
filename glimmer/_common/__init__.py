"""Shared Modal building blocks for the Muse Glimmer deployments.

Added to the Modal image as the final build step via
``image.add_local_python_source("_common")`` in each serve script, so
deployment code imports helpers as ``from _common.<module> import <name>``.

Public modules:
  - ``model_registry`` — checkpoint, pinned revisions, DFlash drafter, GPU budgets.
  - ``sglang_common``  — SGLang image builder, DFlash profiles, serve-cmd builder.
  - ``health``         — /health polling and the memory-snapshot helpers.
"""
