# CLAUDE.md

Guidance for working in this repository.

## This is a public repo

It is open source. Keep it that way: no personal Modal workspace names, account
IDs, hostnames, IP addresses, real names, emails, or private/employer data in
code, comments, docs, or test fixtures. App names, volume names, and secret
names are generic on purpose so anyone can deploy on their own account. If you
port code from elsewhere, scrub it.

## Three independent projects

`gemma4/`, `qwen/`, and `pipeline/` are separate uv projects. `uv`, `modal`, and
`pytest` all run per-project — there is no tooling at the repo root. `cd` into
the project before running anything.

| Project | Entry point |
|---|---|
| `gemma4/` | SGLang serving for the Gemma 4 family, solo + concurrent |
| `qwen/` | SGLang serving for Qwen3.6-27B and 35B-A3B, solo + concurrent |
| `pipeline/` | vLLM serve → eval → corpus → SFT, over the Chinook SQL agent |

## The `_common` skeleton

Each serving project is built from the same parts. Learn it once.

- `_common/model_registry.py` — `ModelSpec` (HF repo, pinned revision, GPU class,
  native context, MTP drafter) keyed by short name. The single source of truth.
  `serve.py` calls `get("<short>")`.
- `_common/sglang_common.py` — `make_sglang_image()` (pinned `lmsysorg/sglang`
  base) and `build_serve_cmd()` (the model- and shape-agnostic argv builder for
  `sglang.launch_server`). MTP speculative profiles live here too.
- `_common/health.py` — `/health` polling and the memory-snapshot helpers
  (`release_`/`resume_memory_occupation`, `send_warmup_request`).
- `pipeline/_common/` adds `vllm_common.py`, the `gemma4_parser.py` client-side
  tool-call parser, and `eval_scoring.py`.

**To tune a deployment, edit only the constants block at the top of its
`serve.py`.** `build_serve_cmd()` is deliberately model-agnostic; keep it that
way.

## Two invariants every serve.py respects

1. **Image build order is fixed.** Base SGLang image → `add_local_file(...,
   copy=True)` for each baked chat template → `add_local_python_source("_common")`
   as the last step. Modal forbids further build steps after a non-copy
   `add_local_*`, so `_common` goes last and the helpers never call `add_local_*`
   themselves.

2. **Ingress is Modal's public web endpoint. There is no Tailscale.** The server
   binds `0.0.0.0:8000`. `@modal.web_server(port=8000)` publishes the public
   `*.modal.run` URL and is the innermost decorator. No auth is baked in — see
   `docs/securing-endpoints.md`. The health/warmup helpers reach the server over
   the loopback; that's the only place `127.0.0.1` should appear.

## Solo vs concurrent, and snapshots

- Single-GPU deployments use `@app.cls` with `@modal.enter(snap=True/False)` and
  `enable_memory_snapshot=True` + `experimental_options={"enable_gpu_snapshot":
  True}` to skip CUDA-graph capture and warmup on cold starts after the first.
- Multi-GPU deployments (the 26B-A4B at TP=2) use a plain `@app.function` —
  snapshots are incompatible with multi-GPU.
- `@modal.concurrent(max_inputs=, target_inputs=)` sets the SGLang scheduler
  ceiling. Match `CUDA_GRAPH_BS` to `max_running_requests`.

## Adding a model or shape

1. Add a `ModelSpec` to the project's `_common/model_registry.py`.
2. Copy the nearest sibling's `serve.py` (dense → dense, MoE → MoE) and change
   `SPEC = get("<short>")` plus the constants block. No code below the constants
   should need to change.
3. Pick the chat template. For Gemma, the 31B/26B/12B upstream templates are
   byte-identical, so the custom fork applies to all three; E2B/E4B use their own
   upstream. If you change a template, re-run the conformance suite.

## Gemma 4 gotchas

- **The Triton attention backend is mandatory** across the family (fixed
  `head_dim=256`, 512-wide global head). FlashInfer/trtllm reject it. The wrong
  backend produces garbled output, not an error — the boot health check on the
  first real request is what catches a misconfigure.
- FP8 KV cache (`fp8_e5m2`) is the default for the dense models at long context.
- The 12B has no published MTP drafter, so it serves without speculative decoding
  (`draft=None`, `MTP_OFF`).

## Testing

```bash
# Chat-template conformance (no GPU, renders Jinja locally):
cd gemma4 && uv run --group dev pytest tests/ -v        # 20 tests
cd qwen   && uv run --group dev pytest tests/ -v        # 39 tests

# Pipeline eval rubric unit tests:
cd pipeline && uv run --group dev pytest eval/test_eval_scoring.py -v
```

## Where to read next

- `gemma4/README.md`, `qwen/README.md`, `pipeline/README.md` — per-project guides.
- `gemma4/chat_templates/TESTING.md` — how the template fork is tested.
- `docs/securing-endpoints.md` — auth options for the public endpoint.
- `docs/deploy-byo-cloud.md` — running off Modal.
