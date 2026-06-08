# gemma4

SGLang serving for the Google Gemma 4 family on Modal, in solo and concurrent
shapes, with a Granite embedding sidecar. Five sizes share one `_common`
skeleton; the differences live in each deployment's constants block.

## Models

| Short | Repo | Arch | Native ctx | Default GPU | MTP |
|---|---|---|---|---|---|
| `e2b` | `google/gemma-4-E2B-it` | Dense+PLE, 5.1B/~2B | 128K | L4 | NEXTN (78M) |
| `e4b` | `google/gemma-4-E4B-it` | Dense+PLE, 8B/4.5B | 128K | L40S | NEXTN (79M) |
| `12b` | `google/gemma-4-12B-it` | Dense, ~12B, 48L | 256K | H100 | none |
| `26b` | `google/gemma-4-26B-A4B-it` | MoE, 25B/3.8B active | 256K | B200×2 | NEXTN (0.4B) |
| `31b` | `google/gemma-4-31B-it` | Dense, 31B, 60L | 256K | B200 | NEXTN (0.5B) |

Full per-model architecture notes are in `_common/model_registry.py`.

## Deploy

```bash
uv sync
uv run modal token new

# deployments/<model>/<shape>/serve.py  ->  app "gemma4-<model>-<shape>"
uv run modal deploy deployments/31b/solo/serve.py
uv run modal deploy deployments/e4b/concurrent/serve.py

uv run modal app stop gemma4-31b-solo
```

`modal deploy` prints the public `*.modal.run` URL. The endpoint is OpenAI-
compatible (`/v1/chat/completions`, `/v1/models`) and public by default — see
[../docs/securing-endpoints.md](../docs/securing-endpoints.md) for auth.

The Granite embedding sidecar is separate:

```bash
uv run modal deploy deployments/embeddings/serve_granite.py   # app "granite-embed"
```

## Solo vs concurrent

- **solo** — one user, the whole GPU, a large context window (e.g. 192K), KV and
  compute tuned for one fast stream. For driving a coding harness.
- **concurrent** — several agents on one GPU, smaller per-session window,
  fair-share scheduling.

Tune a deployment by editing only the constants block at the top of its
`serve.py` (`MAX_RUNNING_REQUESTS`, `CONTEXT_LENGTH`, `CHUNKED_PREFILL_SIZE`,
`CUDA_GRAPH_BS`, `MTP_PROFILE`). Everything below it is shared and
model-agnostic.

## The `_common` skeleton

- `model_registry.py` — `ModelSpec` per size (repo, pinned revision, GPU, native
  context, MTP drafter). The source of truth; `serve.py` calls `get("<short>")`.
- `sglang_common.py` — `make_sglang_image()` (pinned `lmsysorg/sglang`) and
  `build_serve_cmd()` (the argv builder), plus the NEXTN MTP profiles.
- `health.py` — `/health` polling and the memory-snapshot helpers.
- `gemma4_parser.py` — client-side parser for Gemma 4's native tool-call DSL.

## What's specific to Gemma 4

- **Triton attention backend is mandatory** (fixed `head_dim=256`). The wrong
  backend serves garbled output, not an error — the boot warmup request is what
  surfaces it.
- Dense models use FP8 KV cache (`fp8_e5m2`) at long context.
- Single-GPU deployments use Modal memory snapshots (`@app.cls` +
  `@modal.enter(snap=...)`) to skip CUDA-graph capture on warm cold-starts. The
  26B runs TP=2 on two B200s, which is incompatible with snapshots, so it uses a
  plain `@app.function`.
- The 12B has no published MTP drafter, so it serves without speculative
  decoding.

## Chat templates

`chat_templates/` holds the upstream templates, the P1–P5 custom fork, and the
conformance suite. The 31B / 26B / 12B upstream templates are byte-identical, so
the fork applies to all three; E2B/E4B use their own upstream. See
[chat_templates/README.md](chat_templates/README.md) and
[chat_templates/TESTING.md](chat_templates/TESTING.md).

```bash
uv run --group dev pytest tests/ -v       # 20-test conformance suite, no GPU
```

## Benchmarks

`bench/benchmark.py` sweeps concurrency against a live endpoint and scrapes
SGLang `/metrics` (prefix-cache hit rate, MTP acceptance length).
`bench/capture_samples.py` saves full responses for review. Both default their
endpoint to `$ENDPOINT` or `http://localhost:8000` and ship neutral coding and
agentic prompt profiles.
