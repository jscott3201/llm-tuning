# qwen

SGLang serving for Qwen3.6 on Modal, in solo and concurrent shapes, with a
Granite embedding sidecar. Two models share one `_common` skeleton.

## Models

| Short | Repo | Arch | Native ctx | Default GPU |
|---|---|---|---|---|
| `27b` | `Qwen/Qwen3.6-27B` | Dense hybrid (DeltaNet + full-attn), 64L | 256K | B200 |
| `35b` | `Qwen/Qwen3.6-35B-A3B` | MoE hybrid, 35B/~3B active, 256 experts | 256K | B200 |

Both are multimodal checkpoints served text-only, both serve at TP=1 on a single
GPU, and both carry an architectural MTP head (no separate drafter model). Full
notes are in `_common/model_registry.py`.

## Deploy

```bash
uv sync
uv run modal token new

# deployments/<model>/<shape>/serve.py  ->  app "qwen36-<model>-<shape>"
uv run modal deploy deployments/27b/solo/serve.py
uv run modal deploy deployments/35b/concurrent/serve.py

uv run modal app stop qwen36-27b-solo
```

OpenAI-compatible, public by default — see
[../docs/securing-endpoints.md](../docs/securing-endpoints.md). The embedding
sidecar:

```bash
uv run modal deploy deployments/embeddings/serve_granite.py
```

## Solo vs concurrent

- **solo** — one session, the whole GPU, a long context window, `torch.compile`
  on with a cached inductor dir, and the longer-chain MTP profile for single-
  stream throughput.
- **concurrent** — several agents, smaller windows, the lower-overhead MTP
  profile, no `torch.compile`.

Edit only the constants block at the top of each `serve.py`.

## What's specific to Qwen3.6

- The hybrid backbone (Gated DeltaNet linear layers + full-attention layers)
  needs the right SGLang knobs: `trtllm_mha` attention backend, the mamba
  scheduler strategy, and `page_size=64`. A wrong `--linear-attn-decode-backend`
  produces garbled output with no error, so the boot health check is the first
  real validation.
- MTP runs the SPEC_V2 linear profiles only; tree-verify (`eagle_topk>1`) hard-
  errors on this stack, and `build_serve_cmd()` guards against it.
- Parsers: `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`.
- The 35B-A3B tuning is conservative and carries `TODO(benchmark)` markers — it
  was not validated on hardware. Benchmark before you trust the numbers.

## Chat templates

`chat_templates/` holds the upstream template, the Q1–Q8 custom fork, the
conformance suite, a live probe, and the A/B coding results. The fork fixes
multi-turn thinking collapse, the `developer` role, string-typed tool arguments,
`</think>` variant parsing, the OpenAI tool envelope, and a few more. Each patch
is gated. See [chat_templates/README.md](chat_templates/README.md) and
[chat_templates/TESTING.md](chat_templates/TESTING.md).

```bash
uv run --group dev pytest tests/ -v       # 39-test conformance suite, no GPU
```

## Benchmarks

`bench/benchmark.py` and `bench/capture_samples.py` drive a live endpoint with
neutral coding and agentic prompt profiles; `bench/smoke_test.py` is a quick
PASS/FAIL check (a chat completion and one tool call) to run before pointing a
harness at the endpoint. All default to `$ENDPOINT` or `http://localhost:8000`.
