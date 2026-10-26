# glimmer

SGLang serving for Meta's [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
on Modal, using NVIDIA's NVFP4 checkpoint
([`nvidia/Muse-Glimmer-30B-NVFP4`](https://huggingface.co/nvidia/Muse-Glimmer-30B-NVFP4))
and the DFlash speculative-decoding drafter. Two shapes, solo and concurrent,
built on the same `_common` skeleton as `gemma4/` and `qwen/`.

What you get from one deploy:

- the full 131K native context per session
- text + image input (NVIDIA's checkpoint keeps the vision encoder)
- tool calling and reasoning via SGLang's `muse` parsers, OpenAI-compatible
- DFlash speculative decoding (~3x single-stream speed)
- GPU memory snapshots, so cold starts after the first skip weight load,
  kernel autotuning and CUDA-graph capture

## Deploy

```bash
cd glimmer
uv sync
uv run modal token new                      # once

uv run modal deploy deployments/solo/serve.py         # app "glimmer-solo"
uv run modal deploy deployments/concurrent/serve.py   # app "glimmer-concurrent"

# Before pointing a harness at it (6 checks: chat, tools, tool round trip,
# streamed tools, images, reasoning_effort):
uv run python bench/smoke_test.py --base-url https://<printed-url>/v1

uv run modal app stop glimmer-solo          # stop billing
```

`modal deploy` prints the public `*.modal.run` URL. It is open by default; see
[../docs/securing-endpoints.md](../docs/securing-endpoints.md). Setting an
`API_KEY` env var in the container (from a `modal.Secret`) turns on SGLang's
bearer-token check.

**Cold starts.** The first request after a deploy boots the server and takes
a GPU memory snapshot (~5 min). Modal keeps snapshots per worker type, so the
first two or three cold starts may each take a fresh one; after that a cold
start restores in **~8 s** (measured on the RTX PRO 6000). Any redeploy that
changes the image or code starts the cycle again. Weights (~30 GB with the
drafter) are cached in the `glimmer-hf-cache` Modal Volume, so they download
once.

## Which GPU

NVFP4 needs Blackwell. Two Modal GPU classes are tested; set `GPU` at the top
of a `serve.py`:

| | RTX PRO 6000 (default) | B200 |
|---|---|---|
| Memory | 96 GB GDDR7 | 180 GB HBM3e |
| Modal list price | ~$3.03/hr | ~$6.25/hr |
| KV pool (FP8, SWA ratio 0.1, as deployed) | ~2.9M tokens | ~6.9M tokens |
| Single-stream decode (DFlash 16) | ~180–210 tok/s | ~320–370 tok/s |
| Peak aggregate seen (64 in flight, no spec) | ~2,100 tok/s | ~3,200 tok/s |
| Tokens per dollar | **best** | ~0.7–0.9x of RTX |

The RTX PRO 6000 wins on cost at every load measured. Pick the B200 when
single-stream latency matters more than cost; it is ~1.75x faster per stream.
`B200+` is deliberately not used: it can land on a B300, which needs CUDA
13.1+, and the pinned SGLang image is CUDA 13.0.

## Solo vs concurrent

- **solo** — one user driving an agent harness. Up to 8 requests in flight
  (parallel tool calls, sub-agents), DFlash block 16 for the fastest stream.
- **concurrent** — many sessions on one GPU. Up to 64 in flight, DFlash
  block 8, which is the best all-rounder across loads (table below). For
  batch jobs that keep 48+ requests in flight, set `SPECULATIVE = DFLASH_OFF`.
  `MAX_CONTAINERS` lets Modal add GPUs past `TARGET_RUNNING_REQUESTS`.

Tune a deployment by editing only the constants block at the top of its
`serve.py`.

## Benchmarks

Aggregate output tok/s from `bench/benchmark.py` run inside the container
against localhost (coding prompts, `reasoning_strength=low`, up to 1024 output
tokens, model-card sampling: temperature 1.0, top_p 0.95, top_k 64). SGLang
v0.5.22, FP8 KV.

| GPU | Speculative | c=1 | c=4 | c=16 | c=64 |
|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | DFlash 16 | **180** | **499** | 908 | 1,272 |
| RTX PRO 6000 | DFlash 8 | 152 | 455 | **1,118** | 1,899 |
| RTX PRO 6000 | off | 62 | 206 | 670 | **2,135** |
| B200 (`trtllm_mha`) | DFlash 16 | **318** | **706** | 1,172 | 1,584 |
| B200 (FlashInfer) | DFlash 8 | 249 | 659 | **1,315** | 2,313 |
| B200 (`trtllm_mha`) | off | — | — | 1,102 | **3,237** |

DFlash accepts ~2.5–3.5 tokens per verify step at temperature 1.0. Below ~16
concurrent requests the GPU is bandwidth-bound and DFlash is close to free;
above that, verifying 8–16 draft tokens per sequence competes with real work.

Long context: a needle buried at ~81% depth of a 118K-token prompt is retrieved
correctly; the 118K prefill takes ~23 s on the RTX PRO 6000, and a repeat that
hits the prefix cache returns in ~0.5 s.

Reproduce against your own deployment:

```bash
uv run python bench/benchmark.py --endpoint https://<printed-url> \
    --concurrency 1 4 16 --requests 32 --reasoning low --max-tokens 1024
```

Over the public URL the concurrent deployment (RTX PRO 6000, DFlash 8)
measured 152 / 1,158 / 1,708 tok/s at 1 / 16 / 48 in flight, in line with the
in-container numbers; TTFT there includes your network round trip.

## What's specific to Glimmer

- **Reasoning strength** (`low` / `medium` / `high` / `xhigh`, default
  `high`) is a chat-template variable. Send it as
  `chat_template_kwargs={"reasoning_strength": "low"}`, or use OpenAI's
  `reasoning_effort` field (see the template fork below). Don't also write
  "Reasoning strength: …" into the system prompt unless you mean it: the
  template then uses yours and skips its own.
- **Sampling**: the model card recommends temperature 1.0, top_p 0.95, top_k
  64. `top_k` is an SGLang extension, so OpenAI clients pass it in `extra_body`.
- **Reasoning tokens count against `max_tokens`.** At `high` the model can
  think for thousands of tokens; size `max_tokens` for agent work accordingly.
- **Sliding-window KV**: 39 of 52 layers use a 2048-token window. The
  deployments set `--swa-full-tokens-ratio 0.1` so that memory goes to
  full-context KV instead (1.77M → 3.07M tokens on the RTX PRO 6000).
- **No `expandable_segments`**: unlike `qwen/`, the image does not set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, because the
  memory-saver path that snapshots rely on refuses to start with it.
- **Tool-call history**: the template requires each past tool call's
  `arguments` to be a dict. SGLang parses the OpenAI-standard JSON string
  for you before rendering, so clients need do nothing; the smoke test's
  round-trip check covers this path.

## Chat template

`chat_templates/muse_glimmer_upstream.jinja` is Meta's template pinned at
`meta-models/Muse-Glimmer-30B@cec94fc` (newer than the copy in NVIDIA's repo:
it skips the template's own reasoning line when the system prompt already has
one). `custom_chat_template_muse_glimmer.jinja` adds one gated patch:

- **[G1] `reasoning_effort` → reasoning strength.** SGLang forwards the
  request's `reasoning_effort` to the template, but upstream only reads
  `reasoning_strength`, so harnesses that send the standard OpenAI field were
  always served at `high`. Mapping: `none`/`minimal`/`low` → `low`,
  `medium` → `medium`, `high` → `high`, `xhigh`/`max` → `xhigh`. An explicit
  `reasoning_strength` still wins.

Without `reasoning_effort` the fork renders byte-identically to upstream, so
prefix caching is unaffected.

```bash
uv run --group dev pytest tests/ -v    # template conformance + serve-config checks, no GPU
```

## The `_common` skeleton

- `model_registry.py` — the checkpoint and drafter with pinned revisions, plus
  per-GPU profiles (memory fraction, attention backend, price).
- `sglang_common.py` — `make_sglang_image()` (pinned `lmsysorg/sglang`
  v0.5.22 by digest), the DFlash profiles, and `build_serve_cmd()`.
- `health.py` — `/health` polling and the memory-snapshot helpers.

## References

- Model card: https://huggingface.co/meta-models/Muse-Glimmer-30B
- NVFP4 card (quantization recipe, evals): https://huggingface.co/nvidia/Muse-Glimmer-30B-NVFP4
- DFlash drafter: https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant
- SGLang support PR: https://github.com/sgl-project/sglang/pull/34262
- DFlash: https://arxiv.org/abs/2602.06036
