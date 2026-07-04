# llm-tuning

Serving and fine-tuning code for the Google Gemma 4 and Qwen3.6 model families,
built around solo and concurrent agentic-coding workloads. Everything here runs
on [Modal](https://modal.com) out of the box, but the serving layer is a thin
wrapper over [SGLang](https://github.com/sgl-project/sglang) and
[vLLM](https://github.com/vllm-project/vllm), so you can run the same commands on
any GPU host. See [docs/deploy-byo-cloud.md](docs/deploy-byo-cloud.md).

Nothing here is hosted for you. You deploy it to your own account, and the
endpoint URL is yours. There are no API keys, hostnames, or accounts baked into
the code.

## What's in here

The repo is three self-contained projects. Each has its own `_common/` library,
its own `pyproject.toml`, and its own deployments. Work from inside the one you
need.

| Project | What it does | Engine |
|---|---|---|
| [`gemma4/`](gemma4/) | Serve the Gemma 4 family (E2B, E4B, 12B, 26B-A4B, 31B), solo and concurrent, plus a Granite embedding sidecar | SGLang |
| [`qwen/`](qwen/) | Serve Qwen3.6-27B and Qwen3.6-35B-A3B, solo and concurrent, plus a Granite embedding sidecar | SGLang |
| [`pipeline/`](pipeline/) | The research path: serve → score → generate a synthetic corpus → LoRA fine-tune, over the public Chinook SQL agent | vLLM |

Each model ships two shapes:

- **solo** — one user driving a coding harness. The whole GPU and KV cache go to
  one session with a large context window.
- **concurrent** — several agents sharing one GPU with fair-share scheduling and
  a smaller per-session window.

## Models

| Model | Repo | Arch | Native ctx | MTP drafter |
|---|---|---|---|---|
| Gemma 4 E2B-it | `google/gemma-4-E2B-it` | Dense+PLE, 5.1B/~2B | 128K | yes (78M) |
| Gemma 4 E4B-it | `google/gemma-4-E4B-it` | Dense+PLE, 8B/4.5B | 128K | yes (79M) |
| Gemma 4 12B-it | `google/gemma-4-12B-it` | Dense, ~12B | 256K | none published |
| Gemma 4 26B-A4B-it | `google/gemma-4-26B-A4B-it` | MoE, 25B/3.8B active | 256K | yes (0.4B) |
| Gemma 4 31B-it | `google/gemma-4-31B-it` | Dense, 31B | 256K | yes (0.5B) |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` | Dense hybrid | 256K | architectural |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | MoE hybrid, 35B/3B active | 256K | architectural |

All Gemma 4 and Qwen3.6 weights are Apache-2.0 and ungated, so no Hugging Face
token is needed to pull them.

## Quick start (Modal)

You need [uv](https://docs.astral.sh/uv/) and a Modal account. Pick a project and
work from inside it.

```bash
cd gemma4                      # or qwen, or pipeline
uv sync                        # installs the local control plane (modal, openai)
uv run modal token new         # one-time Modal auth

# Deploy a model/shape. The app name is printed, along with a public URL.
uv run modal deploy deployments/12b/solo/serve.py

# Hit the endpoint Modal printed (OpenAI-compatible).
curl $URL/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-it","messages":[{"role":"user","content":"hi"}]}'

# Stop it when you're done so it stops billing.
uv run modal app stop gemma4-12b-solo
```

The endpoint is public the moment you deploy it. That is convenient for testing
and a liability if you leave it up. How to put auth in front of it is your call —
see [docs/securing-endpoints.md](docs/securing-endpoints.md).

## Chat templates

Each family ships the upstream template, a custom harness-friendly fork, a
conformance suite, and a live probe you can point at your own endpoint.

- Gemma 4: [`gemma4/chat_templates/`](gemma4/chat_templates/) — the P1–P5 fork.
  See [README](gemma4/chat_templates/README.md) and
  [TESTING](gemma4/chat_templates/TESTING.md).
- Qwen3.6: [`qwen/chat_templates/`](qwen/chat_templates/) — the Q1–Q8 fork.
  See [README](qwen/chat_templates/README.md) and
  [TESTING](qwen/chat_templates/TESTING.md).

The forks fix edge cases that bite agentic coding harnesses (opencode, pi, and
similar): tool arguments arriving as JSON strings, reasoning getting dropped
across multi-turn tool loops, thinking defaulting off, and a few template bugs.
Each patch is gated, so passing the documented kwargs renders byte-for-byte
identical to upstream. The conformance suites lock that down.

## The research pipeline

[`pipeline/`](pipeline/) is a worked example of taking a base model to a
fine-tuned one, using a SQL agent over the public
[Chinook](https://github.com/lerocha/chinook-database) database as the task:

1. **serve** the model on vLLM,
2. **eval** it on a five-axis tool-use rubric,
3. **corpus** — generate a synthetic SFT corpus with a larger model as the teacher,
4. **sft** — LoRA fine-tune a small model and gate it against capability drift.

## Speculative decoding (research)

[`spec_decode/`](spec_decode/) is a faithful PyTorch **reference harness** for
DeepSeek's [DSpark](https://github.com/deepseek-ai/DeepSpec) speculative-decoding
drafter on Gemma 4. It loads the public draft
([`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7))
and its 12B target and runs one block-7 draft forward on Modal, emitting a JSON
fixture — base + rank-256 Markov logits, confidence, greedy tokens — you can diff
a fast re-implementation against, op-by-op. Writeup:
[docs/dspark-spec-decode.md](docs/dspark-spec-decode.md).

## Running on your own cloud

The serve scripts are Modal wrappers around `python -m sglang.launch_server` and
`vllm serve`. The same image and the same argv run anywhere you have a GPU.
[docs/deploy-byo-cloud.md](docs/deploy-byo-cloud.md) shows how.

## License

Apache 2.0. See [LICENSE](LICENSE). The model weights carry their own licenses
(Apache-2.0 for the Gemma 4 and Qwen3.6 checkpoints used here).

## Attributions

This repository builds on the following projects and research — full credit to
their authors and maintainers:

- **[SGLang](https://github.com/sgl-project/sglang)** — the serving engine behind `gemma4/` and `qwen/`.
- **[vLLM](https://github.com/vllm-project/vllm)** — the serving engine behind `pipeline/`.
- **[Modal](https://modal.com)** — the deployment substrate every project runs on.
- **[Google Gemma](https://ai.google.dev/gemma)** — the Gemma 4 model family (Apache-2.0).
- **[Qwen](https://github.com/QwenLM/Qwen3)** — the Qwen3.6 model family (Apache-2.0).
- **DeepSeek [DeepSpec](https://github.com/deepseek-ai/DeepSpec)** (MIT) and its DSpark drafter — `spec_decode/` is a reference harness for the released checkpoint [`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7).
- **[Chinook](https://github.com/lerocha/chinook-database)** — the public SQL database used as the `pipeline/` fine-tuning task.

Model weights and upstream code remain under their own licenses; this repository
only wraps and runs them.
