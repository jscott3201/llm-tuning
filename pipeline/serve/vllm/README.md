# Serving the Gemma 4 family on vLLM + Modal

Five Modal serve scripts, one per Gemma 4 size. Each exposes an
OpenAI-compatible `/v1/chat/completions` endpoint behind a PUBLIC
`*.modal.run` URL (the URL printed by `modal deploy`). The rest of the
pipeline consumes these endpoints — score a base model, run a corpus
teacher, re-serve a LoRA-merged adapter.

| Script | Model | Class | GPU (default) | MTP drafter |
| --- | --- | --- | --- | --- |
| `serve_e2b.py` | google/gemma-4-E2B-it | dense (PLE) | L4 | google/gemma-4-E2B-it-assistant (78M) |
| `serve_e4b.py` | google/gemma-4-E4B-it | dense (PLE) | L40S | google/gemma-4-E4B-it-assistant (79M) |
| `serve_12b.py` | google/gemma-4-12B-it | dense | H100 | none published |
| `serve_26b.py` | google/gemma-4-26B-A4B-it | sparse MoE (A4B) | B200 | google/gemma-4-26B-A4B-it-assistant (420M) |
| `serve_31b.py` | google/gemma-4-31B-it | dense | B200 | google/gemma-4-31B-it-assistant (470M) |

GPU classes are the registry defaults (`_common/model_registry.py`) and
are overridable per deploy via the `GPU_OVERRIDE` constant at the top of
each script (where present) or by editing the `gpu=` kwarg. The 12B has
no published drafter; the other four do, but MTP is currently blocked in
vLLM 0.19.1 — see below.

The canonical deployment recipe lives at the
[vLLM Gemma 4 recipe page](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).
Defaults in this folder follow it, with two production-tested deviations
(parser-disabled at concurrency, conservative `max_model_len`) called
out inline.

All five are Apache 2.0 and **not gated** — no Hugging Face token is
required to pull them. The scripts mount
`modal.Secret.from_name("huggingface-secret")` only to get authenticated
Hub rate limits; it's optional and you create that Secret yourself in
your own Modal workspace (it holds `HF_TOKEN`). Comment the secret out
if you don't want to mount a token.

## Deploy

```bash
uv run modal deploy serve/vllm/serve_e4b.py
```

Modal prints a PUBLIC base URL of the form `*.modal.run`. Point any
OpenAI-compatible client at that root (no `/v1` suffix needed; the
client appends `/v1/chat/completions` itself).

The vLLM HTTP server binds `0.0.0.0` (hardcoded in
`_common/vllm_common.build_serve_cmd`) so Modal's web ingress can reach
it — do not change the host to `127.0.0.1` under Modal or routing fails.

## Auth — your choice

Endpoints are **public by default**. This folder does not implement
auth. Two ways to lock one down, neither baked in:

- **Modal proxy auth (recommended):** pass `requires_proxy_auth=True`
  to the `@modal.web_server(...)` decorator. Modal enforces it at the
  ingress before any container spins up. See
  [modal.com/docs/guide/webhook-proxy-auth](https://modal.com/docs/guide/webhook-proxy-auth).
- **vLLM `--api-key`:** `build_serve_cmd` has an optional `api_key_env`
  hook (default off). It emits `--api-key` only if the named env var is
  set — e.g. via a Modal Secret. Left off by default.

## Tear-down

```bash
uv run modal app stop gemma4-e4b-solo
```

App names per script:

| Script | App name |
| --- | --- |
| `serve_e2b.py` | `gemma4-e2b-solo` |
| `serve_e4b.py` | `gemma4-e4b-solo` |
| `serve_12b.py` | `gemma4-12b-concurrent` |
| `serve_26b.py` | `gemma4-26b-concurrent` |
| `serve_31b.py` | `gemma4-31b-solo` |

Volumes (`gemma4-hf-cache`, `torchinductor-cache`) persist across
stop/deploy cycles. Remove them with `modal volume rm` if you want to
evict cached weights and torch.compile artifacts.

## SGLang serving lives elsewhere

There is intentionally **no `sglang/` subfolder here**. For SGLang
production serving of Gemma 4 — including the MTP path that explicitly
supports multimodal targets (SGLang's `FROZEN_KV_MTP` algorithm) and
dynamic LoRA hot-swap via REST — see the dedicated `gemma4/` project.
This folder is the vLLM runtime only.

## The Gemma 4 attention-backend constraint (triton)

Gemma 4 fixes `head_dim=256` (explicit in config, **not**
`hidden_size / num_attention_heads`) and uses a hybrid local/global
attention pattern: sliding-window layers interleaved with full-attention
layers (5:1 on the 12B). That combination is incompatible with vLLM's
default flash kernels; the supported path is the **triton** attention
backend.

The registry marks every Gemma 4 size `requires_triton_attention=True`.
The 12B serve script pins it explicitly on the serve subprocess:

```python
env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
```

so vLLM does not silently fall back to a flash kernel that can't handle
`head_dim=256` / the hybrid pattern. If you change GPU class or vLLM
version and hit an attention-backend error or a flash-kernel fallback
warning on any size, set the same env var on that script's subprocess.

## Multi-token prediction (MTP) / speculative decoding — currently broken in vLLM 0.19.1

The E2B, E4B, 26B-A4B, and 31B sizes each ship a tiny companion drafter
under `google/gemma-4-{size}-it-assistant` (78M-470M params). The 12B
has **no published drafter** (`google/gemma-4-12B-it-assistant` does not
exist and the config has no MTP / `num_nextn_predict` head), so MTP on
12B would require a separately sourced or trained draft model.

The Google MTP work and the vLLM Gemma 4 recipe both pitch up to ~3×
throughput at zero quality cost via vLLM's `--speculative-config`. In
practice, **vLLM 0.19.1 raises `NotImplementedError` when you try to
enable it** for any Gemma 4 size. The path that fails:

```
File "vllm/v1/spec_decode/eagle.py", line 290, in _raise_if_multimodal
    raise NotImplementedError(
NotImplementedError: Speculative Decoding with draft models or
parallel drafting does not support multimodal models yet
```

All published checkpoints are `Gemma4*ForConditionalGeneration` —
vLLM's multimodal class — even when you only ever feed text. The
spec-decode code path checks `config.architectures` and refuses to wrap
a multimodal target.

The `ENABLE_MTP = False` flags in the serve scripts stay False until
upstream lands the fix. The drafter repos and `speculative_tokens`
numbers in `_common/model_registry.py` are still captured so the patch
is a one-line flip when vLLM ships the multimodal-target spec-decode
path. The `--speculative-config` argument *does* fire correctly via
`build_serve_cmd` — the failure is upstream, in vLLM's engine init.

For an MTP path that works on Gemma 4 today, see the SGLang serving in
the `gemma4/` project.

## The undocumented gotchas

Three real footguns from the vLLM Gemma 4 issue tracker. Each is flagged
in the relevant serve script and linked back to the upstream issue.

### 1. `<pad>` tokens under concurrent tool calls (vLLM #39392)

The `--tool-call-parser gemma4` flag has shared mutable state across
requests. Under concurrent tool-use traffic the parser leaks `<pad>`
tokens into the response stream — the response is otherwise valid, but
the tool-call body comes back garbled. Status in 0.19.1 is ambiguous;
the conservative play is to disable it at concurrency.

**Mitigation**: the 26B serve script (`serve_26b.py`) runs the parser
**disabled** because it's the corpus-generator endpoint at 20-64-way
concurrent tool-use traffic. The raw `<|tool_call>...` tokens come
through verbatim in the content field, and `_common/gemma4_parser.py`
extracts them client-side. The 12B turns the parser on by default
(moderate concurrency) but documents the same flip. The smaller serve
scripts turn the parser on because their probe traffic is single-stream
and the bug doesn't fire.

### 2. `enable_thinking=false` silently bypasses xgrammar (vLLM #39130)

`--reasoning-parser gemma4` plus `enable_thinking=false` plus a
`response_format` constraint silently skips the xgrammar structured-
output path. Generations come back unconstrained — no error, no log
line, just unconstrained output.

**Mitigation**: set `enable_thinking=true` per-request via
`extra_body.chat_template_kwargs.enable_thinking` (or pin the server
default with `--default-chat-template-kwargs`, which `build_serve_cmd`
exposes via the `default_thinking` kwarg).

### 3. Infinite repetition under structured output (vLLM #40080)

Gemma 4 31B and 26B-A4B can fall into infinite repetition loops under
JSON-schema-constrained generation, especially with free-form string
fields. Root cause is an interaction between xgrammar's bitmask
restriction and Gemma 4's attention pattern.

**Mitigation**: set `repetition_penalty >= 1.05` and/or
`frequency_penalty >= 0.5` on the request. Constrain string fields via
regex in the schema where possible. Keep `enable_thinking=true` to
ensure xgrammar is actually firing (see gotcha #2).

## Where the shared code lives

- `_common/vllm_common.py` — Modal image builder, `vllm serve` argv
  builder (binds `0.0.0.0:8000`), and the `wait_for_health` poll that
  keeps `@modal.web_server` bring-up honest.
- `_common/model_registry.py` — short-name → HF repo + GPU + sizing
  hints (`max_model_len`, concurrency, `head_dim=256`,
  `requires_triton_attention`, `kv_cache_dtype`). The five serve scripts
  in this folder are thin wrappers around it.
- `_common/gemma4_parser.py` — client-side raw `<|tool_call>...` token
  parser, used wherever the server-side parser is disabled.
