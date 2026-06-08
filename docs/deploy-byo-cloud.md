# Running on your own cloud

Modal is what these scripts target, but it isn't load-bearing. Every serve
script does the same thing: build a pinned container image and run a standard
inference server inside it. The image and the command don't depend on Modal, so
you can run them on any GPU you can get a container onto — a bare host, Docker,
Kubernetes, RunPod, Lambda, a cloud VM, whatever.

## What the serve scripts actually run

For the SGLang projects (`gemma4/`, `qwen/`):

- **Image:** the pinned `lmsysorg/sglang` tag in `_common/sglang_common.py`
  (`SGLANG_TAG`), plus a small layer of Python deps and the baked chat templates.
- **Command:** `python -m sglang.launch_server` with an argv that
  `build_serve_cmd()` assembles from the constants at the top of the serve
  script.

For the pipeline (`pipeline/`), the image is built on the pinned vLLM version and
the command is `vllm serve ...`, assembled the same way by
`pipeline/_common/vllm_common.py`.

The only Modal-specific pieces are the decorators (`@app.cls`, `@modal.enter`,
`@modal.web_server`) and the memory-snapshot lifecycle. The model command
underneath is portable.

## See the exact command

`build_serve_cmd()` returns a plain `list[str]`. Print it to get the argv your
deployment would run:

```python
from _common.model_registry import get
from _common.sglang_common import build_serve_cmd, MTP_NEXTN_STANDARD

spec = get("31b")
cmd = build_serve_cmd(
    model_path=spec.hf_repo,
    served_model_names=["gemma-4-31b-it"],
    max_model_len=196_608,
    attention_backend="triton",       # required for Gemma 4
    kv_cache_dtype="fp8_e5m2",
    chat_template="/opt/templates/custom_pub_chat_template_gemma4.jinja",
    host="0.0.0.0",
    port=8000,
)
print(" ".join(cmd))
```

## Run it under Docker

```bash
# Pull the same SGLang image the serve scripts pin (check SGLANG_TAG).
docker run --gpus all -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $PWD/gemma4/chat_templates:/opt/templates:ro \
  lmsysorg/sglang:<SGLANG_TAG> \
  python -m sglang.launch_server \
    --model-path google/gemma-4-31B-it \
    --host 0.0.0.0 --port 8000 \
    --attention-backend triton \
    --kv-cache-dtype fp8_e5m2 \
    --reasoning-parser gemma4 --tool-call-parser gemma4 \
    --chat-template /opt/templates/custom_pub_chat_template_gemma4.jinja
```

Bind `0.0.0.0` so the container is reachable, mount a Hugging Face cache so you
don't re-download weights on every restart, and mount the chat templates the
command references. The flags are the same ones the serve script passes; copy
them from the printed `build_serve_cmd()` output for the model and shape you want.

## GPU sizing

The registry (`_common/model_registry.py`) records the GPU class each model was
tuned for:

| Model | Solo | Notes |
|---|---|---|
| Gemma 4 E2B / E4B | L4 / L40S | small, cost-effective |
| Gemma 4 12B | H100 80GB | ~24 GiB weights + 256K KV headroom |
| Gemma 4 26B-A4B | 2× B200 | MoE, tensor-parallel TP=2 |
| Gemma 4 31B | B200 | ~62–66 GiB weights |
| Qwen3.6-27B | B200 | ~54 GiB weights |
| Qwen3.6-35B-A3B | B200 | MoE, TP=1, ~67 GiB weights |

## Ingress and auth

Off Modal, you own the front door. Put the server behind a reverse proxy (nginx,
Caddy, a cloud load balancer, or a Kubernetes Ingress) and add your own auth
there, or use SGLang/vLLM's `--api-key`. See
[securing-endpoints.md](securing-endpoints.md) for the auth options — the
API-key approach works identically off Modal.

## What you give up

Modal's memory snapshots cut cold-start time by skipping CUDA-graph capture and
warmup. That's Modal-specific. Elsewhere you manage the process lifecycle
yourself: keep the server warm, or wear the full cold start on each restart.
Everything about correctness — the flags, the parsers, the chat template, the
attention backend — is identical wherever you run it.
