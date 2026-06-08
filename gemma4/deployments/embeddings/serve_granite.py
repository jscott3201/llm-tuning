"""IBM Granite embedding sidecar on Modal (vLLM, OpenAI /v1/embeddings).

This sidecar is framework- and model-independent: it serves a dense
retrieval embedding model, not the main chat LLM. It is colocated with
the chat-model deployments only as a convenience for RAG pipelines that
want both an embedder and a generator on the same Modal account.

Model: ibm-granite/granite-embedding-311m-multilingual-r2
  - 311M params (ModernBert), 8K max context, 768-dim output
  - Matryoshka: truncatable to 768/512/384/256/128 with graceful loss
  - 200+ languages broad coverage, 52 with deeper retrieval-pair training
  - CLS pooling + L2 (activation) normalization at inference time
  - Apache-2.0; ungated, so no HF token is required to pull weights.

Runtime: vLLM in pooling/embed mode (``--runner pooling --convert embed``).
  - vLLM 0.19+ recognises ModernBert as an embedding architecture.
  - Exposes the OpenAI-compatible ``POST /v1/embeddings`` endpoint plus a
    ``GET /health`` probe.
  - Continuous batching + paged attention give high throughput on the
    short, bursty requests typical of an indexing or query workload.

Matryoshka output dimensions
----------------------------
The model is trained with Matryoshka Representation Learning, so a single
768-dim vector can be truncated (and re-normalized) to a shorter prefix
with graceful quality loss. We declare the supported lengths to vLLM via
``--hf-overrides '{"matryoshka_dimensions": [...]}'``; clients then request
a specific length per call with the OpenAI ``dimensions`` parameter, e.g.::

    curl <the URL printed by modal deploy>/v1/embeddings \\
      -H 'Content-Type: application/json' \\
      -d '{"model": "...", "input": "hello", "dimensions": 256}'

Omitting ``dimensions`` returns the full 768-dim vector.

GPU: A100-40 class. Embedding workloads are throughput-bound at small
per-token compute, so A100s give the best $/throughput for a 311M model.
Scale horizontally (more replicas) rather than vertically.

Networking
----------
The vLLM server binds ``0.0.0.0`` so Modal's web endpoint can reach it.
``@modal.web_server(port=8000, ...)`` publishes a public ``*.modal.run``
URL that forwards to the container's port 8000; that URL is the intended
ingress. There is no built-in auth — see the note on the decorator below.
"""

from __future__ import annotations

import modal

from _common.health import wait_for_health

# Descriptive app name only — no personal/workspace identifiers. Rename
# freely; this is just the label Modal lists the deployment under.
APP_NAME = "granite-embed"
SERVE_PORT = 8000

EMBED_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
EMBED_REVISION: str | None = None  # pin to a commit SHA after first deploy

# Matryoshka output lengths this model supports (longest first). Declared
# to vLLM via --hf-overrides; clients pick one per request with the OpenAI
# `dimensions` field. 768 is the native (full) embedding size.
MATRYOSHKA_DIMENSIONS = [768, 512, 384, 256, 128]

# Per-replica concurrency target. Embedding requests are short, so many
# can be in flight before the GPU saturates. 32 is a reasonable starting
# point; bench against observed GPU utilization and tune.
TARGET_CONCURRENT_INPUTS = 32

# GPU class. A100-40 is the price/perf sweet spot for a 311M embedder.
# Bump to A100-80 if your batch sizes routinely exceed what 40 GiB holds;
# jump to H100 only if you find yourself memory-bandwidth-bound.
EMBED_GPU = "A100-40GB"

# vLLM image. Kept separate from any SGLang chat-model image since the two
# can carry divergent torch / CUDA pins. uv_pip_install is used for the
# build-time speedup Modal recommends.
embed_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.19.1",
        "transformers>=5.5.0",
        "huggingface_hub[hf_xet]>=1.11",
        "hf-transfer>=0.1.8",
        "sentence-transformers>=3.0",  # Matryoshka helper + tokenizer compat
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Steer the HF weight cache onto the Volume-mounted path so a
            # redeploy reuses the download instead of re-pulling weights.
            "HF_HUB_CACHE": "/modal-cache/huggingface",
        }
    )
    # add_local_python_source MUST be the LAST build step: Modal forbids
    # further image mutations after a non-copy local-file addition. This is
    # what makes `from _common.health import ...` importable in-container.
    .add_local_python_source("_common")
)


app = modal.App(APP_NAME)

# Embedding cache volume, kept separate from any chat-model HF cache so a
# ~1 GiB embedder and multi-GiB LLM weights don't evict each other.
# Generic name — safe to deploy on any Modal account.
embed_hf_cache = modal.Volume.from_name(
    "granite-embed-hf-cache", create_if_missing=True
)


@app.function(
    image=embed_image,
    gpu=EMBED_GPU,
    volumes={"/modal-cache/huggingface": embed_hf_cache},
    # HF token is OPTIONAL for this model (Apache-2.0, ungated). Uncomment
    # the line below only if you later swap in a gated embedder; it expects
    # a Modal Secret you create yourself named "huggingface-secret"
    # (containing HF_TOKEN). See https://modal.com/docs/guide/secrets .
    # secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
    scaledown_window=60 * 10,
    # Embeddings are stateless — scale OUT, not up. Cap at 4 replicas by
    # default to bound cost on bursty indexing jobs; raise if your RAG
    # pipeline needs more throughput.
    max_containers=4,
)
@modal.concurrent(max_inputs=TARGET_CONCURRENT_INPUTS)
# Binds 0.0.0.0 inside serve() so this web endpoint can route to it; Modal
# publishes a public *.modal.run URL. No auth is baked in. To restrict
# access, add Modal proxy-auth tokens or your own gate — see
# https://modal.com/docs/guide/webhook-proxy-auth .
@modal.web_server(port=SERVE_PORT, startup_timeout=60 * 10)
def serve() -> None:
    """Spawn vLLM in embed mode and block until it is serving."""
    import json
    import subprocess

    cmd = [
        "vllm",
        "serve",
        EMBED_MODEL,
        # Modern vLLM pooling-runner form (replaces the older --task embed).
        # --convert embed is explicit; ModernBert is auto-detected as an
        # embedding architecture, but stating it documents intent.
        "--runner",
        "pooling",
        "--convert",
        "embed",
        # Bind 0.0.0.0 (NOT 127.0.0.1) so Modal's @modal.web_server can
        # reach the process on the container's external interface. The
        # in-container /health probe still dials loopback, which 0.0.0.0
        # also accepts.
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVE_PORT),
        # Granite embedding models use CLS pooling with L2 (activation)
        # normalization. These match the Sentence-Transformers config the
        # model ships, but we pass them explicitly as a defence against
        # vLLM default drift across releases.
        "--pooler-config",
        json.dumps({"pooling_type": "CLS", "normalize": True}),
        # Matryoshka: declare the supported truncation lengths so clients
        # can request a shorter vector via the OpenAI `dimensions` field.
        "--hf-overrides",
        json.dumps({"matryoshka_dimensions": MATRYOSHKA_DIMENSIONS}),
        # 8192 is this model's max sequence length.
        "--max-model-len",
        "8192",
        # The model is small; headroom goes to request batching rather
        # than a large KV cache.
        "--gpu-memory-utilization",
        "0.85",
        # Higher batch ceiling for embedding throughput. Tune against the
        # GPU utilization you observe in production.
        "--max-num-seqs",
        "256",
    ]
    if EMBED_REVISION:
        cmd += ["--revision", EMBED_REVISION]

    proc = subprocess.Popen(cmd)
    # Block until /health is 200 so Modal doesn't route to a dead socket.
    wait_for_health(proc, timeout_s=60 * 10, port=SERVE_PORT, label=APP_NAME)
    # Keep the function alive for the life of the server; if vLLM exits,
    # the container exits and Modal reschedules.
    proc.wait()
