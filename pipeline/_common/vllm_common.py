"""Shared Modal image + `vllm serve` command builder + health poll.

Every vLLM serve script (E2B / E4B / 12B / 26B / 31B) imports from here
so the per-deploy file only carries the genuine differences:
GPU class, context window, concurrency, tool-parser on/off.

Ingress model: each serve script wraps `vllm serve` in a Modal function
decorated with `@modal.web_server(port=8000, startup_timeout=...)`,
which publishes a PUBLIC `*.modal.run` URL (use the URL printed by
`modal deploy`). For that to route, the vLLM HTTP server MUST bind the
external interface `0.0.0.0` (the builder below hardcodes that), NOT
`127.0.0.1`. Endpoints are public by default; to require auth, pass
`requires_proxy_auth=True` to the web decorator — see Modal's
endpoint-security docs (modal.com/docs/guide/webhook-proxy-auth). This
module does not implement auth.

The defaults follow the canonical vLLM Gemma 4 recipe at
https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
with two production-tested deviations documented inline:

  - Tool-call parser is OPT-IN (default ON for low-concurrency, OFF
    for a high-concurrency corpus-generator endpoint). vLLM #39392
    reported `<pad>` token leakage under concurrent tool-call traffic
    via `--tool-call-parser gemma4`. Status in 0.19.1 is ambiguous (the
    release notes mention concurrent-correctness work) but the
    mitigation — disable the server-side parser at concurrency, parse
    raw `<|tool_call>...` tokens client-side via `_common.gemma4_parser`
    — costs nothing else and keeps train/serve token shapes identical.
  - `--reasoning-parser gemma4` is always on. There is also a
    server-side default flag
    `--default-chat-template-kwargs '{"enable_thinking": false}'`
    (verified in the recipe and on the Qwen/DeepSeek family); we
    expose it via the `default_thinking` kwarg below so each deploy
    can choose what shows up if the client doesn't pass
    `chat_template_kwargs.enable_thinking` itself. Per-request
    `extra_body.chat_template_kwargs.enable_thinking` always wins.

Two related gotchas you may hit but they don't change the build:

  - vLLM #39130: combining `--reasoning-parser gemma4` with
    `enable_thinking=false` AND a `response_format` constraint silently
    skips xgrammar (no error). Set `enable_thinking=true` per-request
    when grammar enforcement matters.
  - vLLM #40080: Gemma 4 31B and 26B-A4B can fall into infinite
    repetition under structured output (especially JSON schema with
    free-form string fields). Mitigate with `repetition_penalty` /
    `frequency_penalty` on the request, or stage the corpus generator
    around shorter, simpler schemas.

Imported inside the Modal container, so every script that uses this
module must add it to the image's local Python sources via
`.add_local_python_source("_common")` (or include the parent path on
`PYTHONPATH`).
"""

from __future__ import annotations

import modal


# ─────────────────────────────────────────────────────────────────────
# Pinned versions
# ─────────────────────────────────────────────────────────────────────

VLLM_VERSION = "0.19.1"
"""Canonical vLLM pin. Gemma 4 landed in PR #38826 (vLLM 0.19.0); the
`gemma4` tool-call + reasoning parsers require >= 0.19. 0.19.1 is a
patch release on 0.19.0 (Apr 2026) bundling Transformers >= 5.5.3 and
Gemma 4 bugfixes including #38847 (parser missing `tools` parameter).
Latest at the time of writing is 0.20.1; bump cautiously after
verifying #39392 / #39130 / #40080 status against the chosen release."""

DEFAULT_MAX_MODEL_LEN = 16_384
"""vLLM Gemma 4 recipe canonical context window. Smaller than the
model's native window (128K-256K depending on size) so KV cache has
room for concurrent requests without backing them off into eviction
churn."""


# ─────────────────────────────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────────────────────────────


def make_vllm_image(vllm_version: str = VLLM_VERSION) -> modal.Image:
    """Canonical vLLM image: Debian slim + uv + hf-transfer.

    `uv pip install` resolves the CUDA wheel graph an order of magnitude
    faster than vanilla pip, which matters every time the image rebuilds.
    `hf-transfer` saturates Modal's egress when pulling the first set of
    weights — first-boot 50+ GiB downloads (the 26B-A4B and 31B cases)
    drop from ~10 min to ~3 min once it's enabled.

    Transformers 5.5.0+ is required for the `gemma4` model class; older
    versions can't load the architecture. We pin a floor and let pip
    resolve to the latest compatible patch — vLLM 0.19.1 ships its own
    Transformers pin which usually wins anyway.

    The `add_local_python_source` call mounts the `_common` package into
    the container so the per-deploy serve scripts can import it. Every
    serve script tags this image via `make_vllm_image().add_local_python_source(...)`
    if they need extra modules.

    These Gemma 4 checkpoints are ungated/public, so no HF token is
    required to pull them. If you point a deploy at a gated repo, attach
    one with `modal.Secret.from_name("huggingface-secret")` (a secret you
    create in your own Modal workspace holding `HF_TOKEN`).
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            f"vllm=={vllm_version}",
            "transformers>=5.5.0",
            "huggingface_hub[hf_xet]>=1.11",
            "hf-transfer>=0.1.8",
        )
        .env(
            {
                # Saturate Modal egress on the first 50+ GiB pull.
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_python_source("_common")
    )


# ─────────────────────────────────────────────────────────────────────
# Serve command builder
# ─────────────────────────────────────────────────────────────────────


def build_serve_cmd(
    model_path: str,
    served_model_names: list[str],
    *,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.92,
    max_num_batched_tokens: int = 16_384,
    enable_tool_call_parser: bool = True,
    enable_prefix_caching: bool = True,
    enable_async_scheduling: bool = True,
    fast_boot: bool = False,
    default_thinking: bool | None = None,
    speculative_config: dict | None = None,
    kv_cache_dtype: str | None = None,
    max_num_seqs: int | None = None,
    api_key_env: str | None = "API_KEY",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Assemble the `vllm serve` argv for a Gemma 4 endpoint.

    Defaults match the vLLM Gemma 4 recipe (`max_model_len=16384`,
    `gpu_memory_utilization=0.92`). Per-deploy tuning (GPU class,
    concurrency, thinking default) lives in the caller.

    The server always binds `--host 0.0.0.0 --port 8000` so the
    enclosing `@modal.web_server(port=8000, ...)` can reach it and
    publish the public `*.modal.run` URL — do not change the host to
    127.0.0.1 under Modal or routing fails.

    Notable knobs:

    - `enable_tool_call_parser` toggles `--tool-call-parser gemma4 +
      --enable-auto-tool-choice`. Default True for low-concurrency
      probe and eval workloads. Flip OFF for any deployment running
      multiple concurrent tool-call requests — vLLM #39392's shared
      mutable state can leak `<pad>` tokens into responses under
      concurrency.
    - `--reasoning-parser gemma4` is always on. Combine with
      `default_thinking` to pin the server-wide default for
      `chat_template_kwargs.enable_thinking`; per-request
      `extra_body.chat_template_kwargs.enable_thinking` always wins.
    - `--async-scheduling` matches the recipe; default in 0.19 but
      passing it explicitly documents the choice in the launched argv.
    - `fast_boot=True` skips torch.compile, dropping cold-boot ~30s but
      losing ~30% sustained throughput. Useful for one-shot probes;
      keep off for production-sized runs.
    - `default_thinking`: when set, emits
      `--default-chat-template-kwargs '{"enable_thinking": <bool>}'`
      so unconditional clients (no `chat_template_kwargs` in the
      request body) get a predictable behaviour. Leave None to inherit
      the model's template default. Note: when `enable_thinking=false`
      and structured-output (`response_format`) is also requested,
      vLLM #39130 silently bypasses xgrammar — set
      `default_thinking=True` if you need grammar enforcement to fire.
    - `speculative_config`: enables multi-token-prediction (MTP) /
      speculative decoding via `--speculative-config '<json>'`. Pass
      `{"model": "google/gemma-4-E4B-it-assistant",
        "num_speculative_tokens": 4}` (numbers from `model_registry`
      per size). Reported throughput gain is up to ~3× on dense
      models; MoE (26B-A4B) has batch-size constraints due to
      expert-loading overhead — benchmark before enabling for
      high-concurrency workloads. Quality is identical to the target
      because verification is exact (Leviathan et al., 2022).
    - `kv_cache_dtype`: emits `--kv-cache-dtype <value>`. Set to "fp8"
      to halve KV cache footprint at a tiny accuracy cost — useful
      when long-context throughput is bottlenecked by KV cache.
    - `max_num_seqs`: emits `--max-num-seqs <n>`. The recipe's tuning
      cheatsheet: 256-512 for max throughput, 8-16 for min latency,
      128 balanced. Modal's `@modal.concurrent` already caps inbound
      concurrency, so leave None unless you need finer control over
      vLLM's batch shape.
    - `api_key_env` names an environment variable the container reads
      to OPTIONALLY gate the endpoint with `--api-key`. It is off
      unless that env var is set (e.g. via `modal.Secret.from_name(...)`).
      Auth is your choice — Modal endpoints are public by default and
      can instead be locked down at the ingress with proxy auth; see
      modal.com/docs/guide/webhook-proxy-auth.
    - `extra_args`: forwarded verbatim. Use this for size-specific
      flags like `--tensor-parallel-size N`, `--mm-processor-kwargs`,
      `--limit-mm-per-prompt`, `--chat-template <path>`, or LoRA
      serving via `--enable-lora --lora-modules name=hf-repo
      --max-lora-rank N`. vLLM 0.19.1 ships LoRA for the Gemma 4
      language backbone (PR #39291); vision + audio tower LoRA are
      deferred follow-up work.
    """
    import json
    import os

    cmd: list[str] = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        *served_model_names,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--reasoning-parser",
        "gemma4",
    ]

    if enable_tool_call_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "gemma4"]

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if enable_async_scheduling:
        cmd.append("--async-scheduling")

    if fast_boot:
        cmd.append("--enforce-eager")

    if default_thinking is not None:
        cmd += [
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": bool(default_thinking)}),
        ]

    if speculative_config is not None:
        cmd += ["--speculative-config", json.dumps(speculative_config)]

    if kv_cache_dtype is not None:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]

    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]

    if api_key_env and os.environ.get(api_key_env):
        cmd += ["--api-key", os.environ[api_key_env]]

    if extra_args:
        cmd += extra_args

    return cmd


# ─────────────────────────────────────────────────────────────────────
# Health poll
# ─────────────────────────────────────────────────────────────────────


def wait_for_health(
    proc,
    *,
    timeout_s: int = 600,
    poll_interval_s: int = 5,
    label: str = "vllm",
) -> None:
    """Poll http://127.0.0.1:8000/health until vLLM answers 200.

    `@modal.web_server` requires the decorated function to RETURN
    after bring-up so Modal marks the replica live and starts routing
    traffic. Returning too early (before vLLM's HTTP server is up) sends
    requests at a dead socket; returning too late wedges the container
    in cold-start. This helper threads that needle: it polls the local
    health endpoint, raises `CalledProcessError` if the subprocess
    dies during warm-up, and raises `TimeoutError` if the deadline
    passes (terminating the subprocess on its way out).

    Note the poll target is `127.0.0.1` (the loopback inside the
    container) even though the server binds `0.0.0.0` — the process is
    checking its own local socket, while Modal's ingress reaches the
    server over the `0.0.0.0` interface.
    """
    import subprocess
    import time
    import urllib.request

    start = time.time()
    deadline = start + timeout_s
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            # Redact `--api-key <value>` from `proc.args` before
            # surfacing — a failed warm-up that happens to have a secret
            # mounted must not leak the key into Modal's crash report.
            raise subprocess.CalledProcessError(rc, cmd=_redact_args(proc.args))
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/health", timeout=5,
            ) as resp:
                if resp.status == 200:
                    elapsed = int(time.time() - start)
                    print(f"[{label}] ready after {elapsed}s", flush=True)
                    return
        except Exception:
            # HTTPError, URLError, ConnectionRefused are all expected
            # during warm-up — the server isn't bound yet.
            pass
        time.sleep(poll_interval_s)

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise TimeoutError(
        f"{label} did not become ready within {timeout_s}s",
    )


def _redact_args(args: list[str]) -> list[str]:
    """Return a copy of argv with the value after `--api-key` redacted.

    Used on error paths that surface `proc.args` so an exception stack
    never carries a plaintext secret. If the flag appears without a
    following value (malformed argv) the original list is returned
    unchanged — we'd rather surface a real bug than smear over it.
    """
    if "--api-key" not in args:
        return list(args)
    redacted = list(args)
    idx = redacted.index("--api-key")
    if idx + 1 < len(redacted):
        redacted[idx + 1] = "<redacted>"
    return redacted
