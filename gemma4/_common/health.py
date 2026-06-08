"""Health-check + snapshot-lifecycle helpers used by serve scripts.

Modal needs to know when the inference server is actually serving so it
can route requests. Returning too early from the entrypoint sends traffic
to a dead socket; returning too late wedges the container in cold-start.
This module threads that needle by polling /health.

The SGLang server binds ``0.0.0.0`` so Modal's web endpoint can reach it,
but these helpers run inside the same container and connect over the
loopback interface, so they default to ``127.0.0.1``.

Also provides the warmup + release/resume helpers that drive Modal's
Memory Snapshot lifecycle:

  - ``send_warmup_request()`` triggers CUDA-graph capture and any
    first-request JIT compilation so the snapshot captures the
    post-warmup state.
  - ``release_memory_occupation()`` calls SGLang's REST endpoint to move
    weights / KV-cache off the GPU before the snapshot is taken.
  - ``resume_memory_occupation()`` calls the inverse endpoint post-restore
    so the GPU is repopulated before traffic arrives.

References
----------
- Modal SGLang snapshot example: https://modal.com/docs/examples/sglang_snapshot
- SGLang memory-saver mode: --enable-memory-saver / --enable-weights-cpu-backup
"""

from __future__ import annotations


def wait_for_health(
    proc=None,
    *,
    timeout_s: int = 1200,
    poll_interval_s: int = 5,
    host: str = "127.0.0.1",
    port: int = 8000,
    label: str = "server",
) -> None:
    """Block until SGLang's /health endpoint returns 200, or raise.

    Args:
        proc: Optional subprocess.Popen handle. If provided, the function
            checks ``proc.poll()`` each iteration and raises
            CalledProcessError if the subprocess exits before /health goes
            ready. Useful when polling from the same process that spawned
            SGLang.
        timeout_s: Hard deadline.
        host/port: Where /health is reached. Defaults to the loopback
            interface (the server itself binds 0.0.0.0, reachable as
            localhost from inside the container).
        label: Used in log lines.
    """
    import subprocess
    import time
    import urllib.request

    start = time.time()
    deadline = start + timeout_s
    url = f"http://{host}:{port}/health"

    while time.time() < deadline:
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                raise subprocess.CalledProcessError(rc, cmd=_redact(proc.args))
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    elapsed = int(time.time() - start)
                    print(f"[{label}] ready after {elapsed}s", flush=True)
                    return
        except Exception:
            pass
        time.sleep(poll_interval_s)

    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    raise TimeoutError(f"[{label}] not ready within {timeout_s}s")


def send_warmup_request(
    *,
    model: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_tokens: int = 32,
    timeout_s: float = 120.0,
) -> None:
    """Send one inference request to populate CUDA graphs + caches.

    Called pre-snapshot from ``@modal.enter(snap=True)`` so the captured
    state includes a fully-warmed server (CUDA graphs captured, any
    first-request JIT compilation done, the SGLang scheduler in its
    steady-state batch shape).

    Uses /v1/chat/completions so the gemma4 tool-call + reasoning parsers
    also exercise their first-pass init.
    """
    import json
    import urllib.request

    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"warmup request failed: HTTP {resp.status}")
    print("[warmup] CUDA graphs + caches populated", flush=True)


def release_memory_occupation(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_s: float = 60.0,
) -> None:
    """Tell SGLang to move weights/KV-cache off the GPU before snapshot.

    Requires ``--enable-memory-saver`` to have been passed to
    launch_server. Called from ``@modal.enter(snap=True)`` AFTER warmup but
    BEFORE the Modal snapshot is taken — the snapshot then captures a
    smaller, GPU-empty state that restores faster.
    """
    import urllib.request

    url = f"http://{host}:{port}/release_memory_occupation"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        if resp.status >= 400:
            raise RuntimeError(
                f"release_memory_occupation failed: HTTP {resp.status}"
            )
    print("[snapshot] released GPU memory occupation", flush=True)


def resume_memory_occupation(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_s: float = 120.0,
) -> None:
    """Tell SGLang to repopulate GPU memory after a snapshot restore.

    Called from ``@modal.enter(snap=False)``. SGLang's CPU-backed weights
    (``--enable-weights-cpu-backup``) get moved back to the GPU; the
    server resumes accepting requests.
    """
    import urllib.request

    url = f"http://{host}:{port}/resume_memory_occupation"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        if resp.status >= 400:
            raise RuntimeError(
                f"resume_memory_occupation failed: HTTP {resp.status}"
            )
    print("[snapshot] resumed GPU memory occupation", flush=True)


def _redact(args: list[str]) -> list[str]:
    """Strip ``--api-key <value>`` from argv before raising to crash report."""
    if "--api-key" not in args:
        return list(args)
    out = list(args)
    idx = out.index("--api-key")
    if idx + 1 < len(out):
        out[idx + 1] = "<redacted>"
    return out
