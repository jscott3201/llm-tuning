"""Health-check helpers used by serve scripts.

A serve entrypoint launches the inference server as a subprocess and then
needs to know when it is actually ready to take traffic. This module polls
the server's ``/health`` endpoint until it answers 200, while also watching
for the subprocess dying early (a misconfigured launch crashes before it
ever binds the port).

Under Modal's ``@modal.web_server`` decorator the web endpoint only routes
public traffic once the named port is accepting connections, so calling
``wait_for_health`` before blocking on ``proc.wait()`` keeps the cold-start
window honest.
"""

from __future__ import annotations


def wait_for_health(
    proc,
    *,
    timeout_s: int = 1200,
    poll_interval_s: int = 5,
    host: str = "127.0.0.1",
    port: int = 8000,
    label: str = "server",
) -> None:
    """Poll ``http://{host}:{port}/health`` until 200 or timeout.

    The health probe runs *inside* the container, so it dials the loopback
    interface even though the server binds 0.0.0.0 for public ingress;
    0.0.0.0 accepts connections on every interface, including 127.0.0.1.

    Raises ``subprocess.CalledProcessError`` if the server process exits
    before becoming healthy, or ``TimeoutError`` if it never does.
    """
    import subprocess
    import time
    import urllib.request

    start = time.time()
    deadline = start + timeout_s
    url = f"http://{host}:{port}/health"

    while time.time() < deadline:
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

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise TimeoutError(f"[{label}] not ready within {timeout_s}s")


def _redact(args: list[str]) -> list[str]:
    """Strip ``--api-key <value>`` from argv before raising to a crash report."""
    if "--api-key" not in args:
        return list(args)
    out = list(args)
    idx = out.index("--api-key")
    if idx + 1 < len(out):
        out[idx + 1] = "<redacted>"
    return out
