"""Bounded local-process and raw remote-stream ownership for QAT preflight."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal


async def finish(awaitable, *, propagate=False, on_cancel=None):
    """Join owned cleanup despite repeated cancellation of its caller."""
    task = asyncio.ensure_future(awaitable)
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                raise
            interrupted = True
            if on_cancel:
                on_cancel()
            asyncio.current_task().uncancel()
    if interrupted and propagate:
        raise asyncio.CancelledError()
    return result


async def collect_remote(process, deadline, *, stdout_limit=32768, stderr_limit=8192):
    """Require remote completion and both byte-stream EOFs; never use read-all."""
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    complete = {"stdout": False, "stderr": False}
    reason = None
    code = None

    async def drain(name, limit):
        iterator = aiter(getattr(process, name))
        try:
            async for chunk in iterator:
                if not isinstance(chunk, bytes):
                    raise ValueError("nonbinary_stream")
                remaining = limit - len(buffers[name])
                buffers[name].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    raise ValueError("output_limit")
            complete[name] = True
        finally:
            close = getattr(iterator, "aclose", None)
            if close:
                await close()

    tasks = [asyncio.create_task(drain("stdout", stdout_limit)),
             asyncio.create_task(drain("stderr", stderr_limit)),
             asyncio.create_task(process.wait())]
    try:
        async with asyncio.timeout_at(deadline):
            await asyncio.gather(*tasks)
        code = tasks[2].result()
    except TimeoutError:
        reason = "remote_timeout"
    except ValueError as exc:
        reason = str(exc) if str(exc) in {"output_limit", "nonbinary_stream"} else "remote_error"
    except asyncio.CancelledError:
        raise
    except Exception:
        reason = "remote_error"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await finish(asyncio.gather(*tasks, return_exceptions=True), propagate=True)
    if code is None and tasks[2].done() and not tasks[2].cancelled() and tasks[2].exception() is None:
        code = tasks[2].result()
    return {"reason": reason, "returncode": code, "complete": complete,
            "stdout": bytes(buffers["stdout"]), "stderr": bytes(buffers["stderr"])}


def evaluate_capture(capture):
    """Accept the tracked validator's full passing contract, not just exit zero."""
    summary = {"remote_exit_code": capture["returncode"], "streams_complete": capture["complete"],
               "stream_bytes": {key: len(capture[key]) for key in ("stdout", "stderr")},
               "stream_sha256": {key: hashlib.sha256(capture[key]).hexdigest() for key in ("stdout", "stderr")},
               "validator_passed": False, "reason": capture["reason"]}
    if capture["reason"] or not all(capture["complete"].values()):
        summary["reason"] = capture["reason"] or "incomplete_streams"
        return summary
    if type(capture["returncode"]) is not int or capture["returncode"] != 0:
        summary["reason"] = "remote_nonzero"
        return summary
    try:
        from _common.qat_stack import pip_policy
        report = json.loads(capture["stdout"].decode("utf-8", errors="strict"))
        if not isinstance(report, dict) or report.get("schema") != "qat-cpu-stack-v1" or report.get("status") != "passed":
            raise ValueError()
        checks = report["checks"]
        if not isinstance(checks, dict) or set(checks) != {"metadata", "pip_check", "native", "vllm_cli"}:
            raise ValueError()
        selected = {}
        for name, check in checks.items():
            if not isinstance(check, dict) or type(check.get("exit_code")) is not int:
                raise ValueError()
            if name == "pip_check":
                if check.get("policy_status") != "passed" or (check.get("status"), check["exit_code"]) not in {
                        ("passed", 0), ("failed", 1)}:
                    raise ValueError()
            elif check.get("status") != "passed" or check["exit_code"] != 0:
                raise ValueError()
            if name in {"metadata", "native"} and check.get("observation", {}).get("status") != "passed":
                raise ValueError()
            if name == "pip_check":
                overrides = checks["metadata"]["observation"].get("declared_dependency_overrides", [])
                if not pip_policy(check, overrides):
                    raise ValueError()
            selected[name] = {key: check[key] for key in ("status", "exit_code", "policy_status") if key in check}
        summary.update(validator_passed=True, reason=None, checks=selected)
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        summary["reason"] = "invalid_validator_report"
    return summary


async def run_local(argv, deadline, *, environment=None, initial=None, event=None, stdout_limit=524288):
    """Own a CLI/SDK worker, its process group, both bounded pipes and wait task."""
    spawning = asyncio.create_task(asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=environment, start_new_session=True))
    cancelled_before_reply = False
    try:
        process = await asyncio.shield(spawning)
    except asyncio.CancelledError:
        # Losing the local spawn reply must not lose ownership of its process.
        process = await finish(spawning)
        cancelled_before_reply = True
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = asyncio.Event()
    failed = None
    pending = bytearray()

    async def drain(name, limit):
        while chunk := await getattr(process, name).read(4096):
            remaining = limit - len(buffers[name])
            buffers[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.set()
            if name == "stdout" and event and not overflow.is_set():
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending[:] = rest
                    if line.startswith(b"QAT_EVENT "):
                        reply = await event(json.loads(line[len(b"QAT_EVENT "):]))
                        if reply:
                            process.stdin.write(b"ack\n")
                            await process.stdin.drain()
        if name == "stdout" and event and pending.startswith(b"QAT_EVENT "):
            raise ValueError("incomplete_worker_event")

    def kill():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            # A process-group error must not bypass the remaining local joins.
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
            return "local_cleanup_error"
        return None

    tasks = [asyncio.create_task(drain("stdout", stdout_limit)),
             asyncio.create_task(drain("stderr", 8192)), asyncio.create_task(process.wait())]
    completion = asyncio.gather(*tasks)
    limit_task = asyncio.create_task(overflow.wait())
    try:
        if cancelled_before_reply:
            raise asyncio.CancelledError()
        if initial is not None:
            process.stdin.write(initial)
            await process.stdin.drain()
        if event is None:
            process.stdin.close()
        async with asyncio.timeout_at(deadline):
            done, _ = await asyncio.wait((completion, limit_task), return_when=asyncio.FIRST_COMPLETED)
            if limit_task in done:
                failed = "local_output_limit"
            else:
                await completion
    except TimeoutError:
        failed = "local_timeout"
    except asyncio.CancelledError:
        raise
    except Exception:
        failed = "local_process_error"
    finally:
        cleanup_error = kill()
        if cleanup_error:
            failed = cleanup_error
        process.stdin.close()
        limit_task.cancel()
        async def join():
            try:
                async with asyncio.timeout(2):
                    await asyncio.shield(completion)
            except Exception:
                for task in tasks:
                    task.cancel()
            finally:
                await asyncio.gather(*tasks, limit_task, return_exceptions=True)
                if not completion.done():
                    completion.cancel()
                await asyncio.gather(completion, return_exceptions=True)
        await finish(join(), propagate=True)
    if overflow.is_set():
        failed = "local_output_limit"
    if process.returncode is None:
        failed = "local_join_unknown"
    return {"returncode": process.returncode, "reason": failed,
            "stdout": bytes(buffers["stdout"]), "stderr": bytes(buffers["stderr"])}
