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


CHECK_STAGES = ("metadata", "pip_check", "native", "vllm_cli")
CHECK_FAILURE_REASONS = frozenset({
    "nonzero_exit", "timeout", "output_limit", "process_error", "cleanup_error", "cleanup_timeout",
    "check_rejected", "invalid_check_output", "expected_help_missing", "dependency_policy_rejected",
})


def accepted_check(name, check, checks):
    """Recheck a successful stage, including the exact raw pip exception evidence."""
    from _common.qat_stack import pip_policy
    if not isinstance(check, dict) or type(check.get("exit_code")) is not int:
        return False
    if name == "pip_check":
        overrides = checks["metadata"]["observation"].get("declared_dependency_overrides", [])
        return (check.get("policy_status") == "passed"
                and (check.get("status"), check["exit_code"], check.get("reason")) in {
                    ("passed", 0, None), ("failed", 1, "nonzero_exit")}
                and pip_policy(check, overrides))
    return (check.get("status") == "passed" and check["exit_code"] == 0 and check.get("reason") is None
            and (name not in {"metadata", "native"} or check.get("observation", {}).get("status") == "passed"))


def failed_check_reason(name, check, checks):
    """Select a known failure reason without retaining diagnostic text."""
    from _common.qat_stack import pip_policy
    if not isinstance(check, dict) or "exit_code" not in check or (check.get("exit_code") is not None and type(check["exit_code"]) is not int):
        raise ValueError("invalid_failed_check")
    reason, code = check.get("reason"), check.get("exit_code")
    if name == "pip_check":
        overrides = checks["metadata"]["observation"].get("declared_dependency_overrides", [])
        if check.get("policy_status") != "failed" or pip_policy(check, overrides):
            raise ValueError("contradictory_pip_failure")
        if check.get("status") == "passed" and code == 0 and reason is None:
            return "dependency_policy_rejected"
    if check.get("status") != "failed" or reason not in CHECK_FAILURE_REASONS - {"dependency_policy_rejected"}:
        raise ValueError("invalid_failed_check_status")
    if reason == "nonzero_exit" and (type(code) is not int or code == 0):
        raise ValueError("contradictory_child_exit")
    if reason in {"check_rejected", "invalid_check_output"} and (name not in {"metadata", "native"} or code != 0):
        raise ValueError("contradictory_check_rejection")
    if reason == "check_rejected" and ("observation" not in check or (isinstance(check["observation"], dict) and check["observation"].get("status") == "passed")):
        raise ValueError("contradictory_observation")
    if reason == "expected_help_missing" and (name != "vllm_cli" or code != 0):
        raise ValueError("contradictory_help_rejection")
    return reason


def metadata_mismatch(check):
    """Recognize the metadata check's mismatch shape; expose only a fixed category."""
    if (check.get("reason"), check.get("exit_code")) not in {("nonzero_exit", 1), ("check_rejected", 0)}:
        return False
    try:
        from packaging.specifiers import SpecifierSet
        from _common.qat_stack import EXPECTED_VERSIONS
        observation = check.get("observation")
        if observation is None:
            observation = json.loads(check["stdout"])
        if not isinstance(observation, dict) or observation.get("status") != "failed":
            return False
        versions = observation["versions"]
        mismatches, issues = observation["version_mismatches"], observation["dependency_issues"]
        count = observation["dependency_issue_count"]
        if (not isinstance(versions, dict) or set(versions) != {"torch", "torchvision", "vllm", "transformers"}
                or any(value is not None and not isinstance(value, str) for value in versions.values())
                or not isinstance(mismatches, list) or len(mismatches) > 4
                or type(count) is not int or count < 0 or not isinstance(issues, list) or len(issues) != min(count, 16)
                or type(observation["dependency_issues_truncated"]) is not bool
                or observation["dependency_issues_truncated"] != (count > 16)
                or not isinstance(observation["declared_dependency_overrides"], list)):
            return False
        expected_mismatches = []
        for name, expected in EXPECTED_VERSIONS.items():
            actual = versions[name] or ""
            base, _, build = actual.partition("+")
            if base != expected or (build and build != "cu129"):
                expected_mismatches.append({"package": name, "expected": expected + "+cu129", "actual": actual})
        if versions["transformers"] is None:
            expected_mismatches.append({"package": "transformers", "expected": "installed", "actual": None})
        if mismatches != expected_mismatches:
            return False
        for row in issues:
            if (not isinstance(row, dict) or set(row) != {"package", "package_version", "dependency", "required", "installed"}
                    or any(not isinstance(row[key], str) for key in ("package", "package_version", "dependency", "required"))
                    or (row["installed"] is not None and not isinstance(row["installed"], str))):
                return False
            if row["installed"] is not None and SpecifierSet(row["required"]).contains(row["installed"], prereleases=True):
                return False
        return bool(mismatches or count)
    except (ValueError, KeyError, TypeError):
        return False


def evaluate_capture(capture):
    """Accept a full passing report or classify bounded evidence of a failed check."""
    summary = {"remote_exit_code": capture["returncode"], "streams_complete": capture["complete"],
               "stream_bytes": {key: len(capture[key]) for key in ("stdout", "stderr")},
               "stream_sha256": {key: hashlib.sha256(capture[key]).hexdigest() for key in ("stdout", "stderr")},
               "validator_passed": False, "validator_report_status": "unknown", "reason": capture["reason"]}
    if capture["reason"] or not all(capture["complete"].values()):
        summary["reason"] = capture["reason"] or "incomplete_streams"
        return summary
    code = capture["returncode"]
    if type(code) is not int or code != 0:
        summary["reason"] = "remote_nonzero"
    if type(code) is not int or code not in {0, 1}:
        return summary
    try:
        report = json.loads(capture["stdout"].decode("utf-8", errors="strict"))
        status = "passed" if code == 0 else "failed"
        if not isinstance(report, dict) or report.get("schema") != "qat-cpu-stack-v1" or report.get("status") != status:
            raise ValueError()
        checks = report["checks"]
        if (not isinstance(checks, dict) or not checks or set(checks) != set(CHECK_STAGES[:len(checks)])
                or (status == "passed" and len(checks) != len(CHECK_STAGES))):
            raise ValueError()
        selected = {}
        for index, name in enumerate(CHECK_STAGES[:len(checks)]):
            check = checks[name]
            if status == "failed" and index == len(checks) - 1:
                reason = failed_check_reason(name, check, checks)
            elif not accepted_check(name, check, checks):
                raise ValueError()
            else:
                reason = check.get("reason")
            selected[name] = {key: check[key] for key in ("status", "exit_code", "policy_status") if key in check}
            if reason is not None:
                selected[name]["reason"] = reason
            if status == "failed" and name == "metadata" and metadata_mismatch(check):
                selected[name]["rejection_kind"] = "metadata_mismatch"
        summary.update(validator_passed=status == "passed", validator_report_status=status, checks=selected)
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        if code == 0:
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
