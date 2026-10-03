"""CPU checks for the pinned QAT image; no serving, model or GPU allocation."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import re
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

EXPECTED_VERSIONS = {"torch": "2.13.0", "torchvision": "0.28.0", "vllm": "0.30.0"}
CUDA_VERSION = "12.9"
STREAM_LIMIT = 8192
REPORT_LIMIT = 32768
# Exact, reviewed upstream exceptions only; never ignore a dependency category.
DEPENDENCY_OVERRIDES = frozenset({
    ("torch", "2.13.0+cu129", "nvidia-nccl-cu12", "==2.29.7", "2.30.7"),
})


def dependency_report(distributions, environment):
    """Check active requirements and separately report exact declared exceptions."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = {canonicalize_name(dist.metadata["Name"]): dist for dist in distributions}
    if len(installed) != len(distributions):
        raise RuntimeError("duplicate installed distribution identities")
    # Propagate requested extras: Torch activates cuda-toolkit's component pins.
    extras = {name: {""} for name in installed}
    requirements = {name: [Requirement(text) for text in dist.requires or []]
                    for name, dist in installed.items()}
    def active(name, requirement):
        return not requirement.marker or any(
            requirement.marker.evaluate({**environment, "extra": extra}) for extra in extras[name])
    changed = True
    while changed:
        changed = False
        for name, rows in requirements.items():
            for requirement in rows:
                dependency = canonicalize_name(requirement.name)
                if dependency in extras and active(name, requirement):
                    before = len(extras[dependency])
                    extras[dependency].update(requirement.extras)
                    changed |= len(extras[dependency]) != before
    issues, overrides = [], []
    for name, dist in sorted(installed.items()):
        for requirement in requirements[name]:
            if not active(name, requirement):
                continue
            dependency = canonicalize_name(requirement.name)
            actual = installed[dependency].version if dependency in installed else None
            if actual is not None and requirement.specifier.contains(actual, prereleases=True):
                continue
            row = {"package": name, "package_version": dist.version,
                   "dependency": dependency, "required": str(requirement.specifier), "installed": actual}
            identity = (name, dist.version, dependency, str(requirement.specifier), actual)
            (overrides if identity in DEPENDENCY_OVERRIDES else issues).append(row)
    return issues, overrides


def metadata_check():
    """Reject unexpected selected packages and incompatible active dependencies."""
    from packaging.markers import default_environment
    from packaging.utils import canonicalize_name

    distributions = list(metadata.distributions())
    versions = {canonicalize_name(dist.metadata["Name"]): dist.version for dist in distributions}
    selected = {name: versions.get(name) for name in (*EXPECTED_VERSIONS, "transformers")}
    mismatches = []
    for name, expected in EXPECTED_VERSIONS.items():
        actual = selected[name] or ""
        base, _, build = actual.partition("+")
        if base != expected or (build and build != "cu129"):
            mismatches.append({"package": name, "expected": expected + "+cu129", "actual": actual})
    if selected["transformers"] is None:
        mismatches.append({"package": "transformers", "expected": "installed", "actual": None})
    issues, overrides = dependency_report(distributions, default_environment())
    return {"status": "failed" if mismatches or issues else "passed", "versions": selected,
            "version_mismatches": mismatches, "dependency_issue_count": len(issues),
            "dependency_issues": issues[:16], "dependency_issues_truncated": len(issues) > 16,
            "declared_dependency_overrides": overrides[:16]}


def native_check():
    """Load native libraries and run a tiny, deterministic CPU NMS operation."""
    import torch

    if str(torch.__version__) != EXPECTED_VERSIONS["torch"] + "+cu129" or torch.version.cuda != CUDA_VERSION:
        raise RuntimeError(f"unexpected Torch runtime: {torch.__version__}, CUDA {torch.version.cuda}")
    vision = metadata.distribution("torchvision")
    candidates = [vision.locate_file(path) for path in vision.files or []
                  if path.parts[0] == "torchvision" and path.name.startswith("_C") and path.name.endswith(".so")]
    if len(candidates) != 1 or not Path(candidates[0]).is_file():
        raise RuntimeError("unique TorchVision native extension not found")
    # TorchVision can hide this loader exception before failing NMS registration.
    torch.ops.load_library(str(candidates[0]))
    if not torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "CPU"):
        raise RuntimeError("TorchVision CPU NMS kernel is not registered")
    import torchvision

    if str(torchvision.__version__) != EXPECTED_VERSIONS["torchvision"] + "+cu129":
        raise RuntimeError(f"unexpected TorchVision runtime: {torchvision.__version__}")
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.], [4., 4., 6., 6.]], device="cpu")
    scores = torch.tensor([.9, .8, .7], device="cpu")
    selected = torchvision.ops.nms(boxes, scores, .5).tolist()
    if selected != [0, 2]:
        raise RuntimeError(f"incorrect CPU NMS result: {selected}")
    nccl = nccl_runtime_version()
    return {"status": "passed", "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "torchvision": str(torchvision.__version__), "cpu_nms_registered": True, "cpu_nms_result": selected, "nccl": nccl}


def nccl_runtime_version():
    """Query the selected library's version; ncclGetVersion initializes no device."""
    import base64
    import ctypes
    import hashlib

    dist = metadata.distribution("nvidia-nccl-cu12")
    if dist.version != "2.30.7":
        raise RuntimeError("unexpected NCCL distribution version")
    files = [path for path in dist.files or [] if str(path) == "nvidia/nccl/lib/libnccl.so.2"]
    if len(files) != 1 or not files[0].hash or files[0].hash.mode != "sha256":
        raise RuntimeError("NCCL library RECORD hash missing")
    path = dist.locate_file(files[0])
    with open(path, "rb") as library:
        actual_hash = base64.urlsafe_b64encode(hashlib.file_digest(library, "sha256").digest()).rstrip(b"=").decode()
    if actual_hash != files[0].hash.value:
        raise RuntimeError("NCCL library differs from its distribution RECORD")
    version = ctypes.c_int()
    query = ctypes.CDLL(str(path)).ncclGetVersion
    query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    if query(ctypes.byref(version)) != 0 or version.value != 23007:
        raise RuntimeError(f"unexpected NCCL runtime version: {version.value}")
    return {"distribution": dist.version, "runtime": version.value, "record_sha256_verified": True}


def pip_policy(outcome, overrides):
    """Keep actual pip output/exit status and accept only corroborated exceptions."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if not overrides:
        return (outcome["status"] == "passed" and outcome["exit_code"] == 0
                and outcome["stdout"].strip() == "No broken requirements found." and not outcome["stderr"])
    if (outcome["status"] != "failed" or outcome.get("reason") != "nonzero_exit"
            or outcome["exit_code"] != 1 or outcome["stderr"]):
        return False
    expected = {(row["package"], row["package_version"], row["dependency"], row["required"], row["installed"])
                for row in overrides}
    if not expected <= DEPENDENCY_OVERRIDES:
        return False
    observed = []
    for line in outcome["stdout"].splitlines():
        match = re.fullmatch(r"(\S+) (\S+) has requirement (.+), but you have (\S+) (\S+)\.", line)
        if not match:
            return False
        owner, version, text, dependency, actual = match.groups()
        requirement = Requirement(text)
        if canonicalize_name(requirement.name) != canonicalize_name(dependency):
            return False
        observed.append((canonicalize_name(owner), version, canonicalize_name(dependency),
                         str(requirement.specifier), actual))
    return len(observed) == len(expected) and set(observed) == expected


def run_child(argv, *, timeout, environment=None, stream_limit=STREAM_LIMIT):
    """Drain both pipes concurrently with byte limits; kill and join on failure."""
    result = {"status": "failed", "exit_code": None, "stdout": "", "stderr": ""}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    process = None
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=environment, start_new_session=True)
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            for name in buffers:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result["reason"] = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    name = key.data
                    chunk = os.read(key.fd, min(4096, stream_limit - len(buffers[name]) + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    space = stream_limit - len(buffers[name])
                    buffers[name].extend(chunk[:space])
                    if len(chunk) > space:
                        result.update(reason="output_limit", overflow_stream=name)
                        break
                if "reason" in result:
                    break
            if "reason" not in result:
                result["status"] = "passed" if process.returncode == 0 else "failed"
                if process.returncode != 0:
                    result["reason"] = "nonzero_exit"
    except Exception as exc:
        result.update(reason="process_error", error=f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        if process is not None:
            # Also close descendants holding inherited pipes after the leader exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                result.update(status="failed", reason="cleanup_error", error=str(exc)[:1000])
            try:
                result["exit_code"] = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                result.update(status="failed", reason="cleanup_timeout")
            for name in buffers:
                getattr(process, name).close()
        for name, content in buffers.items():
            result[name] = content.decode("utf-8", errors="replace")
    return result


def check_stack():
    """Require all CPU stages to pass; process completion alone is insufficient."""
    executable = sys.executable
    script = str(Path(__file__).resolve())
    offline = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "CUDA_VISIBLE_DEVICES": ""}
    offline.pop("VLLM_TARGET_DEVICE", None)
    # Explicit CPU selection is confined to the CLI-help child, never the image.
    stages = [
        ("metadata", [executable, script, "--metadata-check"], 20, offline),
        ("pip_check", [executable, "-m", "pip", "check"], 20, offline),
        ("native", [executable, script, "--native-check"], 40, offline),
        ("vllm_cli", [executable, "-m", "vllm.entrypoints.cli.main", "--help"], 40,
         {**offline, "VLLM_TARGET_DEVICE": "cpu"}),
    ]
    report = {"schema": "qat-cpu-stack-v1", "status": "failed", "checks": {},
              "scope": "CPU package/native-extension/CLI checks; GPU serving remains unqualified"}
    for name, argv, timeout, environment in stages:
        outcome = run_child(argv, timeout=timeout, environment=environment)
        report["checks"][name] = outcome
        if name == "pip_check":
            overrides = report["checks"]["metadata"]["observation"].get("declared_dependency_overrides", [])
            accepted = pip_policy(outcome, overrides)
            outcome["policy_status"] = "passed" if accepted else "failed"
            if not accepted:
                return report
            continue
        if outcome["status"] == "passed":
            if name == "vllm_cli":
                # Python -m sets argv[0] to main.py; the pinned parser uses
                # argparse's default prog rather than the console-script name.
                # Platform diagnostics can precede help on the same stream.
                if not re.search(r"^usage: main\.py ", outcome["stdout"], re.MULTILINE) or "serve" not in outcome["stdout"]:
                    outcome.update(status="failed", reason="expected_help_missing")
            else:
                try:
                    observation = json.loads(outcome["stdout"])
                    if not isinstance(observation, dict) or observation.get("status") != "passed":
                        outcome.update(status="failed", reason="check_rejected")
                    outcome["observation"] = observation
                    outcome.pop("stdout")
                except (ValueError, TypeError):
                    outcome.update(status="failed", reason="invalid_check_output")
        if outcome["status"] != "passed":
            return report
    report["status"] = "passed"
    return report


def main(argv=None):
    """Emit one bounded JSON report and return nonzero for every failed check."""
    args = sys.argv[1:] if argv is None else argv
    try:
        if args == ["--metadata-check"]:
            report = metadata_check()
        elif args == ["--native-check"]:
            report = native_check()
        elif not args:
            report = check_stack()
        else:
            raise ValueError("unsupported validator arguments")
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:1600]}
    encoded = json.dumps(report, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("ascii")) + 1 > REPORT_LIMIT:
        report = {"status": "failed", "error": "report_output_limit",
                  "check_statuses": {name: value["status"] for name, value in report.get("checks", {}).items()}}
        encoded = json.dumps(report, separators=(",", ":"))
    print(encoded)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
