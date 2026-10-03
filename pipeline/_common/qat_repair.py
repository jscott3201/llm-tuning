"""Build-only repair for the selected vLLM image; never used by serving processes."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from .qat_stack import active_distributions
else:
    from qat_stack import active_distributions

PYTHON = "/usr/bin/python3.12"
TORCH_URL = (
    "https://download.pytorch.org/whl/cu129/"
    "torch-2.13.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"
    "#sha256=df28741fcd89e3da7cce2d48cbe5299d6732d510ac20f5d422d0b85edf18c327"
)
# These CUDA 13 libraries came from the displaced Torch 2.14 stack. Their files
# overlap CUDA 12 packages, so remove them before forcibly restoring CUDA 12.
OLD_NATIVE = {
    "nvidia-cublas": "==13.1.1.3.*", "nvidia-cuda-runtime": "==13.0.96.*",
    "nvidia-cufft": "==12.0.0.61.*", "nvidia-cufile": "==1.15.1.6.*",
    "nvidia-cuda-cupti": "==13.0.85.*", "nvidia-curand": "==10.4.0.35.*",
    "nvidia-cusolver": "==12.0.4.66.*", "nvidia-cusparse": "==12.6.3.3.*",
    "nvidia-nvjitlink": ">=13.0.88,<14", "nvidia-cuda-nvrtc": "==13.0.88.*",
    "nvidia-nvtx": "==13.0.85.*", "nvidia-cudnn-cu13": "==9.24.0.43",
    "nvidia-cusparselt-cu13": "==0.8.1", "nvidia-nccl-cu13": "==2.30.7",
    "nvidia-nvshmem-cu13": "==3.4.5",
}
RESTORE_NATIVE = {
    "nvidia-cublas-cu12": "12.9.1.4", "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cufft-cu12": "11.4.1.4", "nvidia-cufile-cu12": "1.14.1.1",
    "nvidia-cuda-cupti-cu12": "12.9.79", "nvidia-curand-cu12": "10.3.10.19",
    "nvidia-cusolver-cu12": "11.7.5.82", "nvidia-cusparse-cu12": "12.5.10.65",
    "nvidia-nvjitlink-cu12": "12.9.86", "nvidia-cuda-nvrtc-cu12": "12.9.86",
    "nvidia-nvtx-cu12": "12.9.79", "nvidia-cudnn-cu12": "9.20.0.48",
    "nvidia-cusparselt-cu12": "0.8.1", "nvidia-nccl-cu12": "2.30.7",
    "nvidia-nvshmem-cu12": "3.4.5",
}
OWNED_VERSIONS = {**RESTORE_NATIVE, "cuda-toolkit": "12.9.1", "triton": "3.7.1",
                  "cuda-bindings": "12.9.4"}
RETAINED = {"torchvision": "0.28.0+cu129", "vllm": "0.30.0+cu129",
            "torchaudio": "2.11.0", "transformers": "5.17.0"}


def repair_plan(versions, override, constraints):
    """Fail on an unrecognized base and return ordered, interpreter-bound argv."""
    from packaging.specifiers import SpecifierSet

    if versions.get("torch") not in {"2.14.0", "2.14.0+cu130"} or versions.get("cuda-toolkit") != "13.0.3":
        raise RuntimeError("unrecognized base Torch/CUDA stack")
    for name, expected in RETAINED.items():
        allowed = {expected, expected + "+cu129"} if name == "torchaudio" else {expected}
        if versions.get(name) not in allowed:
            raise RuntimeError(f"unrecognized retained package: {name}")
    old = []
    for name, specifier in sorted(OLD_NATIVE.items()):
        if name in versions:
            if versions[name] not in SpecifierSet(specifier):
                raise RuntimeError(f"unrecognized CUDA 13 package: {name}")
            old.append(name)
    commands = []
    if old:
        commands.append(["uv", "--no-config", "pip", "uninstall", "--python", PYTHON, *old])
    install = ["uv", "--no-config", "pip", "install", "--python", PYTHON,
               "--index-url", "https://pypi.org/simple", "--override", str(override),
               "--constraint", str(constraints)]
    for name in sorted(RESTORE_NATIVE):
        install += ["--reinstall-package", name]
    commands.append([*install, TORCH_URL])
    return commands


def constraints_text(versions):
    """Pin owned changes and hold every other installed distribution unchanged."""
    retained = {name: version for name, version in versions.items()
                if name not in {*OLD_NATIVE, "torch", "cuda-bindings", *OWNED_VERSIONS}}
    retained.update(OWNED_VERSIONS)
    return "".join(f"{name}=={version}\n" for name, version in sorted(retained.items()))


def resolver_environment(environment):
    """Exclude inherited resolver policy without exposing environment contents."""
    return {key: value for key, value in environment.items()
            if not key.startswith(("UV_", "PIP_")) and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}


def main():
    """Repair only the known base; the separate validator certifies CPU behavior."""
    if os.path.realpath(sys.executable) != PYTHON or sys.version_info[:2] != (3, 12) or sys.prefix != sys.base_prefix:
        raise RuntimeError("repair requires the image's system Python 3.12")
    versions = {name: dist.version for name, dist in active_distributions().items()}
    with tempfile.TemporaryDirectory(prefix="qat-stack-") as directory:
        override, constraints = Path(directory) / "override.txt", Path(directory) / "constraints.txt"
        # vLLM intentionally uses NCCL >=2.30.4 for DeepEP v2 GIN. Preserve its
        # exact shipped 2.30.7; the validator reports the Torch metadata exception.
        override.write_text("nvidia-nccl-cu12==2.30.7\n")
        constraints.write_text(constraints_text(versions))
        for command in repair_plan(versions, override, constraints):
            subprocess.run(command, env=resolver_environment(os.environ), check=True, timeout=900)


if __name__ == "__main__":
    main()
