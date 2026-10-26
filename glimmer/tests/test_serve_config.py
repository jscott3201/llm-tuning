"""Static checks on the serve-command builder and each deployment's constants.

Imports the serve scripts the way ``modal deploy`` does (with the project
root on sys.path) but never builds an image or touches a GPU.

    uv run --group dev pytest tests/ -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _common.model_registry import GPU_PROFILES, get  # noqa: E402
from _common.sglang_common import (  # noqa: E402
    DFLASH_8,
    DFLASH_16,
    build_serve_cmd,
)

DEPLOYMENTS = sorted((ROOT / "deployments").glob("*/serve.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"serve_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flag(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


BASE = dict(
    model_path="nvidia/Muse-Glimmer-30B-NVFP4",
    served_model_name="muse-glimmer",
    context_length=131_072,
    mem_fraction_static=0.85,
    chunked_prefill_size=16_384,
    max_running_requests=8,
)


def test_parsers_and_bind():
    cmd = build_serve_cmd(**BASE)
    assert _flag(cmd, "--reasoning-parser") == "muse"
    assert _flag(cmd, "--tool-call-parser") == "muse"
    assert _flag(cmd, "--host") == "0.0.0.0"
    assert _flag(cmd, "--kv-cache-dtype") == "fp8_e4m3"
    assert "--speculative-algorithm" not in cmd


def test_dflash_flags():
    spec = get("nvfp4")
    cmd = build_serve_cmd(
        **BASE,
        speculative_config=DFLASH_8,
        draft_model_path=spec.draft.hf_repo,
        draft_revision=spec.draft.hf_revision,
        draft_quantization="fp8",
    )
    assert _flag(cmd, "--speculative-algorithm") == "DFLASH"
    assert _flag(cmd, "--speculative-draft-model-path") == spec.draft.hf_repo
    assert _flag(cmd, "--speculative-draft-model-revision") == spec.draft.hf_revision
    assert _flag(cmd, "--speculative-draft-model-quantization") == "fp8"
    assert _flag(cmd, "--speculative-dflash-block-size") == "8"


def test_dflash_needs_drafter():
    with pytest.raises(ValueError, match="draft_model_path"):
        build_serve_cmd(**BASE, speculative_config=DFLASH_16)


def test_cuda_graph_must_cover_max_running():
    with pytest.raises(ValueError, match="cuda_graph_bs"):
        build_serve_cmd(**BASE, cuda_graph_bs=[1, 2, 4])
    cmd = build_serve_cmd(**BASE, cuda_graph_bs=[1, 2, 4, 8])
    i = cmd.index("--cuda-graph-bs-decode")
    assert cmd[i + 1 : i + 5] == ["1", "2", "4", "8"]


def test_snapshot_flags_back_up_drafter_too():
    spec = get("nvfp4")
    cmd = build_serve_cmd(
        **BASE,
        speculative_config=DFLASH_16,
        draft_model_path=spec.draft.hf_repo,
        enable_memory_saver=True,
        enable_weights_cpu_backup=True,
    )
    for f in ("--enable-memory-saver", "--enable-weights-cpu-backup", "--enable-draft-weights-cpu-backup"):
        assert f in cmd


def test_language_model_only_drops_mm_backend():
    cmd = build_serve_cmd(**BASE, language_model_only=True, mm_attention_backend="fa4")
    assert "--language-model-only" in cmd
    assert "--mm-attention-backend" not in cmd


def test_api_key_only_when_env_set(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert "--api-key" not in build_serve_cmd(**BASE)
    monkeypatch.setenv("API_KEY", "s3cret")
    assert _flag(build_serve_cmd(**BASE), "--api-key") == "s3cret"


def test_swa_ratio_and_backend_flags():
    cmd = build_serve_cmd(**BASE, attention_backend="trtllm_mha", swa_full_tokens_ratio=0.1)
    assert _flag(cmd, "--attention-backend") == "trtllm_mha"
    assert _flag(cmd, "--swa-full-tokens-ratio") == "0.1"
    assert "--attention-backend" not in build_serve_cmd(**BASE)


def test_no_b200_plus():
    # B200+ may land on a B300 (CUDA 13.1+); the pinned image is CUDA 13.0.
    assert "B200+" not in GPU_PROFILES


def test_pinned_revisions():
    spec = get("nvfp4")
    assert spec.hf_revision and len(spec.hf_revision) == 40
    assert spec.draft.hf_revision and len(spec.draft.hf_revision) == 40


@pytest.mark.parametrize("path", DEPLOYMENTS, ids=lambda p: p.parent.name)
def test_deployment_constants(path):
    mod = _load(path)
    assert mod.GPU in GPU_PROFILES, "GPU must be one of the tested classes"
    assert mod.CONTEXT_LENGTH <= mod.SPEC.native_max_model_len
    assert max(mod.CUDA_GRAPH_BS) >= mod.MAX_RUNNING_REQUESTS
    assert mod.TARGET_RUNNING_REQUESTS <= mod.MAX_RUNNING_REQUESTS
    assert mod.TEMPLATE_SRC.exists()
    assert mod.APP_NAME.startswith("glimmer-")


def test_both_shapes_exist():
    assert {p.parent.name for p in DEPLOYMENTS} == {"solo", "concurrent"}
