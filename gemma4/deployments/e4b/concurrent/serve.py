"""Gemma 4 E4B-it on Modal 1xL40S — CONCURRENT deployment (router shape).

Shape this is tuned for
-----------------------
- The agentic harness's **quick info router**: short prompts -> fast decisions
- High in-flight concurrency (16 streams) on a single mid-tier GPU
- 32K context — generous for routing prompts, under E4B's 128K ceiling
- Text-only despite E4B being multimodal — vision + audio encoders load
  inert (we never send image/audio inputs)

The router model
----------------
E4B-it is the family's quick info-router: a Dense + Per-Layer-Embeddings
(PLE) checkpoint, ~8B total / 4.5B effective, with a 78.8M NEXTN drafter.
Because it is the latency-critical router rather than a reasoning workhorse,
it runs HIGHER concurrency than the dense models and uses SHORTER
warmup/health timeouts.

Why L40S over a flagship card
-----------------------------
E4B is small (~16 GiB BF16 weights + 78.8M drafter). An L40S (48 GiB) has
plenty of capacity at a fraction of the cost of a flagship GPU. The router
workload is short-prompt-dominated and prefix-cache-friendly, so memory
bandwidth doesn't dominate.

Why the upstream E4B chat template (no fork)
--------------------------------------------
E4B's upstream template differs from the 31B/26B/12B upstream by a few
lines (it omits the "pre-fill an empty thinking channel when thinking is
off" hack). For a router we want thinking OFF by default (latency-critical,
short decisions), which is exactly E4B's upstream default — so we bake the
verbatim upstream (``chat_templates/gemma4_e4b_upstream.jinja``) with no
fork.

Memory snapshot lifecycle
-------------------------
Same Modal Memory Snapshot pattern as the solo shape. L40S isn't explicitly
named in Modal's snapshot docs (A10 and H100 are the named examples) but is
architecturally similar to A10 (both Ada Lovelace). If a snapshot fails to
create, disable by removing ``enable_memory_snapshot`` and
``experimental_options`` from the ``@app.cls`` decorator (and drop
``enable_memory_saver`` / ``enable_weights_cpu_backup`` below).

Ingress
-------
The SGLang server binds 0.0.0.0 and Modal's ``@modal.web_server`` publishes
a public ``*.modal.run`` URL (the URL printed by ``modal deploy``). To
restrict access, see Modal's endpoint-security docs (proxy auth tokens):
https://modal.com/docs/guide/webhook-proxy-auth . No auth is baked in here.
"""

from __future__ import annotations

from pathlib import Path

import modal

from _common.health import (
    release_memory_occupation,
    resume_memory_occupation,
    send_warmup_request,
    wait_for_health,
)
from _common.model_registry import get
from _common.sglang_common import (
    MTP_NEXTN_STANDARD,
    SGLANG_TAG,
    build_serve_cmd,
    make_sglang_image,
)

SPEC = get("e4b")
DRAFT = SPEC.draft

APP_NAME = "gemma4-e4b-concurrent"
SERVE_PORT = 8000
SERVED_MODEL_NAME = "gemma-4-e4b-it"

# ── Tuned knobs ──────────────────────────────────────────────────────────
MTP_PROFILE = MTP_NEXTN_STANDARD
MAX_RUNNING_REQUESTS = 16
TARGET_RUNNING_REQUESTS = 12  # ~75% load before considering full
CONTEXT_LENGTH = 32_000
CHUNKED_PREFILL_SIZE = 4_096
MAX_PREFILL_TOKENS = 8_192
MEM_FRACTION_STATIC = 0.9
CUDA_GRAPH_BS = [1, 2, 4, 8, 12, 16]

# ── Chat template ──────────────────────────────────────────────────────────
# E4B uses the verbatim UPSTREAM template (no fork) so thinking is OFF by
# default — see module docstring. Resolve from the repo's chat_templates/
# dir via a robust path that works regardless of the deploy CWD, and bake it
# into the image with add_local_file(copy=True).
TEMPLATE_LOCAL_PATH = (
    Path(__file__).resolve().parents[3] / "chat_templates" / "gemma4_e4b_upstream.jinja"
)
TEMPLATE_PATH_IN_IMAGE = "/opt/sglang/templates/gemma4_e4b_upstream.jinja"

assert CONTEXT_LENGTH <= SPEC.native_max_model_len, (
    f"CONTEXT_LENGTH {CONTEXT_LENGTH} exceeds E4B's native ceiling "
    f"{SPEC.native_max_model_len}"
)


app = modal.App(APP_NAME)

image = (
    make_sglang_image(SGLANG_TAG)
    .add_local_file(
        TEMPLATE_LOCAL_PATH,
        TEMPLATE_PATH_IN_IMAGE,
        copy=True,
    )
    # add_local_python_source MUST be the last build step — Modal forbids
    # further build steps after a non-copy local-source addition.
    .add_local_python_source("_common")
)

# Generic HF weight-cache volume, shared across this model's shapes. The HF
# cache path is set by make_sglang_image (HF_HUB_CACHE=/modal-cache/huggingface).
hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=SPEC.default_gpu,  # "L40S"
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/modal-cache/huggingface": hf_cache},
    # HF token is OPTIONAL — E4B is ungated, so no secret is attached by
    # default. If you mirror the weights to a gated/private repo, create the
    # secret yourself (`modal secret create huggingface-secret HF_TOKEN=...`)
    # and add `secrets=[modal.Secret.from_name("huggingface-secret")]` here.
    timeout=60 * 60 * 2,
    scaledown_window=60 * 10,  # router is bursty
    max_containers=1,
)
@modal.concurrent(
    target_inputs=TARGET_RUNNING_REQUESTS,
    max_inputs=MAX_RUNNING_REQUESTS,
)
class Serve:
    @modal.enter(snap=True)
    def startup(self) -> None:
        import os
        import subprocess

        cmd = build_serve_cmd(
            model_path=SPEC.hf_repo,
            served_model_names=[SERVED_MODEL_NAME],
            max_model_len=CONTEXT_LENGTH,
            mem_fraction_static=MEM_FRACTION_STATIC,
            chunked_prefill_size=CHUNKED_PREFILL_SIZE,
            max_running_requests=MAX_RUNNING_REQUESTS,
            tp_size=1,
            revision=SPEC.hf_revision,
            speculative_config=MTP_PROFILE,
            draft_model_path=DRAFT.hf_repo,
            draft_revision=DRAFT.hf_revision,
            # FP8 KV is intentionally NOT used for E4B: the SGLang FP8-KV
            # crash in sgl-project/sglang#22277 is specific to E4B
            # (num_kv_shared_layers=18), unlike the 31B-it which is validated.
            # Leave KV at the safe BF16 default for this variant.
            attention_backend="triton",  # required for Gemma 4's 512-wide global head
            cuda_graph_bs=CUDA_GRAPH_BS,
            max_prefill_tokens=MAX_PREFILL_TOKENS,
            chat_template=TEMPLATE_PATH_IN_IMAGE,
            # Memory-snapshot co-operation — see module docstring.
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            enable_metrics=True,
            enable_request_time_stats=True,
            log_requests=True,
            log_requests_level=1,
            skip_server_warmup=True,
            # Bind 0.0.0.0 so Modal's @modal.web_server ingress can reach the
            # process across the container's external interface; the
            # health/warmup probes below still reach it on localhost.
            host="0.0.0.0",
            port=SERVE_PORT,
        )

        # v0.5.12+ defaults to spec-decode V2; set explicitly as defence
        # against a future opt-in flip while MTP is enabled.
        env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}
        self.process = subprocess.Popen(cmd, env=env)

        # Shorter warmup/health timeouts than the dense models — E4B is small
        # and boots fast.
        wait_for_health(self.process, port=SERVE_PORT, label=APP_NAME, timeout_s=600)
        send_warmup_request(model=SERVED_MODEL_NAME, port=SERVE_PORT)
        release_memory_occupation(port=SERVE_PORT)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        resume_memory_occupation(port=SERVE_PORT)
        wait_for_health(port=SERVE_PORT, label=f"{APP_NAME}-resume", timeout_s=120)

    @modal.web_server(port=SERVE_PORT, startup_timeout=60 * 10)
    def serve(self) -> None:
        pass

    @modal.exit()
    def stop(self) -> None:
        proc = getattr(self, "process", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
