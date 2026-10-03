# Pinned Gemma 4 31B QAT pilot

`serve_31b_qat.py` declares an authenticated, text-only vLLM endpoint for a
bounded integration run. GPU fit, actual attention/kernel selection, cold-start
time, authentication enforcement and shutdown behavior still need live
qualification. The other serving scripts retain their existing settings.

## Fixed inputs and limits

| Setting | Value |
| --- | --- |
| Model | `google/gemma-4-31B-it-qat-w4a16-ct` |
| Model and tokenizer revision | `52f3f65bc7a02d555763bc923bd1d9094898219d` |
| Served model name | `gemma-4-31b-it-qat-w4a16-ct` |
| Image | `docker.io/vllm/vllm-openai:v0.30.0-cu129` |
| Linux AMD64 image digest | `sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688` |
| GPU / tensor parallelism | One exact `H100!` / TP1 |
| CPU / host memory | 4 cores / 65,536 MiB |
| Context / active sequences | 32,768 tokens / 1 |
| Containers / concurrent requests | Minimum 0, maximum 1 / 1 |
| Startup / input timeout / idle window | 1,200 / 1,800 / 300 seconds |
| App | `gemma4-31b-qat-pilot` |

The registry supplies the pinned model and resource defaults. The profile uses
`_common.qat_image.build_image()`, which declares the same image independently
of the serving App and cache Volumes. It clears the entrypoint and mounts
`_common`. The setup layer exposes the image's existing `/usr/bin/python3.12`
as `/usr/local/bin/python`, which Modal requires on PATH. An existing destination
is accepted only when it resolves to that same interpreter. The layer checks the
pip command/module and reports Python, vLLM and Torch package metadata. See
[Modal's existing-image requirements](https://modal.com/docs/guide/existing-images).

### CPU package repair and build guard

The selected upstream image contains Torch 2.14/CUDA 13 while its vLLM 0.30.0
and TorchVision 0.28.0 require Torch 2.13.0. The derived image restores the
[official Torch 2.13.0+cu129 wheel](https://download.pytorch.org/whl/cu129/torch-2.13.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl)
with SHA256 `df28741fcd89e3da7cce2d48cbe5299d6732d510ac20f5d422d0b85edf18c327`.
Its declared CUDA 12 components, CUDA toolkit 12.9.1, cuda-bindings 12.9.4 and
Triton 3.7.1 are version-pinned in `_common/qat_repair.py`. The repair first
removes the displaced Torch stack's known CUDA 13 library distributions, then
forces reinstallation of all 15 selected CUDA 12 library distributions because
the two generations share file paths. It keeps the original Python and holds
other installed packages at their existing versions. Unexpected base versions
or resolver conflicts fail the build. The owned version pins and Torch wheel
hash do not form a complete hash lock for every acquired package; record the
actual derived image ID after a successful build.

One dependency exception is explicit: Torch 2.13.0 declares NCCL 2.29.7, while
the recipe retains upstream NCCL 2.30.7 for DeepEP v2 GIN. The
[pinned vLLM Dockerfile](https://github.com/vllm-project/vllm/blob/v0.30.0/docker/Dockerfile)
uses that override deliberately. This is a documented metadata exception,
not a clean `pip check` result or a Torch/GPU compatibility certification.
The guard rejects every other active dependency conflict, including component
requirements activated through dependency extras.

The image runs `_common/qat_stack.py` on CPU before a serving container starts.
It checks installed versions and requirements; retains bounded `python -m pip
check` stdout, stderr and exit code; loads the actual TorchVision extension;
runs a three-box CPU NMS case; verifies the NCCL library against its installed
RECORD hash and queries its runtime version with `ncclGetVersion`; and invokes
the actual vLLM CLI help with `VLLM_TARGET_DEVICE=cpu` scoped to that child.
The NCCL version query does not initialize a GPU. Checks use offline model
settings, deadlines and bounded output, and fail on child errors, timeouts,
overflow, missing expected results or malformed reports. The sole expected
NCCL conflict can pass the declared policy while its actual pip exit code 1
remains visible in the report. A passing CPU report still leaves GPU serving,
model loading, attention kernels and generation correctness unqualified.

Metadata selection follows Python's search-path precedence. One selected record
per canonical package name supplies both its version and dependency requirements,
including activated extras. Lower-priority OS package records may be shadowed;
multiple records for a package in its winning directory fail as ambiguous.
Malformed names or versions and inconsistent metadata provenance or name lookup
also fail the check. The guard does not remove OS metadata or change `sys.path`.

An offline SDK import validates these declarations without building the image.
The repair and CPU guard still require an actual authorized image build and
Function-creation check before they can be treated as live qualification.

The model's compressed-tensors configuration selects quantization. vLLM
selects attention, model dtype and KV dtype. This profile applies no Triton
override. It uses `--language-model-only`, the `gemma4` tool and reasoning
parsers, and the official tokenizer template at the pinned revision. It has
no custom template, speculative decoder, LoRA adapter or memory snapshot.
See the pinned [model configuration](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/52f3f65bc7a02d555763bc923bd1d9094898219d/config.json)
and [vLLM attention selection](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/cuda.py).

For a request with 16,384 requested completion tokens, the rendered prompt
must fit within the remaining 16,384 tokens. Include the template, tools and
full history in that count. Validate each request or preserve the server's
context rejection; do not silently truncate or resample. A client thinking
budget is not enforced merely by enabling thinking in the chat template.

## CPU preflight of an existing image

`qat_preflight.py` runs the tracked CPU validator against one explicitly selected,
already built Modal image. Import and `--help` allocate nothing. The command
accepts a Modal image ID (`im-...`), not a registry tag or image recipe; it never
calls the image factory or repairs/builds an image implicitly. This also allows
an older image to serve as a negative control with the current validator.

After obtaining a CPU-only runtime allowance, run from `pipeline/` using the
prepared environment. Replace `im-PREBUILT` and `ENVIRONMENT`, and set
`PRIVATE_RUN_DIR` to an existing private directory outside the checkout:

```sh
uv run --frozen --no-sync python serve/vllm/qat_preflight.py run \
  --image-id im-PREBUILT --environment ENVIRONMENT \
  --receipt "$PRIVATE_RUN_DIR/qat-cpu.json"
```

The receipt must be fresh. The command saves a random App description, Sandbox
ownership tag, exact image ID, validator SHA256 and limits before allocating.
Each allocation waits for its intent to be saved, and each returned ID is
recorded immediately. A local lock prevents concurrent run/recovery commands
from changing the same receipt. Keep both the receipt and its `.lock` sidecar
at their original paths during the attempt.

| Bound | Value |
| --- | --- |
| CPU request / limit | 1 / 1 core |
| Memory request / limit | 4,096 / 4,096 MiB |
| Sandbox lifetime | 90 seconds, including startup and validator execution |
| Client budget | 180 seconds, with the final 45 seconds reserved for cleanup |
| Retained validator stdout / stderr | 32,768 / 8,192 bytes |
| Validator source transfer | At most 65,536 bytes, checked by size and SHA256 |

The Sandbox has no GPU, secrets, cache volumes, published ports, PTY or OIDC
identity token, and network access is blocked. A quiet entrypoint keeps the
Sandbox available for a non-PTY exec whose raw stdout/stderr are drained
concurrently with remote completion. The source is transferred into temporary
Sandbox storage and verified before execution. Success requires both stream
EOFs, remote exit zero, a valid passing validator report and verified cleanup.
The declared NCCL metadata exception retains pip exit 1 and its passing policy
status. Every other pip conflict is rejected.

The receipt retains selected check statuses, remote exit code, stream byte
counts and hashes, resource identities and cleanup evidence. It omits raw
account inventory, SDK exceptions, environment contents and diagnostic output.
A private receipt belongs outside version control. The local SDK worker and CLI
children are bounded and joined; their termination does not prove that an
unanswered server allocation request was cancelled.

### Recovery without allocation

On interruption or cleanup failure, reuse the same receipt and environment:

```sh
uv run --frozen --no-sync python serve/vllm/qat_preflight.py recover \
  --environment ENVIRONMENT --receipt "$PRIVATE_RUN_DIR/qat-cpu.json"
```

Recovery creates no App, image, Function or Sandbox. It finds an unacknowledged
App only by the saved unique description in the saved environment, rejects
duplicate or mismatched identities, terminates acknowledged or uniquely tagged
Sandboxes, and stops the exact App ID. Cleanup passes only after the App is
reported `stopped`, its container inventory is empty, and both its full active
Sandbox inventory and the owned-tag subset are empty. An already-stopped stop
error is resolved by this readback.

If an allocation reply was lost, a late resource can still appear after an
empty inventory. Recovery stops subsequently discovered owned resources but
continues to report `cleanup: unknown` while that allocation remains
unresolved. Retain the receipt and reconcile the outstanding request with the
provider; an empty snapshot or joined local worker cannot settle it. Receipt
write errors fail the attempt while cleanup still runs using the known owned
identities.

A result records `validator_report_status` as `passed`, `failed`, or `unknown`.
Only the complete passing report and remote exit zero can set
`validator_passed: true`. A complete, schema-valid failing report with remote
exit one retains only its checked stage prefix, status, exit code and fixed
failure reasons. Raw diagnostic text and observations are excluded. A metadata
child report with consistent version/dependency mismatch evidence adds the
fixed `rejection_kind: metadata_mismatch`; a checker/import failure, timeout or
output limit does not establish that rejection. A failed-image negative control
therefore needs that expected rejection evidence and independently verified
cleanup. An arbitrary nonzero exit or malformed/incomplete report remains
`unknown`, and every failed or unknown validator result still makes `run` exit
nonzero.

Exit 0 means the requested run or recovery passed its checks; exit 1 means
failure or unresolved cleanup; exit 130 reports an interruption after bounded
cleanup. Recovery does not rerun or regrade the CPU validator. The public
command and its offline tests do not qualify an actual image build, acquired
native libraries, cloud cleanup, model loading or GPU serving.

## Authentication and launch

Modal proxy authentication is required at ingress. Clients send
`Authorization: Bearer <proxy-id>.<proxy-secret>` over HTTPS to the selected
endpoint. Reject absent credentials before sending a request. Keep the
credential in the trusted client's memory, outside command arguments and
logs. The server mounts no secret and ignores ambient `API_KEY`; proxy
credentials are distinct from Modal deployment credentials. See
[Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth).

After the resource owner approves the image/model acquisition and runtime
allowance, run from `pipeline/` with its already-prepared locked environment:

```sh
uv run --frozen --no-sync modal serve --env main --timeout 1800 serve/vllm/serve_31b_qat.py
```

This command can build a Modal image, download the checkpoint and incur GPU
and storage charges. Record the exact app ID and selected environment from
the launch. Configure the client's base URL as the selected endpoint plus
`/v1`, and request the served model name above. Exercise an unauthenticated
denial and an authenticated readiness/model check before sending a workload.
The pinned upstream template has expected SHA256
`ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`;
verify the acquired template and record the actual derived image and kernel.

## Process ownership and stopping

`start_serve` launches the original server child and polls local `/health`.
Early exit, readiness timeout and local interruption fail startup. On failure
it terminates the child, waits, escalates to kill if needed, and waits again.
It returns only after readiness succeeds or the original failed child has
been joined. The existing health helper and legacy profile callers retain
their behavior.

After readiness, Modal owns the container and its server process. Local tests
do not qualify GPU-worker cleanup, descendants, shutdown after a successful
startup, connection loss or interrupted remote builds. Check these during the
live pilot. An HTTP health response also does not prove generation correctness
or that the proposed GPU allocation fits the workload.

The `modal serve` timer begins after initial image build and app publication.
Enforce a separate whole-job deadline from client process start. Function
timeouts and idle scaling are not an app lifetime or total spending ceiling.
The five-minute idle window leaves room for local task setup between readiness
and the first model request. Check readiness immediately before starting the
actor's time budget; setup that exceeds this window may require another cold
start, which must remain within the whole-job deadline.
On every terminal path, stop the recorded app ID and verify live inventory:

```sh
uv run --frozen --no-sync modal app stop --env main <recorded-app-id>
uv run --frozen --no-sync modal app list --env main --json
```

The dedicated `gemma4-31b-qat-pilot-hf-cache` and
`gemma4-31b-qat-pilot-vllm-cache` volumes persist after app shutdown. Record
their retention separately; stopping an app does not delete its cached data.

## Offline checks

With the pipeline's locked development dependencies already installed:

```sh
uv run --frozen --no-sync python -m unittest discover -s serve/vllm -p 'test_qat*.py' -v
uv run --frozen --no-sync pytest eval/test_eval_scoring.py -q
```

The tests check fixed inputs, proxy-auth declarations, resource limits, legacy
command defaults, interpreter-alias handling and failed-startup cleanup. They
also check the ordered repair plan, exact dependency-exception policy, extra
propagation, native-library failure handling and bounded child processes. They also import the profile with the installed Modal SDK without
hydrating resources. Their process execution uses temporary alias fixtures and
an authored local Python child; they perform no image build, model download,
deployment or inference.
