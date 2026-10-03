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

The registry supplies the pinned model and resource defaults. The profile's
constants select the immutable image and lifecycle limits. It clears the
image entrypoint and mounts `_common`; it adds no package installation or
Python injection step.

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
uv run --frozen --no-sync python -m unittest discover -s serve/vllm -p 'test_qat_profile.py' -v
uv run --frozen --no-sync pytest eval/test_eval_scoring.py -q
```

The profile tests check immutable inputs, proxy-auth declarations, resource
limits, legacy command defaults and failed-startup cleanup. They also import
the profile with the installed Modal SDK without hydrating resources. Their
only process execution is an authored local Python child; they perform no
image build, model download, deployment or inference.
