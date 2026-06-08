# A/B coding benchmark results

Single-user serving numbers for the four template variants on the `coding`
prompt profile (8 short coding prompts, 512 max tokens each, concurrency 1).
These are the author's own single-run measurements on one B200. They are not a
controlled benchmark suite. Treat them as a sanity check, not a leaderboard.

## The variants

- **upstream** (`upstream-coding.json`) — stock `Qwen/Qwen3.6-27B` chat template,
  served as-is.
- **fixed** (`fixed-coding.json`) — the custom fork (`custom_pub_chat_template_qwen36.jinja`)
  served with its agentic defaults on.
- **fixedreal** (`fixedreal-coding.json`) — the fork served against the real
  upstream-tokenizer baseline. This is the apples-to-apples comparison: same
  tokenizer, same model, fork template vs stock template.
- **-warm** (`fixed-coding-warm.json`, `fixedreal-coding-warm.json`) — a second
  pass after the server is warm, so the first-token cost is not paying for cold
  CUDA-graph capture or weight load.

## How to read the numbers

Each file has one `levels` entry (concurrency 1) with a `summary` and the
per-prompt `samples`.

- `ttft_p50` / `ttft_p95` — time to first token, median and tail, seconds.
- `decode_tps_p50` — per-request decode tokens/sec.
- `aggregate_tps` — total output tokens / wall seconds.
- `prompt_tokens` per sample — the prefill token count.

The prefill token counts are identical across every variant
(`[30, 36, 26, 66, 36, 39, 32, 29]`). That is the point of the byte-identity
invariant: on well-formed coding prompts the fork tokenizes to the exact same
prompt the stock template does, so it costs nothing on the prefill side. The
fork's extra IMPORTANT bullets (Q6) only appear when tools are sent, which the
`coding` profile does not.

The cold `fixed` run has a high `ttft_p95` (5.3s) and lower aggregate TPS. That
is cold-start noise, not template overhead. The warm runs land on top of
upstream: `fixedreal-coding-warm` ttft_p50 0.226s vs upstream 0.347s, decode
within noise. The fork is performance-neutral at single user.

## Reproducing

Point a live SGLang or vLLM endpoint at each template, then run the bench
tooling against it. The endpoint defaults to `$ENDPOINT` or
`http://localhost:8000`.

```bash
export ENDPOINT=http://localhost:8000
# bench each variant against its own served endpoint, profile=coding,
# requests=8, max_tokens=512, temperature=0.6, top_p=0.95, concurrency=1
```

For the message-shape and multi-turn checks, use the live probe in the parent
directory:

```bash
uv run --with openai python ../live_agentic_probe.py \
  --endpoint http://localhost:8000 --model qwen3.6-27b
```

## Sources

The patches these results test are documented against public reports:

- [earendil-works/pi#3325](https://github.com/earendil-works/pi/issues/3325)
- [QwenLM/Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475)
- [block/goose#6883](https://github.com/block/goose/issues/6883)
- [ollama/ollama#14493](https://github.com/ollama/ollama/issues/14493)
- [sudoingX gist](https://gist.github.com/sudoingX/c2facf7d8f7608c65c1024ef3b22d431)
