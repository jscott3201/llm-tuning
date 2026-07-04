# spec_decode — speculative-decoding reference tooling

A faithful PyTorch **reference harness** for DeepSeek's DSpark drafter on the
Gemma 4 family. It loads the public draft
[`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7)
and its `google/gemma-4-12B-it` target and runs one faithful **block-7 draft
forward** on Modal (B200), emitting a JSON fixture per prompt — base logits,
rank-256 Markov logits, confidence, and greedy draft tokens — so you can diff any
fast re-implementation (another framework, a hand-written kernel, a quantized
target) op-by-op.

Full writeup: [`../docs/dspark-spec-decode.md`](../docs/dspark-spec-decode.md).

## Use

```bash
cd spec_decode
uv sync
uv run modal run dspark_reference.py::preflight   # cheap CPU env + access check
uv run modal run dspark_reference.py::main --prompt "What is the capital of France?"
```

Needs a Modal secret `huggingface-secret` holding `HF_TOKEN` that can pull the
models. Fixtures land in `out/` (gitignored).

## Attribution

Builds on DeepSeek's [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT) and
its DSpark drafter. See the repo [README](../README.md#attributions).
