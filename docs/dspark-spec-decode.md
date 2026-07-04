# Speculative decoding for Gemma 4 with DeepSeek DSpark

A field guide to DeepSeek's **DSpark** speculative-decoding drafter for the
Gemma 4 family — what the released draft model actually is, how it plugs into a
target model, and how to get a faithful PyTorch reference so you can port it into
a fast runtime without chasing ghosts. Companion to
[`spec_decode/dspark_reference.py`](../spec_decode/dspark_reference.py).

Everything here is derived from public sources: the MIT-licensed
[`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec) repo and the
public checkpoint
[`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7).

## TL;DR

- **DSpark is a draft-model architecture + an offline training recipe, not a
  kernel and not a serving plugin.** DeepSpec trains drafts (DSpark / DFlash /
  Eagle3) and evaluates acceptance offline in PyTorch. It ships *no* custom CUDA
  kernels and *no* vLLM/SGLang integration. Making it fast is on you.
- The released `dspark_gemma4_12b_block7` is a **5-layer, 3.43 B** draft that
  predicts a **block of 7 tokens** per step, reads **five tapped hidden layers**
  of the `gemma-4-12B-it` target (EAGLE-3 style), and refines its own logits with
  a **rank-256 Markov head** plus a **confidence head**.
- Because the draft is coupled to a *specific* target's hidden states, you cannot
  drop a 12B draft onto a different target (e.g. a smaller Gemma 4) — you retrain.
- The single most useful thing before porting DSpark into a fast loop is a
  **byte-faithful PyTorch reference**. A drafter whose math is even slightly off
  shows **0% acceptance** at any speed. The reference turns that mystery into an
  op-by-op diff.

## Anatomy of the released draft

From the checkpoint's `config.json` (`architectures: ["Gemma4DSparkModel"]`):

| Field | Value | Meaning |
|---|---|---|
| draft layers / hidden | 5 / 3840 | small Gemma-4-shaped draft stack (3.43 B params) |
| `block_size` | 7 | tokens proposed per verify step |
| `target_layer_ids` | `[5, 17, 29, 41, 46]` | which target decoder layers are tapped |
| `num_target_layers` | 48 | target is `gemma-4-12B-it` |
| `enable_confidence_head` | true | per-token accept-probability head |
| `confidence_head_with_markov`, `markov_rank` | true, 256 | rank-256 low-rank token-transition correction |
| `num_anchors` | 512 | **training-only** (sampled anchor blocks per sequence) |
| `mask_token_id` | 4 | placeholder id filling unproposed draft slots |
| dtype | bfloat16 | load bf16 end-to-end |

### How the draft reads the target

After a target forward with `output_hidden_states=True`, DSpark concatenates the
**raw** (un-normalized) outputs of the tapped layers into one context tensor:

```
context = concat( hidden_states[i+1]  for i in [5,17,29,41,46] )   # [B, S, 5*H]
```

Inside the draft this is projected `5H → H` by a linear `fc`, then RMS-normalized
by `hidden_norm`, and fed as **cross-attention keys/values in every draft layer**
(the draft's own block positions are the queries). Two consequences worth
internalizing:

1. **The taps are raw, not normalized.** The draft normalizes them itself. Feed
   it the final, already-normalized hidden state and it diverges — which is why
   DeepSpec explicitly forbids the last layer index from appearing in the taps.
2. **The draft is target-specific.** Those five layer indices, the `5*H` width,
   and the learned `fc` all assume *this* target's geometry. A different target =
   a different draft.

### The block-7 forward

Per step the draft builds a length-7 input: slot 0 is the current committed
token (the "anchor"), slots 1–6 are the mask token (id 4). One non-causal SDPA
forward over `[fused context ⊕ 7-token block]` yields seven hidden states, and
then three heads run:

- **Base logits** — `compute_logits`, i.e. `lm_head` **with the Gemma final-logit
  softcap** (`tanh(x/c)·c`). Never a bare `lm_head`; the softcap changes argmax.
- **Markov correction** — a rank-256 bias `W2(W1[prev])` is added to the base
  logits **autoregressively within the block** (each position conditions on the
  previously chosen token). This is what actually selects the greedy draft
  tokens; ignoring it changes both the tokens and their probabilities.
- **Confidence** — a small head predicts a per-position accept logit; the
  cumulative product of its sigmoids estimates prefix-acceptance probability and
  can early-stop the block.

### Verification (acceptance)

DeepSpec's evaluator verifies with **standard speculative rejection sampling**:
run the target once over `[anchor ⊕ draft tokens]`, accept the longest prefix
where `rand < min(1, P_target/P_draft)`, then take a residual/bonus token. At the
default `temperature=1.0` this **preserves the target distribution** (it is *not*
greedy-identity). At `temperature→0` both distributions collapse to one-hot and
acceptance reduces to plain token-matching. Which regime you want depends on your
serving contract — if you must preserve a target's *exact greedy* output under a
quality guardrail, run the verify step greedy and accept on argmax match.

## Porting it into a fast runtime

DSpark gives you weights and a reference forward. A production speed win needs the
inference path DeepSpec doesn't provide:

1. **Register the draft** as a proposer in your engine (vLLM/SGLang have EAGLE-3
   style hooks; DSpark's multi-layer taps map onto them). Reproduce the tap
   concat → `fc` → `hidden_norm` fusion, the block-7 propose, and the Markov +
   confidence heads exactly.
2. **Graph the block loop.** The per-token Python/launch overhead of seven draft
   steps dwarfs the tiny matmuls on a single stream; capturing the block in one
   replayed CUDA graph (with ping-pong KV buffers) is where most of the wall-clock
   win lives.
3. **Don't pay the full vocab every step.** The 262 k-way projection on each of
   seven draft steps is the bottleneck; the rank-256 Markov head is one principled
   way to get candidate logits cheaply.

Do all of that and *then* measure — but only after the reference confirms your
port is correct.

## The reference harness

[`spec_decode/dspark_reference.py`](../spec_decode/dspark_reference.py) loads the
public draft + target on a single **B200** (the bf16 12B + draft + hidden states
don't fit 24 GB), runs one faithful block-7 forward per prompt, and dumps a JSON
fixture: `input_token_ids`, the tap/hidden shapes, top-k **base** and **Markov**
logits per block position, the **confidence** logits, and the **greedy draft
tokens**. Point any re-implementation (another framework, a hand-written kernel,
a quantized target) at the same prompts and diff:

- Draft tokens never in the reference's top-k → your **draft forward** is wrong
  (tap offset, missing softcap, skipped Markov correction, wrong RMSNorm, ...).
- Draft matches the reference but acceptance is still low against *your* target →
  a **target mismatch** (e.g. drafting against a quantized target whose tapped
  hidden states drifted off the bf16 distribution the draft trained on).

That split — is it the draft, or is it the target it's drafting against? — is the
whole game, and it's invisible until you have a reference.

## Reproduce

```bash
# one-time: a Modal secret 'huggingface-secret' with an HF_TOKEN that can pull
# the models.

# cheap CPU preflight (imports + model access), then the B200 forward:
modal run spec_decode/dspark_reference.py::preflight
modal run spec_decode/dspark_reference.py::main --prompt "What is the capital of France?"
```

## Licensing

DeepSpec is MIT. The Gemma 4 base is Apache-2.0 (as used in this repo); the DSpark
checkpoint carries DeepSeek's own license. Credit DeepSeek's DeepSpec and respect
each. This harness only *loads and runs* the public weights; it redistributes none
of them.
```
