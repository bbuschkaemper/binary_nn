# Discrete FFN Experiment Note

Last updated: 2026-03-13

This document formulates a general research idea for a new experimental
repository.

The goal is to define a clean first-principles experiment around a discrete,
packed feed-forward network (FFN) training method that is related to, but more
aggressive than, BitNet-style low-bit training.

## 1. One-Sentence Summary

Replace standard FFN weights with directly trained discrete signed masks,
represented as two binary channels that encode positive and negative support,
and update those discrete states directly from accumulated batch statistics
instead of maintaining hidden floating-point master weights.

## 2. Core Idea

The motivating intuition is:

- standard transformers spend a large fraction of their compute inside FFN
  matrix multiplies
- FFN weights may tolerate more aggressive discretization than the rest of the
  network
- if the discrete state is the actual parameter, training might not need a
  shadow floating-point weight that is repeatedly quantized back down
- packed binary storage could enable custom GPU kernels based on word-level
  bit operations

The proposal is easiest to state in terms of weights, not neurons.

Instead of learning a real-valued FFN weight matrix `W`, define two binary mask
matrices:

- `P in {0,1}^{m x n}` for positive support
- `N in {0,1}^{m x n}` for negative support

and interpret the effective weight as:

$$
W = P - N
$$

with an exclusivity constraint:

$$
P \odot N = 0
$$

This means each effective weight is ternary:

$$
W_{ij} \in \{-1, 0, +1\}
$$

So the clean formulation is not "true or false weights" in the ordinary sense.
It is a dual-rail binary encoding of ternary signed weights.

## 3. Why This Is Adjacent to BitNet

BitNet is the closest existing line of work, but this idea is not identical to
BitNet.

BitNet and BitNet b1.58:

- replace linear layers with low-bit `BitLinear` layers
- use ternary weights `{-1, 0, +1}` via absmean quantization
- use low-bit activations during forward
- still rely on standard gradient-based optimization of a trainable model
- in practice keep some tensors such as norms and often embeddings in higher
  precision

This proposed experiment differs in one important way:

- the discrete state itself is the parameter being optimized
- updates are applied directly to the discrete state rather than to an unseen
  floating-point master copy

That makes this closer to a shadow-free discrete optimization scheme than to a
conventional quantization-aware training recipe.

## 4. Relevant BitNet Sources

Primary sources worth reading before implementing anything:

- BitNet repository: <https://github.com/microsoft/BitNet>
- BitNet b1.58 model card: <https://huggingface.co/microsoft/bitnet-b1.58-2B-4T>
- The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits:
  <https://arxiv.org/abs/2402.17764>
- BitNet b1.58 2B4T Technical Report:
  <https://arxiv.org/abs/2504.12285>
- bitnet.cpp inference paper:
  <https://arxiv.org/abs/2410.16144>

Useful implementation details from those sources:

- BitNet b1.58 uses ternary weights, not strict binary weights
- activations are quantized separately from weights
- the best published results come from native low-bit training, not from a late
  post-training quantization pass
- the practical inference wins depend on specialized kernels rather than on
  generic framework execution alone

## 5. Precise Parameterization

Consider an FFN linear map with input `x` and output `y`:

$$
y = Wx
$$

Under the dual-mask parameterization:

$$
y = (P - N)x = Px - Nx
$$

This has two immediate consequences:

1. The effective weight set is ternary.
2. Efficient bitwise execution is easiest only if activations are also low-bit.

If `x` stays in BF16 or FP32, then `Px` and `Nx` still mean summing selected
input values, so the kernel is not a pure XNOR-popcount binary network.

Therefore there are two regimes:

### 5.1 Discrete weights, real-valued activations

Pros:

- simpler to stabilize
- closer to standard transformer training
- easier first experiment

Cons:

- kernel efficiency story is weaker
- custom GPU packing may not pay off immediately

### 5.2 Discrete weights, low-bit activations

Pros:

- stronger systems story
- more opportunity for packed `uint32` kernels and bitwise arithmetic

Cons:

- much harder optimization problem
- higher risk of accuracy collapse early in training

For a first experiment, the safer path is discrete weights with higher-precision
activations.

## 6. Proposed Training Rule

The most interesting part of the idea is the training rule.

Instead of:

- keeping a real-valued master weight
- taking an optimizer step in float space
- quantizing back to a discrete representation for forward

use the discrete state itself as the optimized object.

For a weight `w_ij in {-1, 0, +1}`, accumulate a batch-level credit statistic
over a large update window. A natural first choice is the usual outer-product
signal from backpropagation:

$$
G = \sum_b \delta_b x_b^\top
$$

where:

- `x_b` is the input activation to the FFN projection
- `delta_b` is the backward signal at the projection output or preactivation

Then choose the next discrete state directly by minimizing a simple surrogate:

$$
w_{ij}^{new} = \arg\min_{s \in \{-1,0,+1\}} G_{ij}s + \lambda |s|
$$

which gives the thresholding rule:

$$
w_{ij}^{new} =
\begin{cases}
+1 & \text{if } G_{ij} < -\lambda \\
0 & \text{if } |G_{ij}| \le \lambda \\
-1 & \text{if } G_{ij} > \lambda
\end{cases}
$$

Interpretation:

- strong negative gradient evidence turns a connection on positively
- strong positive gradient evidence turns a connection on negatively
- weak evidence prunes the connection to zero

This is a form of direct discrete optimization or periodic state selection,
not a standard optimizer in floating-point parameter space.

## 7. How To Encode It Efficiently

The systems idea is to pack each binary mask into machine words such as
`uint32`.

For each block of 32 weights:

- one `uint32` stores the positive-mask bits
- one `uint32` stores the negative-mask bits

That means 32 ternary weights can be stored in 64 bits, or 2 bits per weight in
storage, while still enabling word-level logical operations.

Potential execution strategies:

1. Real-valued activations:
   expand the set bits and sum corresponding inputs for `P`, then subtract the
   sum for `N`
2. Binary or few-bit activations:
   use packed logical ops and population counts to emulate multiply-accumulate
   more efficiently

The first strategy is the easier research baseline.
The second is the more ambitious hardware-aligned direction.

## 8. Why The Idea Is Plausible

The idea makes technical sense for several reasons.

- Ternary weight sets `{-1, 0, +1}` already have strong precedent from BitNet.
- FFNs dominate a large share of transformer arithmetic cost.
- Zero-valued connections provide built-in sparsity pressure.
- Large-batch accumulation before discrete updates could reduce the instability
  of per-sample bit flips.
- The proposal is modular: it can be tested in FFNs first without changing the
  rest of the network.

## 9. What Is Most Likely To Fail

Several parts of the idea are plausible in principle but risky in practice.

### 9.1 Optimizing neurons instead of weights is the wrong abstraction

The natural optimized object is the connection weight or mask entry, not the
neuron as a whole.

### 9.2 Pure bitwise speedups are not automatic on GPUs

Packing into `uint32` does not guarantee faster execution.

Without carefully designed kernels, the costs of:

- unpacking bits
- irregular access to selected inputs
- reductions across many bit positions

can erase the theoretical gain.

### 9.3 Training without any higher-precision statistics is unlikely to work

Even if the model weights are discrete, it is still likely necessary to keep:

- activations in BF16 or FP32 at first
- backpropagated error signals in higher precision
- accumulated update statistics in higher precision

The strongest version of the proposal is therefore not "fully bitwise training"
on day one. It is "discrete model parameters with higher-precision learning
signals".

### 9.4 Direct bit updates may oscillate

If a weight is updated too often, it may flip between `-1`, `0`, and `+1`.
This likely requires:

- update windows spanning many minibatches
- confidence thresholds
- hysteresis or cooldown rules
- possibly an EMA over batch statistics before state changes

### 9.5 Applying it everywhere is too ambitious for a first study

Attention, embeddings, norms, and output heads have different sensitivity
profiles. The first experiment should isolate FFNs only.

## 10. Strongest First Experiment

For a new experimental repository, the best first study is narrow.

### Phase 1: Dense FFN-only replacement

- use a small transformer or MLP benchmark
- replace only FFN linear layers with dual-mask ternary weights
- keep attention, embeddings, norms, and output projections in BF16
- keep activations in BF16
- use direct discrete updates only on FFN weights

Success criterion:

- the model trains stably at all
- the ternary FFN variant reaches a competitive fraction of the BF16 baseline

### Phase 2: Add packed kernels

- implement a packed storage format for `P` and `N`
- benchmark memory usage and throughput independently from model quality
- compare naive tensor implementation versus custom packed kernel

Success criterion:

- measurable reduction in memory bandwidth or kernel time for FFN layers

### Phase 3: Add low-bit activations

- quantize FFN activations only
- test whether the systems benefit justifies the optimization cost

Success criterion:

- performance degradation remains bounded while runtime improves materially

## 11. Naming The Method

Avoid calling it simply a "1-bit FFN" because that is imprecise.

More accurate names:

- dual-rail binary FFN
- shadow-free ternary FFN
- direct discrete FFN training
- packed signed-mask FFN
- bit-flip FFN optimization

Of these, "shadow-free ternary FFN" is the clearest research label.

## 12. Recommended Minimal Research Question

The cleanest first research question is:

> Can transformer FFN weights be trained directly in a ternary signed-mask
> parameterization, without floating-point master weights, while preserving a
> useful fraction of baseline model quality and creating a path to packed
> low-bandwidth kernels?

That question is precise, testable, and close enough to BitNet to inherit
useful intuition, while still being materially different from existing
quantization-aware training work.

## 13. Bottom-Line Assessment

The idea makes sense.

The strongest version of it is not:

- "all training becomes pure bitwise logic"

The strongest version is:

- FFN weights are directly represented as ternary signed masks
- discrete state changes are learned from accumulated batch evidence
- higher-precision activations and learning statistics are retained initially
- packed kernels are pursued as a second-stage systems optimization

That is a coherent experimental direction for a new general neural network
research repository.
