# 1-Bit Neural Networks: State of the Art

Last updated: 2026-03-16

This note summarizes the current state of the art in 1-bit neural networks with
an emphasis on large language models, especially Microsoft BitNet, while also
tracking newer work on broader binary neural network (BNN) training and
hardware.

## Document Role

Use this file for external research context, not for the repo's current local
implementation state. It answers what the broader field appears to believe right
now.

## 1. Scope

There are now two related but meaningfully different research tracks:

- native 1-bit or ternary LLMs, where the model is designed and trained for
  ultra-low precision from the start
- broader binary neural networks, where both weights and activations are binary
  or near-binary, often for vision, edge inference, or neuromorphic settings

For this repository, the first track is the most relevant systems baseline and
the second track is the most relevant optimization baseline.

## 2. Short Answer

As of March 2026, the strongest public evidence for state of the art in native
1-bit LLMs still comes from the BitNet line:

- `BitNet: Scaling 1-bit Transformers for Large Language Models` established the
  core `BitLinear` recipe and the claim that 1-bit training can follow scaling
  behavior similar to full precision.
- `The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits` refined
  that idea into `BitNet b1.58`, using ternary weights in `{-1, 0, +1}` and
  showing that 1.58-bit training is the practical regime rather than strict
  binary-only weights.
- `BitNet b1.58 2B4T Technical Report` is the current public flagship result:
  an open 2B-parameter native 1-bit model trained on 4T tokens, with benchmark
  results presented as competitive with strong open full-precision models in the
  same size band.
- `bitnet.cpp` and the Microsoft `BitNet` repository are the strongest public
  evidence that 1-bit models are now a real systems target rather than just a
  quantization curiosity.

Outside BitNet, the most important broader trend is that training remains the
main bottleneck. Newer papers are increasingly attacking the root issue: how to
optimize discrete models without relying entirely on hidden floating-point
weights and straight-through estimators.

## 3. What "1-Bit" Means in Practice

The term is now overloaded.

- In older BNN literature, "1-bit" often means strict binary weights and binary
  activations.
- In the current LLM literature, the strongest results are usually not strict
  `{-1, +1}` weights. They are ternary `{-1, 0, +1}` weights, which correspond
  to about `1.58` bits per parameter.

That distinction matters for this repository. If the goal is to learn from what
actually works at scale today, the relevant public SOTA is mostly ternary, not
strictly binary.

## 4. Microsoft BitNet Line

### 4.1 Foundational paper

`BitNet: Scaling 1-bit Transformers for Large Language Models`

- arXiv: <https://arxiv.org/abs/2310.11453>
- Main contribution: introduces `BitLinear` as a drop-in replacement for linear
  layers and argues that 1-bit transformers can scale stably.
- Why it matters: this is the paper that turns binary/1-bit ideas from a small
  model compression story into a language-model scaling story.

### 4.2 Shift from strict 1-bit to 1.58-bit

`The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits`

- arXiv: <https://arxiv.org/abs/2402.17764>
- Main contribution: reframes the practical frontier around ternary weights
  `{-1, 0, +1}` rather than strict binary-only weights.
- Important claim: BitNet b1.58 can match full-precision models of the same
  size and training-token budget while improving memory, latency, throughput,
  and energy.
- Why it matters: this paper is the real conceptual pivot for the field. In
  practice, the public frontier is no longer "binary versus float". It is
  "native ternary training versus post-training quantization".

### 4.3 Activation-side improvement

`BitNet a4.8: 4-bit Activations for 1-bit LLMs`

- arXiv: <https://arxiv.org/abs/2411.04965>
- Main contribution: pushes the activation path below 8 bits by combining 4-bit
  activation quantization with sparsification around outlier channels.
- Claimed outcome: performance comparable to BitNet b1.58 at similar training
  cost, with faster inference from INT4 or FP4-capable kernels.
- Why it matters: it suggests the real deployment frontier is not only weight
  binarization, but end-to-end low-bit system design including activations and
  KV-cache choices.

### 4.4 Systems maturity: CPU and edge inference

`1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs`

- arXiv: <https://arxiv.org/abs/2410.16144>
- Main contribution: specialized CPU kernels for BitNet b1.58 inference.
- Reported result: about `2.37x` to `6.17x` speedup on x86 and `1.37x` to
  `5.07x` on ARM.
- Why it matters: this is the first strong public indication that 1-bit LLMs
  can get real wins from architecture-aware kernels, not just paper-level model
  compression.

`Bitnet.cpp: Efficient Edge Inference for Ternary LLMs`

- arXiv: <https://arxiv.org/abs/2502.11880>
- Main contribution: formalizes the `bitnet.cpp` inference stack around ternary
  lookup-table and `I2_S` kernels for lossless sub-2-bit inference.
- Reported result: up to `6.25x` speedup over full-precision baselines and up
  to `2.32x` over low-bit baselines.
- Why it matters: the inference story is now concrete enough to shape research
  priorities. Any new training idea in this repo should assume that kernel and
  packing choices are part of the model-design problem.

### 4.5 Current flagship model

`BitNet b1.58 2B4T Technical Report`

- arXiv: <https://arxiv.org/abs/2504.12285>
- Main contribution: releases the first open native 1-bit LLM at roughly the
  2B scale trained on 4T tokens.
- Architecture details highlighted in the paper:
  - `BitLinear` layers with ternary weights via absmean quantization
  - 8-bit activations via per-token absmax quantization
  - `subln` normalization
  - `ReLU^2` in the FFN instead of SwiGLU
- Reported comparison summary:
  - non-embedding memory around `0.4 GB`
  - CPU decoding latency reported as `29 ms` time-per-output-token in the paper
  - estimated energy `0.028 J`
  - average benchmark score reported as competitive with strong open 1B-2B
    instruction-tuned models
- Most important claim: this is the current strongest public evidence that
  native ultra-low-precision training can stay near the performance frontier for
  small open LLMs while moving the memory and efficiency frontier materially.

## 5. What the Microsoft Repository Adds Beyond the Papers

Repository: <https://github.com/microsoft/BitNet>

What stands out in the repo today:

- it is not just a paper companion; it is an inference stack with model
  packaging, CPU kernels, GPU kernels, and multiple supported model layouts
- the repository explicitly positions `bitnet.cpp` as the official inference
  framework for 1-bit LLMs
- the repo now includes official GPU inference material for `W1.58A8`
  inference, not just CPU paths
- the code and README make it clear that the public deployment target is really
  ternary-weight plus low-bit-activation inference, not pure textbook BNNs

For this repo, that means any "foundation" work should keep three things in
view from day one:

- the discrete representation has to be training-compatible
- the representation has to have a packing story
- the representation should map to specialized kernels rather than generic GEMM

## 6. Broader SOTA Beyond BitNet

### 6.1 Native training still beats PTQ at the extreme low end

Two broader papers are especially relevant here.

`BiLLM: Pushing the Limit of Post-Training Quantization for LLMs`

- arXiv: <https://arxiv.org/abs/2402.04291>
- Main contribution: a strong post-training 1-bit quantization method for
  existing LLMs using salient-weight selection and binary residual
  approximation.
- Why it matters: it is one of the best reminders that PTQ remains attractive
  because it avoids the cost of retraining from scratch.
- Limitation relative to BitNet: the BitNet b1.58 line argues that native
  low-bit training still dominates the quality-efficiency tradeoff once the bit
  budget becomes this aggressive.

`Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs
with 100T Training Tokens`

- arXiv: <https://arxiv.org/abs/2411.17691>
- Main contribution: studies more than 1500 quantized checkpoints and argues
  that low-bit quantization degrades highly trained models more severely than
  undertrained ones.
- Why it matters: this is a useful warning against naive extrapolation. Just
  because current small models quantize well does not mean future heavily
  trained frontier models will do so gracefully.
- Practical implication: native discrete training methods may become even more
  important as token counts rise.

### 6.2 Broader binary-network training is attacking the optimization problem

This is the most important non-LLM trend for this repository.

`BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training`

- arXiv: <https://arxiv.org/abs/2512.04189>
- Main contribution: proposes a discrete analogue of backpropagation that
  propagates binary error signals and performs forward and backward computation
  with bitwise operations.
- Why it matters: this is directly aligned with the repo's research direction.
  It tries to reduce or remove dependence on floating-point latent weights during
  training.

`Layerwise Progressive Freezing Enables STE-Free Training of Deep Binary Neural
Networks`

- arXiv: <https://arxiv.org/abs/2601.22660>
- Main contribution: introduces `StoMPP`, a layerwise stochastic progressive
  binarization method that avoids the straight-through estimator.
- Why it matters: it shows that the field is actively searching for stable deep
  BNN optimization without relying on classic STE approximations.

`Quadratic Unconstrained Binary Optimisation for Training and Regularisation of
Binary Neural Networks`

- arXiv: <https://arxiv.org/abs/2601.00449>
- Main contribution: reformulates BNN training as QUBO with explicit
  regularization strategies.
- Why it matters: it treats binary training as a discrete optimization problem,
  not a quantized surrogate of floating-point learning.
- Relevance here: conceptually this is close to the repository's current idea of
  direct state selection instead of a shadow floating-point master weight.

### 6.3 Hardware and verification continue to matter

Recent broader BNN work is also moving along hardware and trustworthiness axes:

- `Scalable Digital Compute-in-Memory Ising Machines for Robustness
  Verification of Binary Neural Networks`, arXiv: <https://arxiv.org/abs/2603.05677>
- `Robustness Verification of Binary Neural Networks: An Ising and
  Quantum-Inspired Framework`, arXiv: <https://arxiv.org/abs/2602.13536>
- `PiC-BNN: A 128-kbit 65 nm Processing-in-CAM-Based End-to-End Binary Neural
  Network Accelerator`, arXiv: <https://arxiv.org/abs/2601.19920>

These are not direct LLM baselines, but they reinforce a consistent pattern: the
value proposition of binary networks only becomes compelling when optimization,
representation, and hardware are treated as one stack.

## 7. Current Consensus

The current public consensus appears to be:

- the best open "1-bit" LLM results are really ternary-weight results
- native low-bit training is stronger than post-training quantization at the
  extreme low-bit frontier
- activation quantization and kernel design matter almost as much as the weight
  format
- inference is no longer the main blocker; training remains the harder research
  problem
- straight-through-estimator-style training is still dominant in practice, but
  active work is trying to replace it with more principled discrete
  optimization

## 8. What Is Still Missing

Several gaps remain visible even after the recent progress.

### 8.1 No verified open native 1-bit model beyond the 2B BitNet scale

The biggest public native result remains `BitNet b1.58 2B4T`. That is a strong
result, but it is still far smaller than the scales at which people now judge
general-purpose LLM capability.

### 8.2 The training recipe is still not truly discrete end to end

BitNet is native low-bit training, but the broader field still relies heavily on
surrogate gradients, floating-point optimizer state, and mixed-precision
training infrastructure.

This leaves a clear research opportunity for methods that update discrete states
more directly.

### 8.3 Long-context and adaptation stories are still immature

The public literature is much stronger on base-model pretraining and inference
than on:

- efficient low-bit fine-tuning
- long-context stability
- robust preference optimization at low precision
- multimodal 1-bit architectures

### 8.4 Robustness and formal behavior are less mature than throughput results

The newest BNN verification work suggests that robustness is becoming a live
issue, but the LLM-focused 1-bit literature is still much stronger on
performance and efficiency than on adversarial or safety-critical guarantees.

## 9. Implications for This Repository

If this repository is trying to build a new method rather than reproduce BitNet,
the most defensible direction is not to fight the public SOTA head-on on scale.
It is to push on the most open technical weakness:

- training discrete weights without leaning fully on shadow floating-point
  weights

That makes the following comparisons especially important:

- BitNet and BitNet b1.58 as the systems and scaling baseline
- BiLLM as the strongest reminder that PTQ remains a tough practical competitor
- BEP, StoMPP, and QUBO-based training as signals that the field is moving
  toward direct discrete optimization

## 10. Bottom Line

The present state of the art is best described like this:

- for open native 1-bit LLMs, BitNet b1.58 is the public frontier
- for deployment, `bitnet.cpp` makes the efficiency story credible on real
  hardware
- for research opportunity, the biggest open problem is still optimization of
  discrete parameters during training

That is a good fit for the current repo idea. The strongest gap is no longer
"can 1-bit models run fast?" but "can we train discrete models in a more direct,
stable, and hardware-aligned way than the current recipes?"
