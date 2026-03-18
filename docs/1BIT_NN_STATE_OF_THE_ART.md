# 1-Bit Neural Networks: State of the Art For A GPU-First Training Search

Last updated: 2026-03-18

This note summarizes the outside literature from the perspective of the repository's
new goal: discover a low-bit training idea that can improve GPU training and
eventually GPU inference, without giving up quality.

## Document Role

Use this file for external framing, not for the repository's local implementation
state.

## 1. Scope

The most relevant external threads are now:

- native low-bit LLM work, especially the BitNet line
- broader binary and ternary training papers that attack the optimization problem
- systems papers that show where real speed wins do and do not come from

For this repository, the important question is no longer simply “are 1-bit or ternary
models possible?” The more relevant question is “what ideas from the field point toward
a real GPU quality/speed improvement?”

## 2. Short Answer

As of March 2026, the public frontier still says three things clearly.

1. The practical low-bit frontier is mostly ternary (`1.58-bit`), not strict binary.
2. Speed wins only become real when the model representation and the kernel stack are
   designed together.
3. Training remains the most open problem. Public papers show that low-bit models can be
   competitive, but there is still little open evidence for a training rule that
   clearly lowers GPU training cost while preserving quality.

That last gap is exactly where this repository can contribute.

## 3. What The Public Frontier Already Says

### 3.1 BitNet is still the main practical reference line

The BitNet family remains the strongest public reference for native low-bit LLM work.

- `BitNet: Scaling 1-bit Transformers for Large Language Models`
  - establishes `BitLinear` and the scaling claim
- `The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits`
  - reframes the practical frontier around ternary weights `{-1, 0, +1}`
- `BitNet b1.58 2B4T Technical Report`
  - strongest open flagship result so far for native low-bit LLM training

The most relevant lesson for this repo is not “strict binary wins.” It is:

- native ternary training is the public regime that currently looks most plausible for
  serious quality

### 3.2 Activation precision matters for real systems wins

`BitNet a4.8` matters because it makes the activation path part of the story.

That is important for this repo because a weight-only story is probably not enough to
create a full GPU speed win. If the training idea never touches activations or operator
shapes, it may preserve quality without becoming a good systems target.

### 3.3 Inference wins come from co-design, not from bits alone

The CPU-focused BitNet systems papers (`1-bit AI Infra` and `bitnet.cpp`) are still
relevant even though this repo has moved to a GPU-first goal.

Their key lesson is general:

- low-bit representations only become fast when the kernel stack is designed for them
- speed claims based on generic framework execution are weak compared with
  representation-aware operator implementations

For GPU work, that means a training idea should try to produce the representation a
future kernel actually wants, not just a numerically quantized checkpoint.

## 4. Broader BNN Training Lessons

Outside the BitNet line, the most interesting work is now the work that attacks the
optimization problem directly.

Useful examples:

- binary backward or binary-error ideas such as `BEP`
- progressive or staged discrete-training schemes such as layerwise progressive freezing
- explicitly discrete formulations such as QUBO-style training views

The shared lesson is not that any one of these methods is already the winner.
Instead, the lesson is:

- the field now sees training, not only inference, as the main unresolved bottleneck
- staged or schedule-based discrete updates are a plausible direction

That is especially relevant to the repository's current local evidence, where the best
quality comes from projected / STE training and the most interesting novelty comes from
shadow-free direct state updates.

## 5. What The Field Still Lacks

From the perspective of this repository's new goal, the public literature is still thin
in four places.

### 5.1 Clear GPU training-speed evidence

There are strong public quality and inference stories, but little open evidence for a
low-bit training rule that clearly improves GPU training wall-clock behavior in a way
that survives repeated measurement.

### 5.2 A method that reduces training overhead, not just model size

Many low-bit recipes reduce parameter precision but still keep substantial hidden
floating-point state and optimizer cost. That can preserve quality while leaving GPU
training economics mostly unchanged.

### 5.3 Tight alignment between training representation and inference representation

Many training stories still look like “train with floating-point machinery, then deploy
a quantized approximation.” The repo's new goal is stronger than that. The training
representation should be close to the inference representation from the beginning.

### 5.4 Open, decision-grade measurement methodology

The field still has too many isolated latency numbers and not enough repeated,
apples-to-apples measurement of quality, runtime, and implementation complexity.

## 6. Implications For This Repository

The external literature now suggests a fairly specific local strategy.

- Use ternary, not strict binary, as the main quality-oriented low-bit regime.
- Keep projected / STE ternary as the quality anchor because that is the most
  BitNet-aligned local path.
- Treat shadow-free direct-discrete ideas as the source of training-rule novelty rather
  than as the current final answer.
- Treat activation precision as first-class once the next training-rule experiment is in
  place.
- Make GPU measurement quality part of the research contribution, not just a support
  detail.

## 7. Working Repository Hypothesis

The repository's current best hypothesis is that the next useful idea will look like:

- projected-family quality
- shadow-free-style discrete-state intuition
- refresh-scheduled or staged state updates
- representation stability that future GPU kernels can exploit

In other words, the likely contribution is not “another quantized baseline.” It is a
better low-bit training procedure.

## 8. Bottom Line

The outside world already tells us that low-bit models can be good and can be fast.
What it does **not** yet clearly give us is an open, GPU-first training idea that wins
on both quality and speed. That is the gap this repository should now attack directly.
