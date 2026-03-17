# Roadmap

Last updated: 2026-03-17

This document records the current next-step plan and the open design decisions.
It is meant to be actionable rather than historical.

## Document Role

Use this file after `CURRENT_STATUS.md`. It tells you what should happen next
and what decisions still need to be made.

## 1. What Was Just Finished

The last completed workstream delivered:

- exportable binary sweeps
- packed Triton inference for `BinaryLinear`
- microkernel benchmarking
- trained-model end-to-end inference benchmarking
- comparison-script integration of model-level inference timing
- artifact-level summary and frontier extraction for sweep and trained-model
  benchmark outputs
- default binary shortcut and Triton ablation matrix automation in the
  trained-model benchmark
- output routing under `/mnt` for generated artifacts and checkpoints

That means the repo now has both quality-oriented and systems-oriented evidence,
not just architecture experiments.

## 2. Immediate Next Steps

These are the highest-priority next tasks if work resumes without changing the
overall direction.

Operational caveat:

- the latest stored outputs under `/mnt/binary_nn/artifacts/smoke/` are smoke
  runs, not the full decision-grade benchmark bundle
- use them to validate export and reporting paths only
- do not use them to update the binary versus dense quality story or the Triton
  speedup story

### 2.0 Refresh the decision-grade artifact bundle

- rerun the binary sweep on the normal grid so the frontier reflects real epoch
  budgets rather than smoke settings
- rerun the trained-model inference benchmark on the default batch-size matrix
  and keep the shortcut and Triton ablations enabled
- rerun the packed kernel benchmark on the large CUDA shapes that match the
  documented systems story
- write these outputs to the normal artifact location under `/mnt/binary_nn/artifacts/`
  and preserve `smoke/` as a separate validation-only namespace

### 2.1 Keep benchmark outputs decision-ready

- keep exporting JSON and CSV for all important sweeps and benchmarks
- preserve artifact-level summaries and frontier exports as the default path,
  not an optional extra step
- keep trained-model quality metrics and latency in the same records

### 2.2 Use the ablation matrix as the binary baseline gate

- treat shortcut on or off and Triton on or off as the standard binary systems
  matrix when evaluating major binary changes
- avoid mixing architecture gains with kernel gains in the same conclusion
- reject binary changes that do not beat the current baseline cleanly on that
  matrix

### 2.3 Expand the Triton path carefully

- keep training on the existing PyTorch path unless there is strong reason to
  change it
- extend the packed inference path only where it can be benchmarked clearly
- prefer end-to-end model wins over microkernel wins when choosing priorities
- treat `torch.set_float32_matmul_precision('high'|'medium')` as part of fair
  GPU benchmarking hygiene for wide dense-versus-binary comparisons
- prioritize the `(16384, 1024, 1024)` kernel regression specifically, because
  profiling now shows that this is a kernel-local loss, not just full-model
  overhead

### 2.4 Prepare the next representation baseline

- do not refactor the current binary path away
- add the smallest possible ternary or int2 research prototype beside the
  existing binary baseline
- make sure the prototype is benchmarkable with the same artifact and ablation
  machinery already used for the binary path

## 3. Open Design Decision

The main open decision now is how to approach the next kernel and model family.

There are three plausible paths.

### 3.1 Option A: Binary-first path

Keep pushing the current strict-binary route.

Pros:

- smallest disruption to the current codebase
- easiest to extend from the existing Triton implementation
- fastest path to deeper end-to-end speed experiments

Cons:

- may diverge from the practical BitNet-style deployment frontier
- may over-invest in a representation that is less attractive than ternary in
  the long run

### 3.2 Option B: Ternary or int2 pivot

Shift the systems path toward a BitNet-like packed ternary or int2
representation.

Pros:

- closer to what works publicly at scale for 1-bit or near-1-bit LLM systems
- stronger long-term research relevance for later LLM transfer

Cons:

- larger architecture and kernel change
- likely requires new model layers, packing code, and new benchmarking logic

### 3.3 Option C: Hybrid staged path

Keep the current binary path as the stable baseline and build a parallel ternary
prototype without replacing it yet.

Pros:

- preserves current progress
- gives a clean A/B comparison between binary and ternary systems paths
- reduces risk of losing a working baseline while exploring the more relevant
  longer-term direction

Cons:

- more code to maintain temporarily
- slightly slower than a full commitment to one direction

## 4. Recommended Direction

Current recommendation: **Option C, the hybrid staged path**.

Reasoning:

- the current binary route is now strong enough to serve as a stable systems and
  quality baseline
- the next meaningful research question is whether a ternary or int2-style path
  beats that binary baseline, not whether binary is worth keeping at all
- the hybrid path makes that question testable without sacrificing the working
  benchmark stack already built

## 5. Suggested First Milestone If Option C Is Chosen

If the hybrid path is chosen, the first milestone should be:

1. add a ternary linear prototype layer
2. add a packed ternary or int2 microbenchmark
3. benchmark it against the existing binary Triton path
4. only then decide whether the regression model itself should get a ternary
   training or inference variant

## 6. Recommended Execution Order

If work resumes now, the recommended order is:

1. optimize or retune the packed Triton kernel for the regressing
  `(16384, 1024, 1024)` operating point
2. keep wide dense-versus-binary comparisons on explicit `high` or `medium`
  matmul precision so the baseline story stays fair
3. add one minimal ternary or int2 layer prototype with matching benchmark
  hooks
4. compare binary versus ternary on kernel behavior first, and only then on
  regression quality

This sequence keeps the repo scientifically clean:

- first improve decision quality around the working binary baseline
- then open the next representation branch
- only then spend effort on broader architecture changes

## 7. Decision Note For Future Sessions

If a future session begins without user input on the open decision, default to:

- maintaining the current binary path
- improving documentation and benchmark hygiene
- avoiding a large ternary refactor until the decision is explicit
