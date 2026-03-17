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

That means the repo now has both quality-oriented and systems-oriented evidence,
not just architecture experiments.

## 2. Immediate Next Steps

These are the highest-priority next tasks if work resumes without changing the
overall direction.

### 2.1 Keep improving the benchmark pipeline

- keep exporting JSON and CSV for all important sweeps and benchmarks
- add a small summary table or frontier extraction from exported artifacts
- keep trained-model quality metrics and latency in the same records

### 2.2 Strengthen shortcut ablations

- compare binary-with-shortcut vs binary-without-shortcut more systematically
- avoid mixing architecture gains with kernel gains in the same conclusion
- use the shortcut toggle in sweeps and comparison runs whenever a major model
  change is tested

### 2.3 Expand the Triton path carefully

- keep training on the existing PyTorch path unless there is strong reason to
  change it
- extend the packed inference path only where it can be benchmarked clearly
- prefer end-to-end model wins over microkernel wins when choosing priorities

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

## 6. Decision Note For Future Sessions

If a future session begins without user input on the open decision, default to:

- maintaining the current binary path
- improving documentation and benchmark hygiene
- avoiding a large ternary refactor until the decision is explicit
