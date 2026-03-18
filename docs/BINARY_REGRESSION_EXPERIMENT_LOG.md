# Binary Regression Experiment Log

Last updated: 2026-03-18

This document preserves the binary branch history, but now through the lens of the
repository's new GPU-first goal.

## Document Role

Use this file when you need the stable binary baseline story, the binary systems
groundwork, or the main lessons that still transfer into the newer ternary work.

## 1. Why The Binary Branch Still Matters

The binary workstream is no longer the main frontier, but it still matters for three
reasons.

1. It is the most stable low-bit baseline in the repo.
2. It contains the best current GPU operator groundwork through the Triton packed
   inference path.
3. It taught several design lessons that still transfer to ternary research:
   - residual shortcuts matter
   - fair comparison harnesses matter
   - kernel-aware representations matter

## 2. Implemented Binary Path

Main components:

- `BinaryLinear`
- `BinaryRegressor`
- `src/run_binary_regression.py`
- `src/run_regression_comparison.py`
- `src/run_binary_regression_sweep.py`
- `src/benchmark_model_inference.py`
- `src/benchmark_packed_binary_kernels.py`
- `src/binary_kernels.py`

Current design:

- sign-STE binary weights with per-output scaling
- optional dense shortcut for residual correction
- shared training path with dense baselines
- eval-only Triton packed inference path on CUDA

## 3. Main Experiment Sequence

### 3.1 Shared training and comparison harness

The binary and dense branches were first moved onto the same Lightning training and
evaluation path.

This mattered because it gave the repo:

- consistent metrics
- consistent runtime capture
- fairer dense-vs-binary comparisons
- a cleaner place to add future low-bit ideas

### 3.2 Residual binary architecture changed the frontier

The first big improvement came from adding a dense linear shortcut beside the binary
stack.

Local lesson:

- on these regression tasks, low-bit branches should not be forced to relearn the full
  easy linear backbone if a cheap residual structure is available

This lesson still matters for ternary work too.

### 3.3 Sweeps produced two useful operating points

The binary sweep work produced two practical reference points on the original linear
benchmark.

Quality-oriented binary point:

- hidden dims `(8,)`
- learning rate `3e-3`
- epochs `75`
- RMSE `12.4447`
- total runtime `8.5400s`

Interpretation:

- slightly better quality than the dense baseline on the easy task
- close enough on runtime to remain a useful low-bit baseline

Speed-oriented binary point:

- hidden dims `(8,)`
- learning rate `3e-3`
- epochs `40`
- RMSE `15.1069`
- total runtime `3.5248s`

Interpretation:

- materially faster than the dense baseline on the easy task
- stays close enough on quality to remain an informative speed-oriented low-bit point

### 3.4 Triton packed inference path added systems groundwork

The packed Triton binary path is still the main GPU-kernel proof of concept in the repo.

What it established:

- operator-specific low-bit kernels can outperform generic unpacked paths
- the representation has to be designed around the operator
- kernel work is worth doing, but only if it targets a representation that matters for
  the broader research story

### 3.5 Known caution

The binary kernel path still has a known large-batch regression and is not a universal
end-to-end win.

That is important because it reinforces a broader lesson for the whole repository:

- low-bit systems work is easy to overclaim if the benchmarking method is not careful

## 4. What The Binary Branch Says For The New GPU-First Goal

The binary workstream now serves mainly as:

- a stable low-bit baseline
- a GPU operator sandbox
- evidence that architecture and benchmarking decisions can matter as much as the bit
  representation itself

It is **not** currently the strongest path for the next research leap, because the
harder-task ternary path now gives much stronger evidence on model quality.

## 5. Current Role Of Binary Work

Keep the binary branch healthy, but treat it as secondary to the projected ternary path.

Good reasons to touch the binary branch now:

- keeping the Triton operator path working
- borrowing operator design ideas for future ternary GPU kernels
- preserving a stable low-bit baseline for comparison

Weak reasons to make it the main focus now:

- searching for the next training idea on the harder benchmark
- claiming the main low-bit quality frontier lives here

## 6. Recommended Binary Next Steps

- keep the packed Triton path benchmarkable and correct
- use the binary operator work as a design reference for ternary GPU kernels
- avoid spending the main research budget here until the projected GPU-first ternary
  idea plateaus
