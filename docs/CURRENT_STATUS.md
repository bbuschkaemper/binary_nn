# Current Status

Last updated: 2026-03-18

This document is the short operational memory for the repository. It should answer
three questions quickly:

1. What is implemented right now?
2. What has been validated already?
3. What should the next session assume as the starting point?

## Document Role

Read this file first when resuming work.

## 1. Current Objective

The repository goal has been reframed.

The main question is now:

- can we find a new low-bit training idea that improves GPU training and eventually
  GPU inference
- while preserving or improving quality against tuned dense BF16 baselines

This changes the emphasis:

- away from CPU-only sparse tricks as the primary objective
- toward training-rule design, operator design, and GPU measurement quality
- toward ideas that can help both training and inference, rather than one or the
  other in isolation

## 2. What Is Implemented Right Now

### 2.1 Stable baselines

- Dense regression baselines exist and are stable.
- Binary regression baselines exist and are stable.
- Shared training and evaluation still run through `src/regression_experiment.py`.
- Shared configuration now supports explicit Lightning precision, so BF16 dense
  baselines are reproducible through the same path as the low-bit runs.

### 2.2 Low-bit model families

- `BinaryLinear` / `BinaryRegressor`
  - sign-STE binary weights
  - packed Triton inference path on CUDA for systems benchmarking

- `TernaryLinear` / `TernaryRegressor`
  - STE ternary weights in `{-1, 0, +1}`
  - current quality-oriented ternary anchor

- `ShadowFreeTernaryLinear` / `ShadowFreeTernaryRegressor`
  - direct discrete ternary state updates
  - most novel local optimization idea in the repo
  - still strongest only on the easy benchmark

- hybrid / projected handoff path in `src/run_hybrid_ternary_regression.py`
  - STE warm start followed by projected or shadow-free-style handoff
  - current best bridge between ternary quality and sparse low-bit state

### 2.3 Benchmark surfaces

- original linear regression benchmark
- nonlinear residual benchmark
- wide nonlinear benchmark with `256` features
  - this is now the main local stress test for quality
- CPU inference timing exists in the ternary comparison flow
- GPU inference benchmarking exists only partially through the binary kernel path;
  ternary GPU benchmarking still needs dedicated work

## 3. Current Best Findings For The GPU-First Goal

### 3.1 Projected ternary is the current quality anchor

On the wide nonlinear benchmark, projected ternary is the strongest current
low-bit family.

Replicated projected `0.001` frontier across seeds `42`, `7`, and `123`:

- dense BF16 mean RMSE: `9.8788`
- projected ternary mean RMSE: `9.2311`
- dense BF16 mean total runtime: `7.9445s`
- projected ternary mean total runtime: `13.8249s`

Supporting seed-`42` ultra-low follow-ups:

- projected `0.005`: RMSE `9.2961`
- projected `0.0025`: RMSE `9.2808`

Interpretation:

- this is the best low-bit quality result in the repo
- it says the model-side quality problem is largely solved on the current
  regression stress test
- it does **not** yet provide a robust GPU speed win
- this family should be the accuracy anchor for any new GPU-first training idea

### 3.2 Shadow-free ternary is still the best proof that low-bit can win on
quality and inference speed in one place

On the original linear benchmark:

- dense BF16 RMSE: `14.8845`
- shadow-free ternary RMSE: `12.1707`
- dense BF16 total runtime: `6.9923s`
- shadow-free ternary total runtime: `7.9168s`
- final ternary nonzero density: `0.0172`

CPU inference speedup of shadow-free sparse path versus dense:

- batch `32`: `1.03x`
- batch `128`: `3.46x`
- batch `512`: `3.54x`

Interpretation:

- this is still the cleanest local proof of concept that a low-bit model can beat
  dense BF16 on both quality and inference speed
- however, it only holds on the easy benchmark and therefore should not be treated
  as the main GPU-first research direction by itself

### 3.3 STE ternary remains the simplest harder-task baseline

On the wide nonlinear benchmark:

- dense BF16 RMSE: `10.0171`
- STE ternary RMSE: `9.7710`
- dense BF16 total runtime: `11.8164s`
- STE ternary total runtime: `15.7970s`

Interpretation:

- STE still gives a clean harder-task quality reference
- it is slower than dense and too dense to be the final answer
- but it remains useful as a baseline for any projected or refresh-scheduled idea

### 3.4 What has been ruled out or deprioritized

- naive free-running shadow-free training on the nonlinear benchmark
- simple `row_block` structured projection near the projected frontier
- continuing to treat CPU sparsity alone as the main research target
- interpreting noisy shared-machine small-batch timing quirks as decision-grade
  evidence for speed claims

### 3.5 Working research synthesis

- projected / STE ternary should be the quality anchor
- shadow-free ternary should be mined for a better update rule, not treated as the
  final training recipe yet
- binary Triton work remains useful as operator and benchmarking groundwork
- the next genuinely new contribution should reduce GPU training cost by lowering
  operator overhead, memory traffic, or representation mismatch

## 4. Most Important Files

### 4.1 Models and training

- `src/regression_models.py`
- `src/regression_experiment.py`
- `src/run_ternary_regression.py`
- `src/run_shadowfree_ternary_regression.py`
- `src/run_hybrid_ternary_regression.py`
- `src/run_ternary_research_comparison.py`

### 4.2 Kernels and benchmarking

- `src/binary_kernels.py`
- `src/ternary_kernels.py`
- `src/benchmark_packed_binary_kernels.py`
- `src/benchmark_model_inference.py`
- `src/model_inference_benchmarking.py`

### 4.3 Core memory documents

- `docs/ROADMAP.md`
- `docs/IDEA.md`
- `docs/TERNARY_RESEARCH_EXPERIMENT_LOG.md`
- `docs/BINARY_REGRESSION_EXPERIMENT_LOG.md`

## 5. Current Best Artifacts

- shadow-free proof of concept:
  - `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`
- wide STE reference:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-followup.json`
- replicated projected frontier:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed7.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed123.json`
- supporting ultra-low seed-`42` follow-ups:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0005-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density00025-seed42.json`

## 6. Working Research Hypothesis

The most promising next idea is:

- **refresh-scheduled projected ternary training**

Working intuition:

- keep the projected / STE family as the quality anchor
- keep a cached discrete ternary state stable for multiple optimizer steps
- update or re-project that state only on refresh boundaries
- use that stable discrete state as the representation future fused GPU kernels and
  inference kernels would actually want

This is still a working hypothesis, not an implemented method.

## 7. What The Next Session Should Assume

- low-bit quality is no longer the main blocker on the current regression
  benchmarks
- GPU training speed has not been won yet
- GPU inference speed has not been won yet for ternary models
- projected / STE is the main family to extend
- shadow-free is mainly valuable as a source of update ideas
- new work should measure repeated step time, memory traffic, and end-to-end
  runtime before claiming any speed gain
