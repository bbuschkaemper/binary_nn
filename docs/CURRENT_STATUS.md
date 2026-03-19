# Current Status

Last updated: 2026-03-19

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

- `RefreshScheduledProjectedTernaryLinear` / `RefreshScheduledProjectedTernaryRegressor`
  - latent ternary weights with a cached projected ternary forward state refreshed
    every `K` steps
  - first implemented version of the new GPU-first training hypothesis

- `ControlledRefreshProjectedTernaryBlock` /
  `ControlledRefreshProjectedTernaryRegressor`
  - refresh-projected low-bit bulk path plus a small low-rank dense control path
  - ternary hidden-activation quantization between blocks
  - first concrete prototype of the broader "low-bit bulk + dense control + low-bit
    activations" architecture

### 2.3 Benchmark surfaces

- original linear regression benchmark
- nonlinear residual benchmark
- wide nonlinear benchmark with `256` features
  - this is now the main local stress test for quality
- CPU inference timing exists in the ternary comparison flow
- CUDA runs in the ternary comparison flow now also record repeated `fit`, `test`,
  and `predict` step benchmarks with mean step time, step-time variance, and peak
  GPU memory inside each run's `runtime.stage_benchmarks`
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

### 3.2 Refresh-projected remains the leading GPU-first prototype

Refresh-projected replication at density `0.001` with projection on every refresh:

- `K=4` mean RMSE across seeds `42`, `7`, and `123`: `9.2377`
- `K=8` mean RMSE across seeds `42`, `7`, and `123`: `9.2311`
- projected ternary mean RMSE across the same seeds: `9.2311`

Seed-`42` sweep highlights:

- `K=1`: RMSE `9.2845`
- `K=2`: RMSE `9.2761`
- `K=4`: RMSE `9.2930`
- `K=8`: RMSE `9.2780`
- projection every `2` refreshes: RMSE `9.4618`
- projection every `4` refreshes: RMSE `13.1989`

Interpretation:

- `refresh_projected` is now replicated, not just a single-seed prototype
- `K=8` with projection on every refresh matches the replicated projected quality
  frontier essentially exactly on the current three-seed set
- mean fit-step time improved from `3.3310ms` at `K=4` to `2.1305ms` at `K=8`,
  a `1.56x` speedup versus the earlier refresh default
- projection should still happen on every refresh; relaxing that cadence hurts quality
  sharply
- the fit-stage benchmark path was later corrected for refresh-style cyclic models:
  fit benchmarks now align to a refresh boundary and expand the timed window to cover
  at least two full refresh cycles
- under that corrected single-GPU benchmark, `K=16` became the best fit-oriented
  interval tested so far:
  - corrected `K=8` mean RMSE across seeds `42`, `7`, and `123`: `9.2316`
  - corrected `K=16` mean RMSE across the same seeds: `9.2299`
  - corrected `K=8` mean fit-step time: `3.0086ms`
  - corrected `K=16` mean fit-step time: `2.7552ms`
  - dense BF16 mean fit-step time across the corrected `K=16` runs: `2.0283ms`
- the corrected `K=16` sweep still did not beat dense on average, but it did produce a
  real seed-`7` fit-stage win versus dense (`1.726ms` versus `2.194ms`)
- a corrected seed-`42` `K=32` check regressed fit-stage cost to `3.984ms`, so pushing
  `K` upward blindly no longer looks attractive
- per-layer refresh intervals are now implemented in code, but the first naive
  layer-selective seed-`42` screens were negative:
  - `(16,8)`: RMSE `9.2740`, fit-step `3.648ms`
  - `(32,8)`: RMSE `9.2727`, fit-step `3.703ms`
  - both were slower than uniform `K=16` on the same corrected seed-`42` benchmark
- a validated layer-level operator profiler now exists in
  `src/benchmark_refresh_projected_training_ops.py`
- first same-GPU `K=16` layer measurements on the two wide hidden-layer shapes say:
  - shape `(128,256,256)`: dense `0.6724ms`, refresh non-refresh `0.7647ms`,
    refresh-step `1.3465ms`
  - shape `(128,256,128)`: dense `0.6631ms`, refresh non-refresh `0.8534ms`,
    refresh-step `1.4079ms`
- the refresh hook is the largest spike on refresh steps, but after amortizing by
  `K=16` the mean step gap is still more dominated by the always-on surrogate path:
  - shape `(128,256,256)`: mean gap versus dense `0.1287ms` =
    `0.0923ms` surrogate + `0.0364ms` amortized refresh hook
  - shape `(128,256,128)`: mean gap versus dense `0.2249ms` =
    `0.1903ms` surrogate + `0.0347ms` amortized refresh hook
- disabling density projection reduced refresh-step post cost by `0.3382ms` and
  `0.3267ms` on those same shapes, but that only corresponds to about `0.021ms`
  mean step cost at `K=16`
- a direct pruning microbenchmark showed the current exact `topk` density projection
  is already faster than a `kthvalue` threshold replacement by `1.36x` to `1.82x`
- the shared machine is still noisy enough that these operator-level numbers should
  be treated as directional decomposition, not as final end-to-end speed claims
- this is still the best-supported GPU-first training idea in the repo, and `K=16` is
  now the leading fit-oriented refresh interval, but the family still lacks a replicated
  mean speed win over dense BF16

### 3.3 Selective controlled refresh-projected is the best broader architecture
extension so far

First seed-`42` result on the same wide nonlinear benchmark, starting from the
`refresh_projected K=8, every1, density0.001` operating point:

- refresh-projected RMSE: `9.2780`
- controlled refresh-projected RMSE: `9.2728`
- refresh-projected fit / test / predict mean step time: `1.799ms` / `0.546ms` /
  `0.504ms`
- controlled refresh-projected fit / test / predict mean step time: `4.306ms` /
  `1.574ms` / `1.068ms`
- parameter count change versus refresh-projected: `+14,336`

Interpretation:

- the broader architectural direction is scientifically plausible: a tiny dense control
  path plus ternary hidden activations can slightly improve quality on the hard task
- however, the current low-rank control path is too expensive and clearly moves runtime
  in the wrong direction
- this means `refresh_projected` remains the speed-oriented baseline, while the new
  control-path family should now be optimized for cheaper control capacity

Follow-up sweep findings:

- disabling low-bit activations in the control branch materially hurt quality
  (`9.5991` RMSE at rank `2`)
- the best quality/speed tradeoff in the control sweep was low-rank control only on the
  first hidden block, `control_ranks=(4, 0)`
- clean single-GPU seed-`42` comparison for that selective variant:
  - refresh-projected RMSE: `9.2778`
  - selective controlled refresh-projected RMSE: `9.2744`
  - refresh-projected fit / test / predict mean step time: `2.012ms` / `0.558ms` /
    `0.505ms`
  - selective controlled refresh-projected fit / test / predict mean step time:
    `3.013ms` / `1.564ms` / `1.871ms`
- scalar-gated control screens did not beat the selective low-rank variant

Interpretation:

- low-bit activations appear to be part of the benefit, not just optional decoration
- selective control is clearly better than full control, but it still slows the model
- if the control path is extended further, it should start from first-block-only
  low-rank control or a fused/grouped variant, not from larger dense adapters

### 3.4 Shadow-free ternary is still the best proof that low-bit can win on
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

### 3.5 STE ternary remains the simplest harder-task baseline

On the wide nonlinear benchmark:

- dense BF16 RMSE: `10.0171`
- STE ternary RMSE: `9.7710`
- dense BF16 total runtime: `11.8164s`
- STE ternary total runtime: `15.7970s`

Interpretation:

- STE still gives a clean harder-task quality reference
- it is slower than dense and too dense to be the final answer
- but it remains useful as a baseline for any projected or refresh-scheduled idea

### 3.6 What has been ruled out or deprioritized

- naive free-running shadow-free training on the nonlinear benchmark
- simple `row_block` structured projection near the projected frontier
- continuing to treat CPU sparsity alone as the main research target
- interpreting noisy shared-machine small-batch timing quirks as decision-grade
  evidence for speed claims

### 3.7 Working research synthesis

- projected / STE ternary should be the quality anchor
- shadow-free ternary should be mined for a better update rule, not treated as the
  final training recipe yet
- binary Triton work remains useful as operator and benchmarking groundwork
- the next genuinely new contribution should reduce GPU training cost by lowering
  operator overhead, memory traffic, or representation mismatch
- at the current `K=16` operating point, the first operator target should be the
  always-on surrogate forward/backward path rather than the once-per-refresh hook
- keep the current `topk` density projection until a measured alternative actually wins
- the first control-path prototype says the architecture direction is viable, but the
  dense correction must become much lighter before it can compete on speed

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
- `src/benchmark_refresh_projected_training_ops.py`
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
- GPU-stage wide baseline reruns:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-gpu-stage-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-gpu-stage-density0001-seed42.json`
- replicated refresh-projected frontier:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123.json`
- single-GPU refresh replications:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42-singlegpu.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7-singlegpu.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123-singlegpu.json`
- corrected cycle-aligned refresh replications:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed42-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed7-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed123-singlegpu-cyclebench.json`
- layer-level operator profiling:
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-training-ops-k16.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-training-ops-k16-summary.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-projection-isolation.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-projection-pruner-microbench.json`
- initial layer-selective refresh screens:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-r168-density0001-seed42-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-r328-density0001-seed42-singlegpu-cyclebench.json`
- first controlled refresh-projected architecture check:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-controlled-refresh-projected-k8-r16-a025-density0001-seed42.json`
- best selective controlled-refresh candidate so far:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-controlled-refresh-projected-k8-r40-a025-density0001-seed42-singlegpu.json`
- supporting ultra-low seed-`42` follow-ups:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0005-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density00025-seed42.json`

## 6. Working Research Hypothesis

The most promising next idea family is now:

- **refresh-scheduled projected ternary training**, extended with a much cheaper dense
  control path and low-bit hidden activations when those additions honestly help

Working intuition:

- keep the projected / STE family as the quality anchor
- keep a cached discrete ternary state stable for multiple optimizer steps
- update or re-project that state only on refresh boundaries
- use that stable discrete state as the representation future fused GPU kernels and
  inference kernels would actually want

This is now an implemented and replicated local method, but it is still early-stage
research rather than a final answer.

## 7. What The Next Session Should Assume

- low-bit quality is no longer the main blocker on the current regression
  benchmarks
- GPU training speed has not been won yet
- GPU inference speed has not been won yet for ternary models
- projected / STE is the main family to extend
- `refresh_projected` remains the main training-rule family to extend
- `refresh_projected` with density `0.001` is still the leading low-bit baseline, and
  corrected cycle-aligned GPU benchmarking currently favors `K=16` over `K=8` for the
  fit-stage tradeoff
- the corrected `K=16` sweep preserved the quality edge and improved mean fit time over
  corrected `K=8`, but it still did not beat dense on average across the three seeds
- a validated layer-level profiler now exists for the refresh path, and its first
  same-GPU decomposition says the mean gap at `K=16` is more dominated by the
  always-on surrogate path than by the amortized refresh hook
- density projection is still a real refresh-step spike, but at `K=16` its isolated
  mean contribution is only about `0.02ms` per hidden layer on the representative
  wide-benchmark shapes
- the current exact `topk` projection rule should stay in place until a faster
  measured replacement exists
- per-layer refresh intervals are now available, but naive two-layer schedules
  `(16,8)` and `(32,8)` did not beat uniform `K=16` on seed `42`
- `controlled_refresh_projected` is now implemented; the best current variant is
  first-block-only low-rank control, which slightly improves seed-`42` quality but is
  still slower than `refresh_projected`
- low-bit activations mattered inside the control path; removing them hurt quality
- shadow-free is mainly valuable as a source of update ideas
- the repo now has a repeated GPU stage-benchmark path in
  `src/run_ternary_research_comparison.py`; use that path before making any speed
  claim
- new work should still compare repeated step time, memory traffic, and end-to-end
  runtime before claiming any speed gain
