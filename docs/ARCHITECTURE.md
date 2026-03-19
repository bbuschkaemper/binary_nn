# Architecture

Last updated: 2026-03-18

This document maps the current `src/` layout and explains how the main pieces fit
together under the new GPU-first research goal.

## Document Role

Use this file when you need to change code, wire new experiments, or decide where a
new GPU-oriented training idea should live.

## 1. High-Level Structure

The codebase has six main layers:

1. data generation and loading
2. shared training and evaluation infrastructure
3. model definitions
4. experiment entry points
5. benchmarking and artifact export
6. low-bit kernel code

The repo is still small enough to trace easily, but the ternary branch means the
model layer is no longer binary-only.

## 2. Source Tree Map

### 2.1 Data and training core

- `src/regression_data.py`
  - synthetic regression dataset generation
  - supports both `target_kind="linear"` and `target_kind="nonlinear_residual"`
  - current benchmark surface definitions

- `src/regression_experiment.py`
  - shared Lightning module and trainer construction
  - shared metric computation
  - common fit / test / predict runtime capture
  - the most natural place to add GPU timing and memory instrumentation

### 2.2 Models

- `src/regression_models.py`
  - dense baselines
  - binary `BinaryLinear` / `BinaryRegressor`
  - STE ternary `TernaryLinear` / `TernaryRegressor`
  - shadow-free ternary `ShadowFreeTernaryLinear` / `ShadowFreeTernaryRegressor`
  - projected handoff helpers and density projection logic
  - the most important file for changing the training rule

- `src/binary_kernels.py`
  - packed binary reference path
  - Triton packed inference kernel
  - current example of low-bit GPU operator work in the repo

- `src/ternary_kernels.py`
  - ternary helper ops used by the current ternary inference paths
  - current place to extend if ternary operator work moves toward GPU kernels

### 2.3 Experiment entry points

- `src/run_regression_baseline.py`
  - dense BF16 baseline

- `src/run_binary_regression.py`
  - binary baseline training

- `src/run_ternary_regression.py`
  - STE ternary training

- `src/run_shadowfree_ternary_regression.py`
  - direct-discrete shadow-free training

- `src/run_hybrid_ternary_regression.py`
  - hybrid and projected ternary handoff flow
  - current best place to implement a refresh-scheduled projected variant

- `src/run_regression_comparison.py`
  - dense vs binary comparison
  - runtime and inference artifact generation for the binary branch

- `src/run_ternary_research_comparison.py`
  - dense vs one ternary family comparison
  - current artifact path for ternary quality and runtime evidence
  - natural place to add GPU timing and memory stats

### 2.4 Benchmarking and artifact code

- `src/benchmark_model_inference.py`
  - trained-model end-to-end inference benchmarking for the binary stack

- `src/benchmark_packed_binary_kernels.py`
  - microbenchmark for the packed Triton binary kernel

- `src/model_inference_benchmarking.py`
  - shared inference benchmarking utilities for binary workflows

- `src/output_paths.py`
  - routes artifacts under `/mnt`
  - keeps experiment output out of the repository tree

## 3. Main Execution Flows

### 3.1 Dense baseline flow

`run_regression_baseline.py`
-> `train_regression_baseline`
-> `train_regression_model`
-> `RegressionDataModule`
-> `RegressionLightningModule`
-> `DenseRegressor`

This is the quality and runtime reference path.

### 3.2 Binary baseline and kernel flow

`run_binary_regression.py`
-> shared training path
-> `BinaryRegressor`
-> optional comparison through `run_regression_comparison.py`
-> optional operator benchmarking through `benchmark_packed_binary_kernels.py`

This path matters mainly as a stable baseline and GPU kernel sandbox.

### 3.3 STE and projected ternary flow

`run_ternary_regression.py`
-> shared training path
-> `TernaryRegressor`

`run_hybrid_ternary_regression.py`
-> STE warm start
-> projected handoff into a sparse ternary state
-> optional recovery training

This is the most important path for the next GPU-first training idea.

### 3.4 Shadow-free ternary flow

`run_shadowfree_ternary_regression.py`
-> shared training path
-> `ShadowFreeTernaryRegressor`
-> direct discrete updates from accumulated evidence

This path is the main source of update-rule novelty, but not the current quality
anchor.

### 3.5 Comparison and artifact flow

`run_ternary_research_comparison.py`
-> dense baseline
-> selected ternary family (`shadowfree`, `ste`, `hybrid`, `projected`,
   `refresh_projected`, or `controlled_refresh_projected`)
-> runtime capture
-> CPU inference benchmarking
-> JSON + CSV artifacts under `/mnt`

This is the current decision-grade evidence path. It should be extended with GPU
timing rather than replaced.

## 4. GPU-First Extension Points

### 4.1 Add GPU timing and memory metrics

Best starting files:

- `src/regression_experiment.py`
- `src/run_ternary_research_comparison.py`

Add:

- repeated timing summaries
- step-time statistics
- peak memory
- cleaner separation of fit, test, and inference timing

### 4.2 Change the training rule

Best starting files:

- `src/regression_models.py`
- `src/run_hybrid_ternary_regression.py`
- `src/run_ternary_regression.py`

This is where a refresh-scheduled projected variant should land.

### 4.3 Add activation-side low-bit experiments

Best starting files:

- `src/regression_models.py`
- `src/regression_experiment.py`
- `src/run_hybrid_ternary_regression.py`

Keep activation experiments easy to toggle so they can be benchmarked cleanly against
the current projected anchor. The current controlled-refresh branch is now the main
place where low-bit activation experiments and tiny dense control paths meet.

### 4.4 Add GPU inference operator benchmarking

Best starting files:

- `src/binary_kernels.py`
- `src/ternary_kernels.py`
- `src/benchmark_packed_binary_kernels.py`
- `src/benchmark_model_inference.py`

The binary Triton path is the current example to emulate or generalize.

## 5. Current Design Boundaries

- dense BF16 remains the fairness anchor
- projected / STE is the quality anchor
- shadow-free is the direct-discrete idea source
- the wide nonlinear benchmark is the main local quality test
- CPU timing remains useful, but GPU measurement should drive top-level decisions
- kernel work should follow the representation that the training path actually wants

## 6. Most Likely Next Architectural Extension

The next architectural extension should now be a *lighter* version of the current
controlled-refresh family, not a brand-new family.

That means:

- keep the refresh-scheduled projected low-bit bulk path
- shrink or sparsify the dense control path
- keep low-bit hidden activations only if they survive the cheaper-control ablation
- continue judging variants with repeated GPU stage metrics rather than one-off totals

The first controlled-refresh prototype already proved the architectural idea can help
quality slightly. The next job is to make that benefit cheap enough to matter.

## 7. Practical Rule Of Thumb

- if a change affects low-bit semantics, start in `src/regression_models.py`
- if a change affects training fairness or timing evidence, start in
  `src/regression_experiment.py` or `src/run_ternary_research_comparison.py`
- if a change affects operator shape or packing, start in `src/binary_kernels.py` or
  `src/ternary_kernels.py`
