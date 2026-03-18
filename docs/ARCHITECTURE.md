# Architecture

Last updated: 2026-03-18

This document maps the current `src/` layout and explains how the main pieces
fit together. It is meant to make future code changes faster and less error
prone.

## Document Role

Use this file when you need to understand how the implementation is wired, not
just what the current research status is.

## 1. High-Level Structure

The codebase now has six main layers:

1. data generation and loading
2. shared training and evaluation infrastructure
3. model definitions
4. experiment entry points
5. systems benchmarking and kernel code
6. ternary research comparison tooling

The repository is still small enough that these layers remain easy to trace, but
the ternary branch means the model layer is no longer binary-only.

## 2. Source Tree Map

### 2.1 Data and training core

- `src/regression_data.py`
  - synthetic regression dataset generation
  - supports both:
    - `target_kind="linear"`
    - `target_kind="nonlinear_residual"`
  - train, validation, and test splits
  - scaling and dataloader construction

- `src/regression_experiment.py`
  - shared Lightning module
  - shared trainer construction
  - shared metric computation
  - common training loop used by dense, binary, and ternary experiments
  - explicit Lightning precision support through `TrainingConfig.precision`
  - extra discrete-parameter counting hook for models with non-Parameter state

These two files are still the central backbone of the regression proof of
concept.

### 2.2 Models and kernels

- `src/regression_models.py`
  - dense MLP baseline
  - binary sign STE
  - `BinaryLinear`
  - `BinaryRegressor`
  - `TernaryLinear`
  - `TernaryRegressor`
  - `ShadowFreeTernaryLinear`
  - `ShadowFreeTernaryRegressor`
  - cached sparse CPU inference for both ternary layer families

- `src/binary_kernels.py`
  - binary weight packing and unpacking
  - packed reference path
  - Triton packed inference kernel

- `src/model_inference_benchmarking.py`
  - shared trained-model inference timing utilities for dense and binary models
  - common record schema for latency plus quality metrics
  - artifact export helpers for model-level benchmark data

- `src/output_paths.py`
  - centralizes output routing under `/mnt`
  - keeps artifacts and checkpoints out of the repository tree

### 2.3 Experiment entry points

- `src/run_regression_baseline.py`
  - trains the dense baseline

- `src/run_binary_regression.py`
  - trains the binary model
  - supports enabling or disabling the dense residual shortcut

- `src/run_shadowfree_ternary_regression.py`
  - trains the direct-discrete shadow-free ternary branch

- `src/run_ternary_regression.py`
  - trains the STE ternary branch

- `src/run_hybrid_ternary_regression.py`
  - trains the STE-to-shadow-free handoff path
  - supports an optional target-density projection before recovery training

- `src/run_regression_comparison.py`
  - trains dense and binary back to back
  - prints task-quality deltas
  - prints runtime deltas
  - prints trained-model inference benchmark records by default
  - writes a comparison bundle plus inference benchmark artifacts under `/mnt`

- `src/run_ternary_research_comparison.py`
  - trains a dense baseline plus one ternary branch
  - benchmarks CPU inference with sparse on/off variants
  - writes a comparison JSON plus CPU benchmark CSV under `/mnt`
  - supports `shadowfree`, `ste`, `hybrid`, and `projected` ternary families

- `src/run_binary_regression_sweep.py`
  - sweeps binary architecture and optimization settings
  - prints Pareto candidates
  - exports JSON and CSV sweep results

### 2.4 Systems benchmarks

- `src/benchmark_packed_binary_kernels.py`
  - microbenchmark for the packed Triton binary linear kernel
  - exports JSON, CSV, summary, and frontier artifacts under `/mnt`

- `src/benchmark_model_inference.py`
  - trained-model benchmark for end-to-end dense and binary inference
  - exports latency plus task-quality metrics together

Important boundary:

- the new ternary comparison flow has its own benchmark logic in
  `src/run_ternary_research_comparison.py`
- it is not yet folded into `src/model_inference_benchmarking.py`

## 3. Main Execution Flows

### 3.1 Dense training flow

`run_regression_baseline.py`
-> `train_regression_baseline`
-> `train_regression_model`
-> `RegressionDataModule`
-> `RegressionLightningModule`
-> `DenseRegressor`

### 3.2 Binary training flow

`run_binary_regression.py`
-> `train_binary_regression`
-> `train_regression_model`
-> `RegressionDataModule`
-> `RegressionLightningModule`
-> `BinaryRegressor`
-> `BinaryLinear`

Important detail:

- training uses the normal PyTorch `BinaryLinear` path
- inference may switch to the packed Triton path inside `BinaryLinear`
  depending on mode and device

### 3.3 Shadow-free ternary flow

`run_shadowfree_ternary_regression.py`
-> `train_shadowfree_ternary_regression`
-> `train_regression_model`
-> `RegressionDataModule`
-> `RegressionLightningModule`
-> `ShadowFreeTernaryRegressor`
-> `ShadowFreeTernaryLinear`

Important detail:

- the ternary state is stored directly in `weight_state`
- evidence is accumulated during backward
- direct discrete updates happen from `RegressionLightningModule.optimizer_step`
- CPU inference can switch to cached sparse execution inside
  `ShadowFreeTernaryLinear`

### 3.4 STE ternary flow

`run_ternary_regression.py`
-> `train_ternary_regression`
-> `train_regression_model`
-> `RegressionDataModule`
-> `RegressionLightningModule`
-> `TernaryRegressor`
-> `TernaryLinear`

Important detail:

- training uses STE-style ternary quantization of a latent weight
- eval-time CPU inference can switch to cached sparse execution inside
  `TernaryLinear`

### 3.5 Ternary comparison flow

`run_ternary_research_comparison.py`
-> train dense reference
-> train chosen ternary branch
-> clone both models to CPU
-> benchmark dense
-> benchmark ternary with sparse off
-> benchmark ternary with sparse on
-> write comparison JSON plus CPU benchmark CSV

## 4. Current Design Boundaries

The current implementation intentionally keeps several boundaries sharp.

### 4.1 Binary versus ternary

The binary branch remains the stable low-bit baseline.
The ternary branch is still research-oriented.

### 4.2 Shadow-free versus STE ternary

These are treated as separate ideas, not as one interchangeable model:

- `ShadowFreeTernaryLinear`
  - direct-discrete update rule
  - stronger research novelty
  - currently validated only on the easy linear benchmark

- `TernaryLinear`
  - latent-weight STE training
  - better harder-task quality baseline
  - currently too dense for CPU sparse wins

`run_hybrid_ternary_regression.py` composes the two families into handoff
experiments, including a target-density projection variant.

### 4.3 Training versus inference

Training and inference are still deliberately separated.

- BF16 and optimizer behavior live in the shared training path
- sparse CPU inference is eval-only
- Triton binary kernels are inference-only

### 4.4 Easy benchmark versus harder benchmark

The original linear regression task is still the easiest proof-of-concept
environment.

The new nonlinear residual benchmark is the better ternary quality gate.

Do not treat a ternary result on the easy linear task as sufficient evidence for
the harder nonlinear case.

## 5. Where To Change What

If a future session needs to modify a particular concern, use this map.

### 5.1 Change model architecture

Edit:

- `src/regression_models.py`

Typical examples:

- change shortcut structure
- change hidden activations
- modify ternary quantization thresholds
- modify the shadow-free discrete update rule

### 5.2 Change training behavior

Edit:

- `src/regression_experiment.py`
- `src/run_shadowfree_ternary_regression.py`
- `src/run_ternary_regression.py`
- `src/run_regression_comparison.py`

Typical examples:

- change optimizer settings
- change trainer precision
- change when direct discrete updates happen
- add new CLI toggles

### 5.3 Change benchmark data

Edit:

- `src/regression_data.py`
- `src/run_ternary_research_comparison.py`

Typical examples:

- add a new target type
- tune nonlinear benchmark difficulty
- change feature counts or noise defaults

### 5.4 Change inference behavior

Edit:

- `src/binary_kernels.py`
- `src/regression_models.py`
- `src/run_ternary_research_comparison.py`

Typical examples:

- extend Triton coverage
- change sparse-cache policy
- add a packed ternary kernel
- change CPU latency benchmark settings

## 6. Important Current Assumptions

These assumptions are embedded in the current design.

- The binary residual regressor is still the right stable baseline.
- The dense shortcut is still useful enough to keep.
- The shadow-free ternary route should be read as a sparse residual path, not a
  fully standalone ternary MLP win.
- The nonlinear residual benchmark is the main ternary sanity check for future
  work.
- The projected handoff is currently the best sparse-friendly nonlinear bridge.
- Explicit precision settings matter for fair GPU comparisons.

## 7. Most Likely Next Architectural Extension

If work continues along the currently recommended direction, the next major
structural addition is likely to be one of:

- a structured block-sparse ternary layer
- a lower-density STE-to-shadow-free handoff path
- a packed ternary CPU kernel

Those are more likely next steps than another large refactor of the existing
binary Triton path.
