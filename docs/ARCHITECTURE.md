# Architecture

Last updated: 2026-03-17

This document maps the current `src/` layout and explains how the main pieces
fit together. It is meant to make future code changes faster and less error
prone.

## Document Role

Use this file when you need to understand how the implementation is wired, not
just what the current research status is.

## 1. High-Level Structure

The codebase currently has five main layers:

1. data generation and loading
2. shared training and evaluation infrastructure
3. model definitions
4. experiment entry points
5. systems benchmarking and kernel code

The repository is still small enough that these layers are mostly one file each
or a small cluster of files.

## 2. Source Tree Map

### 2.1 Data and training core

- `src/regression_data.py`
  - synthetic regression dataset generation
  - train, validation, and test splits
  - scaling and dataloader construction

- `src/regression_experiment.py`
  - shared Lightning module
  - shared trainer construction
  - shared metric computation
  - common training loop used by dense and binary experiments

These two files are the central backbone of the regression proof of concept.

### 2.2 Models and kernels

- `src/regression_models.py`
  - dense MLP baseline
  - binary sign STE
  - `BinaryLinear`
  - residual binary regressor

- `src/binary_kernels.py`
  - binary weight packing and unpacking
  - packed reference path
  - Triton packed inference kernel

- `src/model_inference_benchmarking.py`
  - shared trained-model inference timing utilities
  - common record schema for latency plus quality metrics
  - artifact export helpers for model-level benchmark data

These files define the current architecture and the systems path.

### 2.3 Experiment entry points

- `src/run_regression_baseline.py`
  - trains the dense baseline

- `src/run_binary_regression.py`
  - trains the binary model
  - supports enabling or disabling the dense residual shortcut

- `src/run_regression_comparison.py`
  - trains dense and binary back to back
  - prints task-quality deltas
  - prints runtime deltas
  - prints trained-model inference benchmark records by default

- `src/run_binary_regression_sweep.py`
  - sweeps binary architecture and optimization settings
  - prints Pareto candidates
  - exports JSON and CSV sweep results

These files are the normal top-level workflow surface.

### 2.4 Systems benchmarks

- `src/benchmark_packed_binary_kernels.py`
  - microbenchmark for the packed Triton binary linear kernel

- `src/benchmark_model_inference.py`
  - trained-model benchmark for end-to-end dense and binary inference
  - exports latency plus task-quality metrics together

These files answer the systems question directly.

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

### 3.3 Comparison flow

`run_regression_comparison.py`
-> train dense result
-> train binary result
-> compare task metrics
-> compare runtimes
-> benchmark trained-model inference using
   `model_inference_benchmarking.py`

### 3.4 Sweep flow

`run_binary_regression_sweep.py`
-> dense reference once
-> binary runs across hidden dims, learning rates, epochs
-> Pareto filtering
-> optional JSON and CSV export

### 3.5 Kernel benchmark flow

`benchmark_packed_binary_kernels.py`
-> generate synthetic inputs and weights
-> pack binary weights
-> benchmark unpacked reference path
-> benchmark Triton path
-> compare latency and numerical error

### 3.6 Trained-model inference benchmark flow

`benchmark_model_inference.py`
-> train dense and binary models
-> clone trained models for eval benchmarking
-> benchmark model-level latency over selected batch sizes
-> export latency plus quality metrics

## 4. Current Design Boundaries

The current implementation intentionally keeps several boundaries sharp.

### 4.1 Training vs inference

Training and inference are deliberately separated.

- training is stable PyTorch-first
- packed Triton kernels are inference-only for now

This was done to avoid conflating optimization experiments with kernel
experiments.

### 4.2 Architecture vs systems effects

The dense residual shortcut is treated as an architecture feature.
The Triton packed path is treated as a systems feature.

The benchmark and comparison tooling are increasingly structured to keep those
effects separable.

### 4.3 Regression prototype vs long-range goal

The regression task is a proof-of-concept environment, not the final target.
The longer-range research direction still points toward low-bit FFN methods for
later LLM-relevant work.

## 5. Where To Change What

If a future session needs to modify a particular concern, use this map.

### 5.1 Change model architecture

Edit:

- `src/regression_models.py`

Typical examples:

- add or remove shortcut paths
- change hidden path nonlinearities
- introduce a ternary layer variant

### 5.2 Change training behavior

Edit:

- `src/regression_experiment.py`
- `src/run_binary_regression.py`
- `src/run_regression_comparison.py`

Typical examples:

- change optimizer settings
- change trainer config
- add new CLI toggles

### 5.3 Change kernel behavior

Edit:

- `src/binary_kernels.py`
- `src/regression_models.py`

Typical examples:

- add new packed representations
- extend Triton kernel coverage
- change cache policy for packed weights

### 5.4 Change experiment reporting

Edit:

- `src/run_binary_regression_sweep.py`
- `src/run_regression_comparison.py`
- `src/model_inference_benchmarking.py`
- `src/benchmark_model_inference.py`

Typical examples:

- add export fields
- add new benchmark records
- add frontier extraction logic

## 6. Important Current Assumptions

These assumptions are embedded in the current design.

- The binary residual regressor is the right current baseline.
- The dense shortcut is currently useful enough to keep.
- The packed Triton path is worth extending.
- Quality metrics and latency metrics should be tracked together whenever
  possible.

## 7. Most Likely Next Architectural Extension

If work continues along the currently recommended direction, the next major
structural addition is likely to be a parallel ternary or int2 path rather than
another large refactor of the current binary path.

If that happens, the likely new files or modules will be:

- a ternary layer implementation alongside `BinaryLinear`
- a ternary packing or int2 packing module
- a ternary kernel benchmark parallel to the current binary kernel benchmark
- new comparison surfaces that benchmark binary vs ternary systems paths
