# Current Status

Last updated: 2026-03-17

This document is the short operational memory for the repository. It is meant
to answer three questions quickly:

1. What is implemented right now?
2. What has been validated already?
3. What should the next session assume as the starting point?

## Document Role

Use this file as the first read when resuming work. It is the shortest path to
the current implementation state.

## 1. Current Technical State

The repository currently has four working layers of functionality.

### 1.1 Regression training baseline

- Dense regression baseline exists and is stable.
- Binary regression baseline exists and is stable.
- Both run through the shared Lightning training utilities in
  `src/regression_experiment.py`.

### 1.2 Binary model architecture

- The current best binary architecture is a residual binary regressor.
- The binary path uses `BinaryLinear` layers plus `Hardtanh`.
- A dense input shortcut is enabled by default and is currently important for
  quality.
- Current default binary configuration is:
  - hidden dims: `(8,)`
  - learning rate: `3e-3`

### 1.3 Systems path

- `BinaryLinear` has an eval-only packed Triton inference path.
- Training still uses the normal PyTorch floating-point path so optimization is
  unchanged.
- Packed inference is currently implemented for binary sign weights plus a
  per-output scale.

### 1.4 Benchmarking and experiment tooling

- Binary sweep script supports JSON and CSV export.
- Binary sweep script now also emits summary and frontier artifacts.
- Kernel microbenchmark exists for packed Triton inference and now emits JSON,
  CSV, summary, and frontier artifacts.
- Model-level inference benchmark exists for trained dense and binary models.
- Model-level inference benchmark now automates the shortcut and Triton ablation
  matrix by default.
- Model-level inference benchmark now also supports configurable regression
  feature count, so larger-width trained-model runs can be benchmarked without
  editing code.
- Model-level inference benchmark now also supports configurable float32
  matmul precision, which matters on Tensor Core GPUs for fair dense versus
  binary systems comparisons.
- The dense-vs-binary comparison script now includes a model-level inference
  benchmark section by default and now emits a comparison artifact bundle.
- Generated artifacts and Lightning checkpoints are now routed under `/mnt`, not
  back into the repository tree.
- The latest checked-in bundle under `/mnt/binary_nn/artifacts/smoke/` should be
  treated as smoke validation only. It is useful for verifying export paths and
  benchmark plumbing, but it is not the decision-grade baseline for binary
  quality or Triton speedups.

## 2. Most Important Files

The files below are the main entry points to understand or continue the work.

### 2.1 Models and kernels

- `src/regression_models.py`
- `src/binary_kernels.py`
- `src/model_inference_benchmarking.py`

### 2.2 Training and comparison

- `src/run_regression_baseline.py`
- `src/run_binary_regression.py`
- `src/run_regression_comparison.py`

### 2.3 Sweep and benchmark entry points

- `src/run_binary_regression_sweep.py`
- `src/benchmark_packed_binary_kernels.py`
- `src/benchmark_model_inference.py`

### 2.4 Core memory documents

- `docs/BINARY_REGRESSION_EXPERIMENT_LOG.md`
- `docs/CURRENT_STATUS.md`
- `docs/ROADMAP.md`

## 3. Current Best Findings

### 3.1 Quality-oriented binary regression point

On the regression task used in the repo:

- binary hidden dims `(8,)`
- learning rate `3e-3`
- epochs `75`
- RMSE `12.4447`
- total runtime `8.5400s`

This configuration is slightly better than the dense baseline on quality while
remaining very close on runtime.

### 3.2 Speed-oriented binary regression point

On the same task:

- binary hidden dims `(8,)`
- learning rate `3e-3`
- epochs `40`
- RMSE `15.1069`
- total runtime `3.5248s`

This configuration is materially faster than dense while staying close on
accuracy.

### 3.3 Triton kernel finding

The packed Triton binary inference path is already faster than the unpacked
reference path on larger matrix shapes.

Latest larger-shape decision bundle on `NVIDIA L4`:

- `(512, 2048, 2048)`: about `2.65x` speedup
- `(1024, 4096, 4096)`: about `2.54x` speedup
- `(1024, 8192, 8192)`: about `2.72x` speedup
- observed max absolute error stayed around `0.0018` to `0.0020`

### 3.4 End-to-end model finding

The Triton advantage survives at full-model inference level, not just in an
isolated microkernel benchmark.

Important nuance from the refreshed benchmarks:

- on the original tiny `10`-feature regression benchmark, latency is mostly
  overhead-bound, so large batch sizes do not materially change the story
- on a wider trained-model benchmark with `1024` input features and
  hidden dims `(1024, 1024)`, Triton improves binary latency at batch sizes
  `512`, `2048`, and `8192`, but regresses at `16384`
- targeted profiling at shape `(16384, 1024, 1024)` shows that regression is
  kernel-local: the packed Triton binary layer itself is slower than the
  reference path there, not just the full model around it
- wide dense results are highly sensitive to `torch.set_float32_matmul_precision`
  on `NVIDIA L4`; under `medium`, the dense `(1024, 1024)` model becomes much
  faster than the earlier default-precision measurements suggested

## 4. What Has Been Validated

The following has already been checked and should be treated as known working
ground unless a future change breaks it.

### 4.1 Focused tests

The focused regression, sweep, comparison, and kernel tests have been passing
during the recent iterations.

### 4.2 Comparison workflow

`src/run_regression_comparison.py` now prints:

- task-quality metrics
- training/runtime deltas
- model-level inference benchmark records

### 4.3 Artifact export

The repository now emits machine-readable benchmark artifacts under `/mnt`, with
default benchmark outputs written to `/mnt/binary_nn/artifacts/`.

The latest decision-grade refresh is stored under:

- `/mnt/binary_nn/artifacts/2026-03-17-decision/`

## 5. Assumptions For The Next Session

The next session should assume:

- binary residual regression is the correct baseline, not the original plain
  binary MLP
- the dense shortcut is currently a feature, not a bug or temporary hack
- the Triton packed path is real and worth extending
- the smoke artifacts are still validation-only, but a decision-grade refresh
  now exists under `/mnt/binary_nn/artifacts/2026-03-17-decision/`
- the small `10`-feature model benchmark is useful for workflow validation, but
  the wider `1024`-feature benchmark is the more representative systems read
- the current systems question is no longer whether Triton helps at all; it is
  why the current kernel loses at shape `(16384, 1024, 1024)`
- future wide dense-versus-binary comparisons should set float32 matmul
  precision explicitly, preferably `high` or `medium`, to avoid undercounting
  dense and binary no-Triton GEMM throughput on Tensor Core GPUs
- the next major design decision is not whether to do custom kernels, but
  whether to stay strict binary or begin a parallel ternary or int2 path

## 6. Recommended First Read Order

If a future session needs to rebuild context fast, read in this order:

1. `docs/CURRENT_STATUS.md`
2. `docs/ROADMAP.md`
3. `docs/BINARY_REGRESSION_EXPERIMENT_LOG.md`
4. `src/regression_models.py`
5. `src/binary_kernels.py`
