# Current Status

Last updated: 2026-03-18

This document is the short operational memory for the repository. It is meant
to answer three questions quickly:

1. What is implemented right now?
2. What has been validated already?
3. What should the next session assume as the starting point?

## Document Role

Use this file as the first read when resuming work. It is the shortest path to
the current implementation state.

## 1. Current Technical State

The repository now has five working layers of functionality.

### 1.1 Dense and binary baseline path

- Dense regression baseline exists and is stable.
- Binary regression baseline exists and is stable.
- Both still run through the shared Lightning training utilities in
  `src/regression_experiment.py`.
- The binary path remains the main stable low-bit baseline.

### 1.2 Binary systems path

- `BinaryLinear` still has an eval-only packed Triton inference path on CUDA.
- Training still uses the normal PyTorch floating-point path for binary models.
- Packed inference is implemented for binary sign weights plus a per-output
  scale.

### 1.3 New ternary research path

- `src/regression_models.py` now contains two ternary families:
  - `TernaryLinear` / `TernaryRegressor`
    - STE-trained ternary weights in `{-1, 0, +1}`
    - cached sparse CPU inference path
  - `ShadowFreeTernaryLinear` / `ShadowFreeTernaryRegressor`
    - discrete ternary state stored directly
    - batch-evidence accumulation plus thresholded direct state updates
    - cached sparse CPU inference path
- `src/run_ternary_regression.py` trains the STE ternary branch.
- `src/run_shadowfree_ternary_regression.py` trains the direct-discrete
  shadow-free branch.
- `src/run_hybrid_ternary_regression.py` runs an STE-to-shadow-free handoff and
  can optionally prune the transferred ternary state to a target density before
  recovery training.
- `src/run_ternary_research_comparison.py` compares a dense baseline against one
  ternary branch (`shadowfree`, `ste`, `hybrid`, or `projected`) and writes JSON
  plus CSV artifacts under `/mnt`.

### 1.4 Shared training and data improvements

- `TrainingConfig` now supports explicit Lightning `precision`, so BF16 dense
  baselines can be trained from the shared path.
- BF16 prediction outputs are now cast back to float32 before NumPy conversion,
  which fixed an actual bug in the shared evaluation path.
- `RegressionDataConfig` now supports `target_kind="nonlinear_residual"` plus
  nonlinear-control knobs. The original linear regression benchmark remains the
  default.

### 1.5 Benchmarking and artifact tooling

- Binary benchmarking and artifact export remain intact.
- The new ternary comparison script writes:
  - full comparison JSON
  - CPU inference latency CSV
- Generated artifacts and Lightning checkpoints still route under `/mnt`, not
  back into the repository tree.

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
- `src/run_shadowfree_ternary_regression.py`
- `src/run_ternary_regression.py`
- `src/run_hybrid_ternary_regression.py`
- `src/run_ternary_research_comparison.py`

### 2.3 Data and experiment configuration

- `src/regression_data.py`
- `src/regression_experiment.py`

### 2.4 Core memory documents

- `docs/TERNARY_RESEARCH_EXPERIMENT_LOG.md`
- `docs/BINARY_REGRESSION_EXPERIMENT_LOG.md`
- `docs/CURRENT_STATUS.md`
- `docs/ROADMAP.md`

## 3. Current Best Findings

### 3.1 Quality-oriented binary regression point

On the original linear regression task used in the repo:

- binary hidden dims `(8,)`
- learning rate `3e-3`
- epochs `75`
- RMSE `12.4447`
- total runtime `8.5400s`

This configuration is still slightly better than the dense baseline on quality
while remaining close on runtime.

### 3.2 Speed-oriented binary regression point

On the same task:

- binary hidden dims `(8,)`
- learning rate `3e-3`
- epochs `40`
- RMSE `15.1069`
- total runtime `3.5248s`

This configuration is still materially faster than dense while staying close on
accuracy.

### 3.3 Shadow-free ternary proof of concept

The new direct-discrete ternary branch now has one real proof-of-concept win on
the original linear regression benchmark.

Dense BF16 baseline on `NVIDIA L4`:

- hidden dims `(64, 32)`
- RMSE `14.8845`
- total runtime `6.9923s`

Shadow-free ternary residual on the same task:

- hidden dims `(64,)`
- initial density `0.25`
- update interval `1`
- RMSE `12.1707`
- total runtime `7.9168s`
- final ternary nonzero density `0.0172`

CPU inference result from `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`:

- batch `32`: about `1.03x` speedup versus dense
- batch `128`: about `3.46x` speedup versus dense
- batch `512`: about `3.54x` speedup versus dense

Important nuance:

- this is an inference win, not a training-speed win
- the shadow-free branch became extremely sparse on the linear task
- the dense shortcut remains important; the result should be read as a sparse
  residual decomposition, not as a no-shortcut ternary MLP victory

### 3.4 Nonlinear STE follow-up result

On the harder `target_kind="nonlinear_residual"` benchmark:

Dense BF16 baseline:

- hidden dims `(64, 32)`
- RMSE `15.6111`

STE ternary residual:

- hidden dims `(64, 32)`
- threshold scale `0.5`
- RMSE `15.8289`
- ternary nonzero density `0.6851`

Interpretation:

- the STE ternary branch is now the stronger quality baseline on the harder
  nonlinear task
- the quality gap is small, but CPU sparse inference is still slower than dense
  at this density
- the shadow-free route is not yet the right harder-task baseline

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-ste-nonlinear-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-ste-nonlinear-followup-cpu.csv`

### 3.5 Hybrid and projected handoff follow-ups

Two STE-to-shadow-free handoff variants were tested on the same nonlinear task.

Naive hybrid consolidation with free-running direct-discrete updates:

- artifact: `/mnt/binary_nn/artifacts/2026-03-18-hybrid-nonlinear-followup.json`
- RMSE `25.8914`
- ternary nonzero density `0.1105`
- sparse CPU speedup versus dense:
  - batch `128`: about `1.17x`
  - batch `512`: about `1.36x`

Interpretation:

- this path can force sparsity, but the current direct-discrete consolidation
  rule destroys too much quality on the harder task
- it is not the right default handoff recipe

Density-projected handoff with light recovery training:

- artifact: `/mnt/binary_nn/artifacts/2026-03-18-projected-nonlinear-followup.json`
- target density `0.35`
- warm-start epochs `50`
- recovery epochs `25`
- recovery learning rate `3e-4`
- RMSE `18.4060`
- ternary nonzero density `0.3500`

Interpretation:

- this is the best quality observed so far at a sparse-friendly density on the
  nonlinear task
- the current sparse CPU kernel still loses to dense at this density, so the
  next bottleneck is either lower density without another accuracy cliff or a
  better sparse/packed ternary kernel

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-hybrid-nonlinear-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-hybrid-nonlinear-followup-cpu.csv`
- `/mnt/binary_nn/artifacts/2026-03-18-projected-nonlinear-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-projected-nonlinear-followup-cpu.csv`

### 3.6 Binary Triton result still matters

The packed Triton binary inference path is still faster than the unpacked
reference path on larger matrix shapes, but the known large-batch regression at
`(16384, 1024, 1024)` remains unresolved.

## 4. What Has Been Validated

The following has already been checked and should be treated as known working
ground unless a future change breaks it.

### 4.1 Tests

- the full test suite passes
- new ternary sparse-inference equivalence tests pass
- new nonlinear-data smoke tests pass
- new STE-to-shadow-free conversion and target-density projection tests pass

### 4.2 BF16 shared path

- BF16 dense baselines now work from the shared training path
- the earlier BF16 prediction-to-NumPy failure has been fixed

### 4.3 Artifact export

The repository now emits machine-readable ternary comparison artifacts under
`/mnt/binary_nn/artifacts/` in addition to the earlier binary artifacts.

The most important new result bundle is:

- `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`

## 5. Assumptions For The Next Session

The next session should assume:

- binary residual regression is still the correct stable baseline
- the shadow-free ternary route is validated only on the easy linear benchmark
- the shadow-free CPU win comes from extreme sparsification plus cached sparse
  CPU execution
- the current shadow-free result is not evidence of faster GPU training
- the nonlinear residual benchmark is the better sanity check for whether the
  ternary branch is actually carrying nonlinear load
- the STE ternary branch is currently the better harder-task quality reference
- naive free-running STE-to-shadow-free consolidation is too destructive on the
  nonlinear task to be the default bridge
- density-projected STE-to-shadow-free handoff is the better current sparse
  bridge, but it still needs either lower density or a faster sparse kernel
- the next major ternary question is not whether sparse CPU inference can work at
  all; it is how to keep the ternary branch both useful and sparse on harder
  tasks

## 6. Recommended First Read Order

If a future session needs to rebuild context fast, read in this order:

1. `docs/CURRENT_STATUS.md`
2. `docs/TERNARY_RESEARCH_EXPERIMENT_LOG.md`
3. `docs/ROADMAP.md`
4. `src/run_ternary_research_comparison.py`
5. `src/regression_models.py`
6. `src/regression_data.py`
