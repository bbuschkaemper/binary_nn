# Ternary Research Experiment Log

Last updated: 2026-03-18

This document records the concrete ideas tested so far for the new ternary
research branch, the measurements that came out of those tests, and the current
best ternary configurations.

## Document Role

Use this file when you need the detailed story for the ternary branch: what was
implemented, what worked, what failed, and which artifacts contain the latest
evidence.

## 1. Goal

The ternary workstream was started to answer a narrower question than the
original binary proof of concept:

- can we add a BitNet-aligned ternary path beside the working binary baseline
- can that ternary path run efficiently on CPU using sparse execution
- can we find a direct-discrete training route that is scientifically meaningful,
  not just another quantized surrogate

The desired result was a proof of concept with:

- equal or better quality than a dense BF16 baseline on at least one benchmark
- faster CPU inference on that same benchmark

## 2. Implemented Branches

### 2.1 STE ternary branch

Implemented in:

- `TernaryLinear`
- `TernaryRegressor`
- `src/run_ternary_regression.py`

Design:

- latent floating-point weight
- ternary quantization in `{-1, 0, +1}` via STE
- cached sparse CPU inference during eval
- dense shortcut kept on by default

This branch is the quality-oriented ternary baseline.

### 2.2 Shadow-free ternary branch

Implemented in:

- `ShadowFreeTernaryLinear`
- `ShadowFreeTernaryRegressor`
- `src/run_shadowfree_ternary_regression.py`

Design:

- direct ternary state stored in `weight_state`
- accumulated backward evidence in `_accumulated_evidence`
- thresholded, hysteretic direct state updates
- cached sparse CPU inference during eval
- dense shortcut kept on by default

This branch is the direct-discrete research branch.

### 2.3 Shared comparison and benchmark tooling

Implemented in:

- `src/run_ternary_research_comparison.py`

It compares:

- dense BF16 baseline
- one selected ternary branch (`shadowfree` or `ste`)
- CPU inference with sparse off and sparse on

It writes:

- comparison JSON
- CPU benchmark CSV

## 3. Benchmark Surfaces Used

### 3.1 Original linear regression task

This is the existing repository benchmark:

- `target_kind="linear"`
- synthetic regression via `make_regression`
- dense shortcut is especially strong here because the target is highly linear

This benchmark is still valid for a first proof of concept, but it is easy.

### 3.2 New nonlinear residual benchmark

Added in `src/regression_data.py`:

- `target_kind="nonlinear_residual"`
- linear backbone plus sinusoidal, tanh, and quadratic residual terms

Purpose:

- make the residual ternary branch do real nonlinear work
- detect when a ternary result is only exploiting the linear shortcut

## 4. Main Experiments and Results

### 4.1 Shadow-free proof of concept on the original linear task

Command surface:

- `src/run_ternary_research_comparison.py --model-family shadowfree`

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`
- `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc-cpu.csv`

Measured result:

Dense BF16 baseline:

- hidden dims `(64, 32)`
- RMSE `14.8845`
- total runtime `6.9923s`

Shadow-free ternary residual:

- hidden dims `(64,)`
- initial density `0.25`
- update interval `1`
- RMSE `12.1707`
- total runtime `7.9168s`
- final ternary nonzero density `0.0172`

CPU inference speedup of shadow-free sparse path versus dense:

- batch `32`: `1.03x`
- batch `128`: `3.46x`
- batch `512`: `3.54x`

Interpretation:

- this is a real proof-of-concept win on the established repo benchmark
- the win is inference-only; training was slightly slower than the dense BF16
  baseline
- the ternary branch became extremely sparse, which is what enabled the CPU win

Important caution:

- the dense shortcut is doing a lot of the work here
- the shadow-free result should be read as a sparse residual decomposition

### 4.2 Negative result: shadow-free without shortcut

On the original linear benchmark, removing the shortcut caused the direct
shadow-free path to degrade badly even with larger hidden sizes.

Observed ranges:

- RMSE roughly `85` to `181`

Interpretation:

- the current shadow-free branch is not yet a no-shortcut ternary MLP solution
- shortcut ablations are important whenever this branch is discussed

### 4.3 Negative result: shadow-free on the nonlinear benchmark

Direct shadow-free ternary on `target_kind="nonlinear_residual"` stayed far
behind the dense baseline.

Representative observation:

- dense BF16 baseline RMSE around `15.6`
- shadow-free ternary residual RMSE around `23.6`

Interpretation:

- the current direct-discrete update rule is not yet good enough for the harder
  nonlinear regime

### 4.4 STE ternary follow-up on the nonlinear benchmark

Command surface:

- `src/run_ternary_research_comparison.py --model-family ste --target-kind nonlinear_residual ...`

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-ste-nonlinear-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-ste-nonlinear-followup-cpu.csv`

Measured result:

Dense BF16 baseline:

- hidden dims `(64, 32)`
- RMSE `15.6111`
- total runtime `6.9419s`

STE ternary residual:

- hidden dims `(64, 32)`
- threshold scale `0.5`
- RMSE `15.8289`
- total runtime `15.5148s`
- nonzero density `0.6851`

CPU inference result:

- sparse CPU path is slower than dense at every measured batch
- the ternary branch is still too dense for sparse execution to pay off

Interpretation:

- this is the strongest harder-task ternary quality result so far
- the quality gap is small
- the systems story is still missing because density stayed high

## 5. Main Conclusions

The ternary workstream now supports three clear conclusions.

### 5.1 Sparse CPU wins are possible

The shadow-free branch proved that cached sparse CPU inference can beat a dense
BF16 baseline on the established repo benchmark while matching or exceeding its
accuracy.

### 5.2 Direct-discrete optimization is not solved yet

The same shadow-free branch fails on the harder nonlinear benchmark and collapses
without the shortcut. That means the direct-discrete update rule is still a live
research problem, not a finished method.

### 5.3 The harder-task ternary quality baseline is STE for now

The STE ternary branch is the better quality anchor on the nonlinear benchmark,
but it is too dense to deliver CPU sparse wins yet.

## 6. Best Current Research Framing

The most defensible current framing is:

- shadow-free ternary is a promising sparse residual path with real CPU wins on
  easy linear-style workloads
- STE ternary is the current quality anchor on harder tasks
- the next meaningful experiment is probably not "more of the same" on either
  branch alone
- the next meaningful experiment is a hybrid or staged method that uses STE to
  reach a good ternary solution and then consolidates into a sparse shadow-free
  state

## 7. Recommended Next Experiments

1. Add an STE-to-shadow-free warm-start handoff experiment.
2. Add target-density or structured-sparsity controls to the ternary branches.
3. Rebenchmark CPU latency only after density has fallen materially on the
   nonlinear benchmark.
4. Explore a block-sparse or bit-packed ternary CPU kernel once the model-side
   sparsity story is stable.
5. Avoid strong claims about GPU training acceleration until step-time evidence
   exists.
