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

### 3.6 Wide nonlinear follow-up with a tuned dense reference

On a wider nonlinear benchmark with:

- `n_features=256`
- `n_informative=256`
- `nonlinear_pair_count=64`
- `nonlinear_scale=1.5`
- hidden dims `(256, 128)`

the dense BF16 reference was explicitly retuned first.

Best observed standalone dense setting:

- learning rate `3e-4`
- epochs `75`
- RMSE `10.0171`
- total runtime `6.6501s`

Wide STE follow-up artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-followup.json`
- RMSE `9.7710`
- ternary density about `0.7461`

Interpretation:

- at this width, ternary quality is no longer the main problem
- STE beats the tuned dense reference on RMSE
- inference cost is still much worse than dense, especially on CPU sparse mode

Wide projected follow-up artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-followup.json`
- RMSE `9.3466`
- ternary density about `0.3500`

Interpretation:

- this is now the strongest ternary quality result in the repo at a sparse-ready
  density
- projected handoff beats both the tuned dense reference and the wide STE point
  on RMSE
- sparse CPU inference still loses to dense, but the penalty is much smaller than
  for wide STE
- one paired comparison run reported projected total runtime below the dense run,
  but standalone dense timing was also much faster in the tuning sweep, so GPU
  speed should still be treated as unresolved rather than claimed

Additional negative follow-ups:

- per-row projection balancing and layer-skewed density schedules did not improve
  the projected `0.35` nonlinear point
- teacher-style activation calibration of projected scales and biases did not
  improve the projected `0.35` point either
- a naive index-based CPU kernel prototype for `ShadowFreeTernaryLinear` was much
  slower than both dense execution and the current `torch.sparse.mm` path

### 3.6.1 CPU systems follow-up on the wide projected frontier

A dedicated CPU systems pass was run directly on the wide projected `0.35`
frontier.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-indexed-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-csr-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-cached-dense-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-final-followup.json`

What was learned:

- a forced indexed-gather ternary kernel was numerically correct but catastrophically
  slower on the real projected model, especially at larger batch sizes
- switching the sparse backend to CSR made the sparse path cleaner and much more
  realistic to benchmark, but a forced sparse path at density `0.35` still lost to
  dense execution at every tested batch size
- the largest practical CPU win came from caching the exact dense ternary weight in
  eval mode instead of rebuilding it every forward pass

Best current wide projected CPU numbers after the cache change:

- tuned dense BF16 baseline latency: `0.1775 / 0.2450 / 0.4441 / 1.0431 ms` at batch
  `32 / 128 / 512 / 2048`
- projected cached-dense latency: `0.2498 / 0.3883 / 0.7959 / 1.1286 ms` with RMSE
  `9.3466`
- forced sparse CSR latency: `0.6591 / 0.7706 / 1.5097 / 2.6216 ms`

Interpretation:

- at density `0.35`, cached dense ternary inference is now the correct default CPU
  path
- sparse inference should only be enabled at materially lower density, or after a
  more aggressive packed or structured kernel exists
- the CPU gap to the tuned dense baseline is now much smaller than before,
  especially at the largest batch, but it is still not yet a speed win

### 3.6.2 Lower-density projected follow-up and packed-kernel evaluation

A dual follow-up then tested both remaining levers:

- lower projected density on the wide nonlinear benchmark
- a genuinely packed ternary CPU lookup prototype

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density025-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density020-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-dual-eval-lean.json`
- `/mnt/binary_nn/artifacts/2026-03-18-dual-eval-lean.csv`

What was learned:

- projected target density `0.20` preserved wide-benchmark quality surprisingly well,
  reaching RMSE `9.3308` at density `0.20`
- projected target density `0.25` also held the same wide quality regime (`9.3268`),
  which supports the conclusion that the projected frontier can move materially
  below `0.35`
- this established projected `0.20` as the best observed **replicated** ternary
  quality-versus-density point in the repo so far
- cached dense inference remained the fastest projected CPU path in the clean
  follow-up; forced sparse CSR was still slower even at density `0.20`
- the packed lookup prototype was strongly negative at both densities and both tested
  block sizes (`4` and `8`)

Clean lean CPU compare on the wide benchmark:

- dense baseline latency: `0.2638 ms` at batch `128`, `1.5478 ms` at batch `2048`
- projected `0.35` cached-dense latency: `0.4819 / 1.1263 ms`
- projected `0.35` forced sparse CSR latency: `0.9765 / 2.3463 ms`
- projected `0.35` packed lookup (`b4 / b8`): `4.4172 / 5.2148 ms` at batch `128`,
  `230.4038 / 100.0477 ms` at batch `2048`
- projected `0.20` cached-dense latency: `0.3466 / 1.0618 ms`
- projected `0.20` forced sparse CSR latency: `0.6607 / 2.7884 ms`
- projected `0.20` packed lookup (`b4 / b8`): `2.8692 / 4.0883 ms` at batch `128`,
  `219.2001 / 80.7821 ms` at batch `2048`

Interpretation:

- the model side moved forward: lower projected density is now a better frontier than
  `0.35`
- the systems side did not: sparse CSR and lookup-packed kernels both still lose to
  cached dense inference on the projected models
- because the clean packed-kernel follow-up intentionally used short timing loops to
  keep the negative result affordable, any apparent dense-versus-cached-dense
  crossover at very large batch should be treated as suggestive rather than
  decision-grade

### 3.6.3 Seed replication of projected `0.20` and a single-seed `0.15` follow-up

The next follow-up checked whether the new `0.20` point was stable across more than
one seed, and then pushed one step lower.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density020-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density020-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density015-seed42.json`

Seed replication of projected `0.20`:

- seed `7`: dense RMSE `9.9809`, projected RMSE `9.4234`, density `0.20`
- seed `123`: dense RMSE `9.6383`, projected RMSE `9.0070`, density `0.20`

Single-seed lower-density follow-up:

- seed `42`, projected `0.15`: dense RMSE `10.0171`, projected RMSE `9.3295`,
  density about `0.15`

CPU inference on the `0.15` follow-up:

- cached dense projected latency: `0.3695 ms` at batch `128`, `1.0463 ms` at batch
  `2048`
- forced sparse CSR latency: `0.7057 ms` at batch `128`, `2.0630 ms` at batch
  `2048`

Interpretation:

- projected `0.20` is now stable across multiple seeds and should be treated as the
  current replicated frontier
- projected `0.15` is the strongest single-seed quality-versus-density point seen so
  far, but it still needs replication before it replaces `0.20` as the default
  recommendation
- even at density `0.15`, sparse CSR still loses to cached dense projected inference

### 3.6.4 Replication of projected `0.15` and a single-seed `0.10` follow-up

After the first promising `0.15` run, that point was replicated on the same extra
seeds previously used for the `0.20` check.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density015-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density015-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density010-seed42.json`

Replication results for projected `0.15`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.3295`, density `0.15`
- seed `7`: dense RMSE `9.9809`, projected RMSE `9.4330`, density `0.15`
- seed `123`: dense RMSE `9.6383`, projected RMSE `9.0266`, density `0.15`

Single-seed lower-density follow-up:

- seed `42`, projected `0.10`: dense RMSE `10.0171`, projected RMSE `9.3453`,
  density about `0.10`

CPU inference on the `0.10` follow-up:

- cached dense projected latency: `0.3495 ms` at batch `128`, `1.0411 ms` at batch
  `2048`
- forced sparse CSR latency: `0.5716 ms` at batch `128`, `1.7894 ms` at batch
  `2048`

Interpretation:

- projected `0.15` is now the current best **replicated** quality-versus-density
  point in the repo
- projected `0.10` is the strongest single-seed candidate so far, but it still
  needs replication before it replaces `0.15` as the default projected frontier
- cached dense projected inference is still the correct CPU default even at `0.10`

### 3.6.5 Replication of projected `0.10` and a single-seed `0.05` follow-up

After projected `0.10` looked strong on seed `42`, that point was replicated on the
same additional seeds already used for the projected `0.15` and `0.20` checks.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density010-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density010-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density005-seed42.json`

Replication results for projected `0.10`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.3453`, density `0.10`
- seed `7`: dense RMSE `9.9809`, projected RMSE `9.4950`, density `0.10`
- seed `123`: dense RMSE `9.6383`, projected RMSE `9.0607`, density `0.10`

Single-seed lower-density follow-up:

- seed `42`, projected `0.05`: dense RMSE `10.0171`, projected RMSE `9.4578`,
  density about `0.05`

CPU timing note:

- the new `0.10` replication timing runs were noisy enough that only the quality
  result should be treated as decision-grade
- on the cleaner single-seed `0.05` follow-up, cached dense projected latency was
  `0.4216 / 1.1203 ms` at batch `128 / 2048`, while forced sparse CSR was
  `0.7145 / 1.6468 ms`

Interpretation:

- projected `0.10` is now the current best **replicated** quality-versus-density
  point in the repo
- projected `0.05` is the strongest single-seed candidate so far, but it still
  needs replication before it replaces `0.10` as the default projected frontier
- cached dense projected inference remains the CPU default; the new replications do
  not justify a stronger CPU speed claim

### 3.6.6 Replication of projected `0.05` and a lower-density `0.02` probe

After projected `0.05` looked strong on seed `42`, that point was replicated on the
same additional seeds already used for the projected `0.10`, `0.15`, and `0.20`
checks.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density005-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density005-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density002-seed42.json`

Replication results for projected `0.05`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.4578`, density `0.05`
- seed `7`: dense RMSE `9.9809`, projected RMSE `9.5298`, density `0.05`
- seed `123`: dense RMSE `9.6383`, projected RMSE `9.1808`, density `0.05`

Lower-density follow-up:

- seed `42`, projected `0.02`: dense RMSE `10.0171`, projected RMSE `9.7952`,
  density about `0.02`

CPU note:

- cached dense projected inference still beat forced sparse CSR on the clean `0.05`
  and `0.02` runs
- no new CPU speed claim should be made here; the main new result is model-side
  density robustness

Interpretation:

- projected `0.05` is now the current best **replicated** quality-versus-density
  point in the repo
- projected `0.02` still beats dense on seed `42`, but it is the first lower-density
  follow-up that shows a noticeable quality bend relative to the `0.05` frontier
- the model-side density chase is now far enough along that future work can either
  replicate `0.02` or pivot back to structured sparsity and CPU execution around the
  stronger replicated `0.05` point

### 3.7 Binary Triton result still matters

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
  bridge, and on the wider `256`-feature nonlinear benchmark the best **replicated**
  point is now projected `0.05`
- projected `0.02` is the strongest lower-density single-seed candidate so far, but
  it already shows the first meaningful quality bend relative to the `0.05` frontier
- cached dense ternary inference is still the best CPU default on the projected
  frontier; forced sparse CSR and the lookup-packed prototype are both slower there
- dense wide references must be retuned before using wide comparisons as evidence
- the current wide results are not yet enough to claim a robust GPU speed win
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
