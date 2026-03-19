# Ternary Research Experiment Log

Last updated: 2026-03-19

This document records the ternary workstream through the new GPU-first lens: what was
implemented, what worked, what failed, and what the results imply for the next low-bit
training idea.

## Document Role

Use this file when you need the detailed ternary story rather than the short operational
summary in `CURRENT_STATUS.md`.

## 1. Goal Under The New Framing

The original ternary goal was to beat a dense BF16 baseline on quality while also
getting a real CPU inference win.

The goal is now broader and more GPU-first:

- keep or improve low-bit quality versus tuned dense BF16
- discover a training idea that can improve GPU training cost
- eventually produce a representation that also supports faster GPU inference

CPU findings still matter, but mainly as supporting systems evidence rather than the
primary decision criterion.

## 2. Implemented Branches

### 2.1 STE ternary branch

Implemented in:

- `TernaryLinear`
- `TernaryRegressor`
- `src/run_ternary_regression.py`

Role:

- simplest harder-task ternary baseline
- best local quality anchor before the projected handoff path took over

### 2.2 Shadow-free ternary branch

Implemented in:

- `ShadowFreeTernaryLinear`
- `ShadowFreeTernaryRegressor`
- `src/run_shadowfree_ternary_regression.py`

Role:

- most novel direct-discrete optimization idea in the repo
- strongest proof of concept on the easy benchmark
- not yet stable enough on the harder benchmark to be the main path

### 2.3 Hybrid / projected handoff branch

Implemented in:

- `src/run_hybrid_ternary_regression.py`

Role:

- warm-start from STE
- hand off to a sparse ternary state
- optionally project to a target density
- current best bridge between low-bit quality and a discrete ternary representation

### 2.4 Refresh-scheduled projected branch

Implemented in:

- `RefreshScheduledProjectedTernaryLinear`
- `RefreshScheduledProjectedTernaryRegressor`
- `src/run_hybrid_ternary_regression.py` via `consolidation_variant="refresh_projected"`

Role:

- keep a latent ternary weight while holding a cached projected forward state fixed
  for `K` steps
- first concrete implementation of the new GPU-first training hypothesis

### 2.5 Controlled refresh-projected branch

Implemented in:

- `ControlledRefreshProjectedTernaryBlock`
- `ControlledRefreshProjectedTernaryRegressor`
- `src/run_hybrid_ternary_regression.py` via
  `consolidation_variant="controlled_refresh_projected"`
- `src/run_ternary_research_comparison.py` via
  `--model-family controlled_refresh_projected`

Role:

- keep the refresh-projected low-bit bulk path
- add a small dense control path for hard-to-quantize corrections
- add ternary hidden activations between blocks
- serve as the first concrete probe of the broader architecture direction

### 2.6 Shared comparison tooling

Implemented in:

- `src/run_ternary_research_comparison.py`

Role:

- compare dense BF16 baseline versus one ternary family
- write JSON plus CPU benchmark CSV artifacts under `/mnt`
- for CUDA runs, attach repeated `fit`, `test`, and `predict` step benchmarks plus
  peak memory to each serialized runtime
- current decision-grade evidence path for ternary work

## 3. Benchmark Surfaces Used

### 3.1 Original linear regression task

Useful for:

- quick proof-of-concept checks
- seeing whether a low-bit branch can beat dense at all

Limitation:

- too easy to serve as the main research gate

### 3.2 Nonlinear residual benchmark

Useful for:

- making the low-bit branch do real nonlinear work
- exposing when a branch only benefits from the dense shortcut

### 3.3 Wide nonlinear benchmark (`256` features)

Current role:

- main local quality stress test
- default place to compare dense BF16, STE ternary, and projected ternary

## 4. Main Experiment Story

### 4.1 Shadow-free proof of concept on the easy benchmark

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`

Measured result:

Dense BF16 baseline:

- RMSE `14.8845`
- total runtime `6.9923s`

Shadow-free ternary residual:

- RMSE `12.1707`
- total runtime `7.9168s`
- final density `0.0172`

CPU inference speedup versus dense:

- batch `32`: `1.03x`
- batch `128`: `3.46x`
- batch `512`: `3.54x`

Interpretation:

- this remains the strongest local proof that a low-bit model can beat a dense BF16
  baseline on both quality and inference speed
- it also proves that direct-discrete ternary updates are scientifically worth taking
  seriously
- however, this result lives on the easy benchmark and therefore does not settle the
  main GPU-first research question by itself

### 4.2 Shadow-free negative results on the harder benchmark

The same direct-discrete branch failed to hold up cleanly once the benchmark became more
nonlinear.

Local conclusion:

- shadow-free is still the right place to look for update-rule novelty
- it is not yet the right quality anchor for the harder task

### 4.3 STE ternary established the harder-task baseline

Wide reference artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-followup.json`

Result on the wide nonlinear benchmark:

- dense BF16 RMSE `10.0171`
- STE ternary RMSE `9.7710`
- dense BF16 total runtime `11.8164s`
- STE ternary total runtime `15.7970s`

Interpretation:

- STE showed that low-bit quality on the harder task is real
- but it was too dense and too slow to be the final answer
- it remains the simplest harder-task baseline beside projected ternary

### 4.4 Projected handoff became the best bridge

Wide projected follow-up artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-followup.json`

Early wide projected point:

- dense BF16 RMSE `10.0171`
- projected ternary RMSE `9.3466`
- projected density about `0.35`

Interpretation:

- projected handoff immediately became the strongest quality-versus-density bridge in
  the ternary branch
- this was the point where the ternary story stopped being “can ternary work?” and
  became “how low can the projected frontier go?”

### 4.5 The wide projected frontier moved much lower than expected

Across the later density sweep, the projected frontier kept holding quality as density
moved lower.

Replicated mean dense BF16 RMSE across the three-seed wide benchmark set:

- `9.8788`

Key projected frontier points:

| Density | Status | Mean / representative projected RMSE | Interpretation |
| --- | --- | --- | --- |
| `0.05` | replicated | `9.3895` mean | clearly better than dense |
| `0.02` | replicated | `9.4076` mean | still better than dense |
| `0.01` | replicated | `9.2408` mean | stronger than expected |
| `0.001` | replicated | `9.2311` mean | current best replicated frontier |
| `0.005` | seed `42` only | `9.2961` | still beats dense |
| `0.0025` | seed `42` only | `9.2808` | still beats dense |

Training runtime at the current best replicated projected point (`0.001`):

- dense BF16 mean total runtime: `7.9445s`
- projected ternary mean total runtime: `13.8249s`

Interpretation:

- the model-side quality problem is no longer the main blocker on this benchmark
- the projected family is now the best quality anchor in the repo
- the missing piece is speed, not proof of quality

### 4.6 Structured sparsity pivot was negative

The first structured follow-up used a `row_block` projection rule near the projected
frontier.

Results at target density `0.05`:

- block size `8`: structured projected RMSE `10.0622`
- block size `16`: structured projected RMSE `10.1407`
- dense BF16 reference on the same seed: `10.0171`

Interpretation:

- naive row-wise block structure is not the bridge we want
- if structure work continues, it should be materially different from this rule

### 4.7 Systems interpretation under the new goal

CPU-side findings are still useful, but the interpretation has changed.

What they tell us now:

- forced sparse CSR remained slower than cached dense projected inference across the
  projected frontier
- projected auto-sparse was disabled because the sparse path was not honestly winning
- small-batch timing on the shared machine became noisy in some ultra-low runs, so
  isolated speedups should not be treated as decision-grade evidence

Why this still matters for a GPU-first goal:

- it shows that lower density alone does not create speed
- kernel and representation design still dominate the systems story

### 4.8 GPU measurement plumbing was added to the ternary comparison flow

Smoke-validation artifact:

- `/mnt/binary_nn/artifacts/gpu_stage_smoke.json`

What changed:

- CUDA comparison runs now serialize nested `runtime.stage_benchmarks.fit`,
  `test`, and `predict` summaries
- each stage now records repeated mean step time, step-time variance, and peak GPU
  memory
- comparison JSON now also includes a `gpu_stage_benchmark_comparison` section for
  dense-versus-ternary stage deltas

Why this matters:

- the repo no longer has to rely only on one-off end-to-end wall-clock totals for
  GPU-side comparisons
- future projected-refresh experiments can be judged on a stable measurement path
- the smoke numbers themselves are only plumbing validation, not decision-grade
  research claims

### 4.9 First refresh-projected prototype

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-refresh-projected-k4-density0001-seed42.json`

Measured result on the wide nonlinear benchmark:

- projected ternary RMSE `9.2760`
- refresh-projected ternary RMSE `9.2930`
- fit-step speedup versus projected: `2.24x`
- test-step speedup versus projected: `1.78x`
- predict-step speedup versus projected: `1.15x`

Interpretation:

- this is the first concrete evidence that the new training idea may preserve the
  projected quality regime while lowering measured GPU stage cost
- the quality delta versus projected is small on this seed (`+0.0170` RMSE)
- this is promising but still only single-seed evidence and must be replicated

### 4.10 Refresh-projected replication and sweep

Decision artifacts:

- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k4-density0001-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k4-density0001-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k1-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k2-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k4-density0001-every2-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k4-density0001-every4-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123.json`

Main results:

- `K=4` mean RMSE across seeds `42`, `7`, and `123`: `9.2377`
- `K=8` mean RMSE across the same seeds: `9.2311`
- projected ternary mean RMSE across the same seeds: `9.2311`
- `K=8` mean fit-step time across those seeds: `2.1305ms`
- `K=4` mean fit-step time across those seeds: `3.3310ms`

Seed-`42` sweep details:

- `K=1`: RMSE `9.2845`
- `K=2`: RMSE `9.2761`
- `K=4`: RMSE `9.2930`
- `K=8`: RMSE `9.2780`
- project every `2` refreshes: RMSE `9.4618`
- project every `4` refreshes: RMSE `13.1989`

Interpretation:

- refresh-projected now has replicated evidence, not just a single promising seed
- `K=8`, with projection on every refresh, is the best current default
- on the current three-seed set it matches the replicated projected quality frontier
  to four decimals while improving mean fit time versus the earlier `K=4` setting by
  `1.56x`
- projection cadence is not a free knob; skipping projection too often damages quality

### 4.11 Controlled refresh-projected prototype

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-19-wide-controlled-refresh-projected-k8-r16-a025-density0001-seed42.json`

Measured result on the wide nonlinear benchmark:

- refresh-projected RMSE `9.2780`
- controlled refresh-projected RMSE `9.2728`
- refresh-projected fit / test / predict mean step time: `1.799ms` / `0.546ms` /
  `0.504ms`
- controlled refresh-projected fit / test / predict mean step time: `4.306ms` /
  `1.574ms` / `1.068ms`
- parameter count delta versus refresh-projected: `+14,336`

Interpretation:

- the broader architecture direction is not a dead end: a small dense control path plus
  low-bit activations can improve quality slightly on the hard task
- however, the current low-rank control path is too expensive and clearly loses on GPU
  stage cost
- this means `refresh_projected` stays the speed-oriented default, while the next
  control-path pass should focus on much cheaper capacity such as smaller rank, fewer
  controlled layers, or scalar / diagonal gates

### 4.12 Controlled refresh-projected sweep and control ablations

Decision artifacts:

- `/mnt/binary_nn/artifacts/2026-03-19-screen-controlled-refresh-k8-r2-a025-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-screen-controlled-refresh-k8-r4-a025-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-screen-controlled-refresh-k8-r2-noacts-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-screen-controlled-refresh-k8-r40-a025-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-screen-controlled-refresh-k8-r04-a025-density0001-seed42.json`
- `/mnt/binary_nn/artifacts/2026-03-19-screen-controlled-refresh-k8-scalar10-a025-density0001-seed42.json`

Main findings:

- full rank-`4` control gave the best raw seed-`42` RMSE in the screen (`9.2717`), but
  it remained too slow to become the preferred control layout
- disabling low-bit activations at rank `2` degraded RMSE to `9.5991`
- the best control tradeoff moved to first-block-only low-rank control,
  `control_ranks=(4, 0)`
- first-block-only scalar-gated control did not beat the selective low-rank variant

Why this matters:

- the control idea is real, but low-bit activations are part of it
- control should be sparse in *where* it is applied, not just small in rank
- the next control-path attempt should target fused or grouped first-block control, not
  just more naive control capacity

### 4.13 Single-GPU refresh replication

Decision artifacts:

- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42-singlegpu.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7-singlegpu.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123-singlegpu.json`

Main results:

- dense BF16 mean RMSE across the three reruns: `9.8788`
- refresh-projected mean RMSE across the three reruns: `9.2316`
- dense BF16 mean fit-step time across the three reruns: `1.6885ms`
- refresh-projected mean fit-step time across the three reruns: `2.1569ms`

Seed-specific fit comparisons:

- seed `42`: dense `2.038ms`, refresh-projected `2.012ms`
- seed `7`: dense `1.245ms`, refresh-projected `2.278ms`
- seed `123`: dense `1.782ms`, refresh-projected `2.181ms`

Interpretation:

- the quality result replicated cleanly again under the cleaner single-GPU setup
- the seed-`42` dense-level fit result is promising, but it did not replicate across the
  other two seeds
- `refresh_projected` remains the best training-rule candidate, but a clean speed win
  over dense still needs more work

### 4.14 Cycle-aligned refresh benchmark fix

Code change:

- the fit-stage benchmark path was corrected so cyclic refresh models now align to a
  refresh boundary before timing
- fit-stage timed windows now expand to cover at least two full refresh cycles

Why this was necessary:

- the original fit benchmark cloned refresh models with their internal refresh counters
  wherever training happened to stop
- that made short timing windows phase-sensitive and could over- or under-sample refresh
  work depending on alignment

Outcome:

- future GPU fit-stage claims for refresh-style models should use the corrected benchmark
- older refresh fit numbers remain useful as rough history, but not as the best current
  decision-grade evidence

### 4.15 Corrected single-GPU refresh `K` sweep

Decision artifacts:

- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed42-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed7-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed123-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k32-density0001-seed42-singlegpu-cyclebench.json`

Main corrected results:

- corrected `K=8` mean RMSE across seeds `42`, `7`, and `123`: `9.2316`
- corrected `K=16` mean RMSE across the same seeds: `9.2299`
- corrected `K=8` mean fit-step time across those seeds: `3.0086ms`
- corrected `K=16` mean fit-step time across those seeds: `2.7552ms`
- dense BF16 mean fit-step time across the corrected `K=16` runs: `2.0283ms`

Seed-specific corrected `K=16` fit comparisons:

- seed `42`: dense `1.560ms`, refresh `2.991ms`
- seed `7`: dense `2.194ms`, refresh `1.726ms`
- seed `123`: dense `2.332ms`, refresh `3.548ms`

Additional corrected large-`K` check:

- seed `42`, corrected `K=32`: RMSE `9.2772`, fit-step `3.984ms`

Interpretation:

- the benchmark fix promoted `K=16` over `K=8` as the best fit-oriented refresh interval
  tested so far
- the quality result stayed effectively unchanged, which is important
- the speed story improved relative to corrected `K=8`, but still did not cross dense
  BF16 on average across the three-seed set
- corrected `K=32` regressed again, so the refresh interval should not just keep growing

### 4.16 Initial layer-selective refresh screen

Code change:

- per-layer refresh intervals are now threaded through `refresh_projected` and
  `controlled_refresh_projected`
- this enables schedules where different hidden blocks refresh at different optimizer
  intervals

Decision artifacts:

- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-r168-density0001-seed42-singlegpu-cyclebench.json`
- `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-r328-density0001-seed42-singlegpu-cyclebench.json`

Main seed-`42` results:

- selective `(16,8)`: RMSE `9.2740`, fit-step `3.648ms`
- selective `(32,8)`: RMSE `9.2727`, fit-step `3.703ms`
- uniform corrected `K=16`: RMSE `9.2740`, fit-step `2.991ms`

Interpretation:

- naive two-layer selective refresh did not beat uniform `K=16`
- the idea is now implemented and testable, but the first concrete schedules are not a
  promising speed direction
- future refresh work should likely pivot to operator-level improvements or a more
  principled selective rule rather than more hand-picked two-layer schedules

### 4.17 Layer-level operator profiling and projection isolation

Code change:

- a reusable training-op profiler now exists in
  `src/benchmark_refresh_projected_training_ops.py`
- it measures dense, refresh non-refresh, and refresh refresh-step training costs
- it reports forward / backward / optimizer / post-step timing per mode
- it now also reports mean-gap-versus-dense and amortized refresh-hook cost, and it
  accepts `--refresh-target-density none` to isolate projection cost directly

Decision artifacts:

- `/mnt/binary_nn/artifacts/2026-03-19-refresh-training-ops-k16.json`
- `/mnt/binary_nn/artifacts/2026-03-19-refresh-training-ops-k16-summary.json`
- `/mnt/binary_nn/artifacts/2026-03-19-refresh-projection-isolation.json`
- `/mnt/binary_nn/artifacts/2026-03-19-refresh-projection-pruner-microbench.json`

Main same-GPU `K=16` layer results from the first profiler run:

- shape `(128,256,256)`:
  - dense `0.6724ms`
  - refresh non-refresh `0.7647ms`
  - refresh refresh-step `1.3465ms`
  - mean gap versus dense `0.1287ms` =
    `0.0923ms` surrogate + `0.0364ms` amortized refresh hook
- shape `(128,256,128)`:
  - dense `0.6631ms`
  - refresh non-refresh `0.8534ms`
  - refresh refresh-step `1.4079ms`
  - mean gap versus dense `0.2249ms` =
    `0.1903ms` surrogate + `0.0347ms` amortized refresh hook

Projection isolation result:

- disabling density projection reduced refresh-step post cost by `0.3382ms` on
  `(128,256,256)` and `0.3267ms` on `(128,256,128)`
- at `K=16`, that only corresponds to about `0.0211ms` and `0.0204ms` mean step cost
  per hidden layer

Pruner microbenchmark result:

- on `(256,256)`, exact `topk` pruning took `0.2715ms` while the `kthvalue` replacement
  took `0.4930ms` (`1.82x` slower)
- on `(128,256)`, exact `topk` pruning took `0.2728ms` while the `kthvalue` replacement
  took `0.3699ms` (`1.36x` slower)

Interpretation:

- the refresh hook is the dominant spike on refresh steps, but that is not the same as
  the dominant mean-step bottleneck
- under the current `K=16` operating point, the always-on surrogate / non-refresh path
  now looks like the first operator target to attack
- projection-hook work remains real, but it looks more like a secondary target unless
  the interval is reduced or refresh latency itself becomes the product constraint
- a naive `kthvalue` rewrite is not a shortcut to a faster projection rule; the current
  exact `topk` path should stay in place until a better measured alternative exists

## 5. Main Conclusions For The New Goal

The ternary workstream now supports nine strong conclusions.

1. **Projected / STE ternary is the best current quality anchor.**
   The best replicated low-bit point on the wide benchmark is now projected `0.001`.

2. **Shadow-free ternary is the best local source of optimization novelty.**
   It is not yet the harder-task winner, but it is the best evidence that directly
   maintained discrete state can matter.

3. **Density itself is no longer the main research bottleneck.**
   The projected frontier moved much lower than expected without producing a robust speed
   win.

4. **The main blocker is now GPU-side training and inference efficiency.**
   Quality is already strong enough that the next meaningful contribution must attack
   training cost or operator design.

5. **Refresh-projected remains the leading GPU-first training prototype.**
   Under the corrected cycle-aligned fit benchmark, `K=16` is now the best fit-oriented
   refresh interval tested so far.

6. **The new control-path architecture is promising for quality, but not yet for speed.**
   The best current variant is first-block-only low-rank control, and even that remains
   slower than `refresh_projected` on GPU stages.

7. **The clean seed-`42` fit-time edge for refresh-projected did not replicate yet.**
   Single-GPU reruns across seeds preserved the quality result but still left mean
   fit-step time slower than dense BF16.

8. **Naive layer-selective refresh schedules are not a free win.**
   The first `(16,8)` and `(32,8)` seed-`42` screens both lost to uniform `K=16` on the
   fit-stage tradeoff.

9. **Operator-level profiling now points at the always-on surrogate path as the first
   speed target.**
   The refresh hook is a large spike on refresh steps, but once that spike is amortized
   across `K=16`, the mean gap versus dense is still more dominated by the non-refresh
   surrogate path, and a naive `kthvalue` pruner rewrite is slower than the current
   exact `topk` rule.

## 6. Best Decision-Grade Artifacts

- shadow-free proof of concept:
  - `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`
- wide STE reference:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-followup.json`
- projected frontier:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed7.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed123.json`
- refresh-projected frontier:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123.json`
- single-GPU refresh replications:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42-singlegpu.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7-singlegpu.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123-singlegpu.json`
- corrected cycle-aligned refresh sweep:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed42-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed7-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k8-density0001-seed123-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed42-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed7-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k16-density0001-seed123-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-k32-density0001-seed42-singlegpu-cyclebench.json`
- operator profiling and projection isolation:
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-training-ops-k16.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-training-ops-k16-summary.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-projection-isolation.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-refresh-projection-pruner-microbench.json`
- initial layer-selective refresh screens:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-r168-density0001-seed42-singlegpu-cyclebench.json`
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-refresh-projected-r328-density0001-seed42-singlegpu-cyclebench.json`
- controlled refresh-projected probe:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-controlled-refresh-projected-k8-r16-a025-density0001-seed42.json`
- best selective controlled-refresh candidate:
  - `/mnt/binary_nn/artifacts/2026-03-19-wide-controlled-refresh-projected-k8-r40-a025-density0001-seed42-singlegpu.json`
- supporting ultra-low seed-`42` checks:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0005-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density00025-seed42.json`

## 7. Recommended Next Experiments

1. treat corrected `refresh_projected K=16, every1, density0.001` as the current
   fit-oriented prototype
2. if extending the control branch, start from first-block-only low-rank control
3. keep low-bit activations on by default in that branch
4. require the corrected cycle-aligned GPU fit benchmark before claiming a speed win
5. do not spend more time on naive hand-picked selective refresh schedules unless a
   stronger rule or systems hypothesis appears
6. use `src/benchmark_refresh_projected_training_ops.py` before changing kernels so the
   next optimization target is chosen from measured operator cost
7. prioritize cheaper non-refresh surrogate forward/backward work or representation
   changes before projection-pruner rewrites
8. revisit projection-hook micro-optimizations or GPU inference operators after that
   first operator target is clearer
9. revisit pure shadow-free or new structured variants only if the projected-refresh
   path and lighter control variants both plateau
