# GPU-First Low-Bit Training Idea

Last updated: 2026-03-19

This document defines the current working research idea for the repository.

## Document Role

Use this file when you need to understand the main *new* idea we want to test next.
It now has an initial replicated result, but it is still a research direction rather
than a settled conclusion.

## 1. One-Sentence Summary

Train a projected ternary model using a cached discrete weight state that is refreshed
only every `K` steps, so the model keeps projected-family quality while amortizing
quantization / projection work and moving closer to a kernel-friendly GPU
representation.

## 2. Why This Is The Right Direction Now

The current local evidence points to a very specific gap.

- projected / STE ternary is the strongest quality family on the hard benchmark
- shadow-free ternary is the most interesting optimization idea, but it is not yet
  stable enough on the hard benchmark to become the main path
- CPU density chasing already moved to projected `0.001`, so lower density alone is
  clearly not the main blocker anymore
- the next useful gain has to come from a training rule or operator design that makes
  low-bit training cheaper on GPU while preserving projected-like quality

## 3. Core Proposal: Refresh-Scheduled Projected Ternary Training

Working name:

- **refresh-scheduled projected ternary training**

Abbreviation used locally in planning:

- **RSPT**

The idea is to combine three things that the repo already learned separately:

1. projected / STE training preserves quality best
2. shadow-free training suggests that stable discrete state matters
3. kernels want a representation that does not change format every step

## 4. Proposed State Representation

The first version should keep the representation simple.

State:

- latent weights `W` in BF16 or FP32
- cached ternary state `Q in {-1, 0, +1}`
- per-row or per-channel scale `s`
- optional accumulated evidence buffer `A` for refresh-time decisions

Interpretation:

- `W` is the optimizer-facing state
- `Q` is the forward-facing discrete state
- `A` is optional support for more shadow-free-like refresh updates

## 5. Training Loop

The first experimental loop should be:

1. start from the current projected / STE recipe or warm start
2. build the discrete state `Q` from `W`
3. keep `Q` fixed for `K` optimizer steps
4. run forward and backward through the cached `Q`
5. update `W` every step, but do not rebuild `Q` every step
6. on refresh boundaries, rebuild or re-project `Q`
7. optionally use the evidence buffer `A` to bias refresh-time add / drop decisions

This is intentionally conservative. The first version should try to keep projected-like
quality, not maximize novelty.

## 6. Why It Might Improve GPU Training

Potential benefits:

- quantization and projection overhead are amortized across multiple steps
- the forward representation becomes more stable and therefore more kernel-friendly
- representation churn is reduced
- the experiment becomes easier to instrument at the operator level
- it moves training closer to the representation that inference would later use

The goal is not only fewer bits in storage. The real target is lower GPU-side work per
useful unit of training progress.

## 7. Why It Might Improve GPU Inference Later

If the training loop already spends most of its time using a stable ternary state, then
a future GPU inference kernel can target the same representation directly.

That matters because the current repo evidence strongly suggests:

- speed wins do not appear automatically from lower precision alone
- representation and kernel design have to be aligned from the start

## 8. The First Knobs To Sweep

Start with a small, interpretable set of ablations.

- refresh interval `K`
- whether projection happens on every refresh or only some refreshes
- density schedule across training
- whether refresh uses only latent `W` or also the evidence buffer `A`
- whether activations remain BF16 or are lightly quantized in forward
- whether all layers participate or only the largest hidden layers

## 9. Minimal First Experiment

Benchmark:

- the wide nonlinear residual benchmark

Compare:

- tuned dense BF16 baseline
- current projected ternary baseline
- first RSPT variant

Measure:

- RMSE
- total runtime
- repeated mean step time
- timing variance
- peak memory

A first-pass success condition is modest but meaningful:

- keep quality in the projected / STE regime while lowering GPU training cost
- or improve quality further without making runtime worse

## 10. Main Failure Modes

The most likely ways this can fail are:

- stale `Q` hurts optimization too much
- `W` still dominates memory and the refresh schedule saves almost nothing
- refresh events are too expensive and erase the gains
- activation-side low-bit changes destabilize the comparison
- benchmark noise hides small but real timing changes

## 11. What This Idea Is Not

This is **not**:

- a CPU-sparsity project
- a claim that direct-discrete shadow-free training already works on the hard task
- a late post-training quantization pass
- proof that strict binary is better than ternary

It is a GPU-first training idea built from the strongest current local evidence.

## 12. Why This Is Still Adjacent To BitNet

The proposal is still close to the BitNet line because:

- ternary weights remain central
- low-bit activations are treated as important, not optional forever
- kernel-aware representation design is part of the training story

The local novelty is the refresh-scheduled discrete state, which is motivated by the
combination of:

- projected-family quality
- shadow-free discrete-update intuition
- the need for a GPU-friendly representation that stays stable for long enough to matter

## 13. Bottom Line

The best next bet is not lower density or more CPU sparsity. It is a training idea that
keeps projected-family quality while making the discrete state stable enough to optimize
and stable enough for future GPU kernels to exploit.

At the current corrected `K=16` operating point, the first operator evidence says the
next speed gain probably has to come from making the always-on surrogate / non-refresh
path cheaper, not from a simple rewrite of the refresh projection step alone.

## 14. First Implemented Variant

The first implemented `refresh_projected` variant now exists in code.

What it does:

- keeps latent ternary weights as the optimizer-facing state
- caches a projected ternary forward state and scale buffers
- refreshes that cached state every `K` optimizer steps
- forces one final refresh before evaluation so testing and prediction use the latest
  discrete state
- uses a straight-through latent-weight surrogate during training so the forward path
  stays on the cached discrete state

Replication and first sweep summary at projected density `0.001`:

- `K=4` mean RMSE across seeds `42`, `7`, and `123`: `9.2377`
- `K=8` mean RMSE across seeds `42`, `7`, and `123`: `9.2311`
- projected ternary mean RMSE across those seeds: `9.2311`

Seed-`42` sweep details:

- `K=1`: `9.2845`
- `K=2`: `9.2761`
- `K=4`: `9.2930`
- `K=8`: `9.2780`
- cadence every `2` refreshes: `9.4618`
- cadence every `4` refreshes: `13.1989`

Interpretation:

- the idea is now replicated across the current three-seed benchmark set
- `K=8`, with projection on every refresh, is the best current default
- on the current three-seed set it matches the projected mean RMSE exactly to four
  decimals while reducing mean fit-step time versus the earlier `K=4` refresh setting
- projection cadence must stay frequent; projecting less often was clearly negative

## 15. Second Implemented Variant: Controlled Refresh-Projected

The first broader architecture extension now also exists in code.

What it adds on top of `refresh_projected`:

- a large refresh-projected ternary bulk path
- a small low-rank dense control path in each hidden block
- ternary hidden-activation quantization between blocks

Implemented in:

- `ControlledRefreshProjectedTernaryBlock`
- `ControlledRefreshProjectedTernaryRegressor`
- `src/run_hybrid_ternary_regression.py` via
  `consolidation_variant="controlled_refresh_projected"`
- `src/run_ternary_research_comparison.py` via
  `--model-family controlled_refresh_projected`

First wide nonlinear seed-`42` result at density `0.001`, `K=8`, projection every
refresh, control rank `16`, activation threshold scale `0.25`:

- refresh-projected RMSE: `9.2780`
- controlled refresh-projected RMSE: `9.2728`
- refresh-projected fit / test / predict mean step time: `1.799ms` / `0.546ms` /
  `0.504ms`
- controlled refresh-projected fit / test / predict mean step time: `4.306ms` /
  `1.574ms` / `1.068ms`
- parameter count delta versus refresh-projected: `+14,336`

Interpretation:

- the broader architecture direction can improve quality slightly, so the "low-bit bulk
  + dense control + low-bit activations" idea is worth keeping
- the current dense control path is too heavy to help the GPU speed story
- the next step is therefore not a bigger model-side sweep; it is to make the control
  path much cheaper while checking whether the quality gain survives

Follow-up sweep summary:

- rank-`2` full control: RMSE `9.2757`
- rank-`4` full control: RMSE `9.2717`
- rank-`2` with low-bit activations disabled: RMSE `9.5991`
- best selective low-rank point: `control_ranks=(4, 0)`
- first-block-only scalar gate did not beat the selective low-rank point

Best control-path tradeoff so far, from a clean single-GPU seed-`42` comparison:

- refresh-projected RMSE: `9.2778`
- selective low-rank controlled refresh-projected RMSE: `9.2744`
- refresh-projected fit / test / predict mean step time: `2.012ms` / `0.558ms` /
  `0.505ms`
- selective low-rank controlled refresh-projected fit / test / predict mean step time:
  `3.013ms` / `1.564ms` / `1.871ms`

Interpretation update:

- low-bit activations are part of the gain in this architecture, not a harmless toggle
- the right control question is now "can we make the first-block control path much
  cheaper or fuse it?" rather than "should we keep adding more dense correction?"
- until that happens, `refresh_projected` remains the default GPU-first path

## 16. Corrected cycle-aligned refresh benchmark

The fit-stage benchmark path was later corrected for cyclic refresh models.

What changed:

- fit benchmarks now align refresh-style models to a refresh boundary before timing
- the timed window now expands to cover at least two full refresh cycles
- this makes the benchmark slower to run, but much more trustworthy for refresh-scheduled
  models

Corrected single-GPU three-seed refresh sweep:

- corrected `K=8` mean RMSE: `9.2316`
- corrected `K=16` mean RMSE: `9.2299`
- corrected `K=8` mean fit-step time: `3.0086ms`
- corrected `K=16` mean fit-step time: `2.7552ms`
- dense BF16 mean fit-step time across the corrected `K=16` runs: `2.0283ms`

Seed-specific corrected `K=16` fit results:

- seed `42`: dense `1.560ms`, refresh `2.991ms`
- seed `7`: dense `2.194ms`, refresh `1.726ms`
- seed `123`: dense `2.332ms`, refresh `3.548ms`

Additional corrected seed-`42` large-`K` check:

- corrected `K=32` RMSE: `9.2772`
- corrected `K=32` fit-step time: `3.984ms`

Interpretation:

- the benchmark fix changed the speed story materially enough that older fit-stage
  claims should no longer be treated as decision-grade
- `K=16` is now the best fit-oriented refresh interval tested so far
- however, the family still does not beat dense BF16 on average, so the main goal is
  not yet solved

## 17. First layer-selective refresh probe

Per-layer refresh intervals are now implemented for both `refresh_projected` and
`controlled_refresh_projected`.

First corrected seed-`42` screens:

- `(16,8)`: RMSE `9.2740`, fit-step `3.648ms`
- `(32,8)`: RMSE `9.2727`, fit-step `3.703ms`
- uniform corrected `K=16`: RMSE `9.2740`, fit-step `2.991ms`

Interpretation:

- the naive selective idea did not help
- slowing only the largest hidden block while keeping the smaller block faster was not
  enough to improve the fit-stage tradeoff
- if selective refresh is revisited, it should come with a stronger hypothesis than just
  "refresh the big layer less often"

## 18. First operator-level cost decomposition

A reusable layer-level training-op profiler now exists in
`src/benchmark_refresh_projected_training_ops.py`.

What it measures:

- dense train step
- refresh non-refresh step
- refresh refresh-step
- forward / backward / optimizer / post-step timing
- estimated mean step cost at a chosen refresh interval
- optional projection-disabled runs via `--refresh-target-density none`

First same-GPU `K=16` measurements on the two wide hidden-layer shapes:

- shape `(128,256,256)`: dense `0.6724ms`, refresh non-refresh `0.7647ms`,
  refresh-step `1.3465ms`
- shape `(128,256,128)`: dense `0.6631ms`, refresh non-refresh `0.8534ms`,
  refresh-step `1.4079ms`

Decomposition:

- shape `(128,256,256)`: mean gap versus dense `0.1287ms` =
  `0.0923ms` surrogate + `0.0364ms` amortized refresh hook
- shape `(128,256,128)`: mean gap versus dense `0.2249ms` =
  `0.1903ms` surrogate + `0.0347ms` amortized refresh hook

Projection isolation:

- disabling density projection reduced refresh-step post cost by `0.3382ms` and
  `0.3267ms` on those same shapes
- at `K=16`, that corresponds to only about `0.021ms` mean step cost per hidden layer

Pruning microbenchmark:

- the current exact `topk` density pruner beat a `kthvalue` threshold replacement by
  `1.36x` to `1.82x` on representative shapes

Interpretation:

- the refresh hook is a real spike on refresh boundaries
- however, at the current `K=16` operating point the mean fit-step gap is still more
  dominated by the always-on surrogate path than by the amortized hook
- that means the next operator target should be ternary-aware non-refresh
  forward/backward work or a cheaper surrogate representation
- projection-hook work is still a worthwhile secondary target, but it no longer looks
  like the first bottleneck to attack
