# Roadmap

Last updated: 2026-03-19

This document records the current next-step plan and the open design decisions.
It is meant to be actionable rather than historical.

## Document Role

Read this after `CURRENT_STATUS.md`. It tells you what should happen next under the
new GPU-first objective.

## 1. New North Star

The repository goal is now:

- discover a new low-bit training idea that improves the GPU quality/speed frontier
- make that idea relevant to both training and inference
- keep comparisons honest against tuned dense BF16 baselines

A useful result now needs to move at least one of these in the right direction
without breaking the others:

- quality
- GPU training step time / total runtime
- GPU inference friendliness
- implementation simplicity and repeatability

## 2. What Was Just Learned

The latest work established seventeen important facts.

1. Projected ternary is the strongest current quality family and now replicates on the
   wide nonlinear benchmark all the way down to density `0.001`.
2. Shadow-free ternary is still the best local proof that a low-bit model can beat a
   dense BF16 baseline on both quality and inference speed, but only on the easy
   benchmark.
3. Lower density alone is no longer the main blocker. The projected frontier already
   moved far lower than expected without solving GPU speed.
4. The next serious problem is not “can low-bit work?” It is “which training idea can
   keep the projected-quality regime while reducing GPU cost and supporting future
   inference kernels?”
5. `refresh_projected` is now replicated across seeds, not just a single-seed
   prototype.
6. `K=8`, projection every refresh, and density `0.001` matched the replicated
   projected mean RMSE (`9.2311`) across seeds `42`, `7`, and `123`.
7. Projection cadence is sensitive: projecting every `2` refreshes already hurt
   quality (`9.4618` RMSE on seed `42`), and every `4` refreshes collapsed the run
   (`13.1989` RMSE).
8. A first `controlled_refresh_projected` prototype with a low-rank dense control path
   and ternary hidden activations slightly improved the seed-`42` RMSE (`9.2728`
   versus `9.2780` for `refresh_projected K=8`), but it was materially slower on all
   measured GPU stages.
9. Low-bit activations matter inside the control architecture: disabling them in a rank-2
   control run degraded RMSE to `9.5991`.
10. The best control sweep point so far is first-block-only low-rank control,
    `control_ranks=(4, 0)`. It kept seed-`42` RMSE at `9.2744` but remained slower than
    `refresh_projected` on all measured GPU stages.
11. Clean single-GPU refresh comparisons across seeds `42`, `7`, and `123` preserved the
    quality result (mean RMSE `9.2316`), but mean fit-step time remained slower than
    dense BF16 (`2.1569ms` versus `1.6885ms`). The seed-`42` fit-speed edge over dense
    was therefore promising, but not replicated.
12. The fit-stage GPU benchmark path was then fixed for cyclic refresh models: fit
    benchmarks now align to a refresh boundary and expand the timed window to cover at
    least two full refresh cycles.
13. Under that corrected benchmark, `K=16` slightly beat corrected `K=8` on both mean
    RMSE (`9.2299` versus `9.2316`) and mean fit-step time (`2.7552ms` versus
    `3.0086ms`) across seeds `42`, `7`, and `123`, but still remained slower than dense
    BF16 on average (`2.0283ms` dense fit-step mean on the `K=16` runs).
14. Per-layer refresh intervals are now implemented, but the first naive selective
    schedules on seed `42` — `(16,8)` and `(32,8)` — were both slower than uniform
    `K=16` while preserving only roughly the same quality.
15. A validated layer-level training-op profiler now exists in
    `src/benchmark_refresh_projected_training_ops.py`, including an option to disable
    density projection directly via `--refresh-target-density none`.
16. On the first same-GPU `K=16` operator decomposition, the refresh hook was the
    biggest spike on refresh steps, but the mean gap versus dense was still more
    dominated by the always-on surrogate / non-refresh path once the hook was
    amortized across the interval.
17. Disabling density projection removed about `0.33ms` of refresh-step post cost on the
    representative hidden-layer shapes, but that only corresponds to about `0.02ms`
    mean step cost per layer at `K=16`, and a direct microbenchmark showed the current
    exact `topk` pruner already beats a `kthvalue` replacement.

## 3. Immediate Next Steps

### 3.1 Use the new decision-grade GPU measurement path

The ternary comparison flow now has a repeated GPU-side benchmark path for CUDA
runs.

It records:

- mean step time and step-time variance
- separate `fit`, `test`, and `predict` stage summaries
- peak GPU memory
- nested stage data inside each run's `runtime.stage_benchmarks`

The next immediate use of this path should be:

- rerun the frozen dense / STE / projected benchmark trio on the wide nonlinear task
- make any future speed claim against those repeated stage metrics first
- treat one-off wall-clock totals as supporting context, not the primary evidence

### 3.2 Freeze the current quality anchor

The default quality test should now be:

- the wide nonlinear benchmark
- tuned dense BF16 baseline
- projected / STE ternary reference runs as the low-bit anchor

The main baseline set should be:

- dense BF16
- STE ternary
- projected ternary at the current strong operating point

### 3.3 Use corrected `K=16` refresh-projected as the fit-oriented baseline

The first corrected replication pass is now complete.

Immediate next experiments:

- treat corrected `K=16`, projection every refresh, and density `0.001` as the current
  fit-oriented `refresh_projected` baseline
- keep using it as the fairness anchor for any broader architectural variant
- require multi-seed repeated GPU benchmarks before claiming any speed edge over dense
- continue comparing against the frozen projected baseline using the same GPU stage
  metrics
- treat naive layer-selective refresh schedules as provisionally negative unless a new
  hypothesis justifies them
- use `src/benchmark_refresh_projected_training_ops.py` to evaluate non-refresh
  surrogate-path changes before spending time on more selective schedules or naive
  projection-pruner rewrites

Why this remains the right baseline:

- projected-family quality
- shadow-free-style discrete-state motivation
- a path toward fewer GPU-side representation changes and better kernel alignment
- it currently dominates corrected `K=8` on the fit-stage tradeoff without losing
  quality

### 3.4 Keep the control branch, but only in its best current form

The first broader architecture pass is now implemented:

- refresh-projected low-bit bulk
- low-rank dense control path
- ternary hidden activations

What the first result says:

- the architecture can help quality a little
- the current dense control path is too expensive to be the answer as-is
- low-bit activations appear to be necessary inside that branch
- first-block-only low-rank control is the current best control layout
- scalar-gated control did not beat selective low-rank control in the first screen

Immediate follow-ups:

- start from `control_ranks=(4, 0)` rather than full control
- keep low-bit activations enabled by default in this branch
- if control work continues, focus on grouped or fused first-block control rather than
  larger low-rank adapters
- only revisit scalar / diagonal gating if there is a systems reason it can be fused
  much more cheaply than the current low-rank path

### 3.5 Keep activation-side low-bit experiments, but under tighter scope

The repo currently focuses mostly on low-bit weights.

The next training-idea pass should still treat activation precision as a first-class
knob, but now through the lighter control-path variants:

- start with easy-to-toggle activation quantization in forward
- keep evaluation fair against the same dense reference
- use the current controlled-refresh family as the main place to test activation ideas
- avoid growing the dense control branch at the same time
- treat "activation quantization off" as mostly negative unless a new architecture
  overturns the current ablation

### 3.6 Prepare a GPU inference path that matches the training representation

GPU inference should not be treated as a separate late-stage afterthought.

Instead:

- reuse the binary Triton experience as operator-design groundwork
- benchmark operator-level ternary paths before claiming end-to-end wins
- prefer training representations that future kernels would actually want to run

## 4. Open Design Decision

There are three plausible paths forward.

### 4.1 Option A: projected-refresh first (recommended)

Keep projected / STE ternary as the quality anchor and make that training path cheaper.

Pros:

- starts from the strongest current quality regime
- lowest scientific risk
- easiest to compare fairly against dense BF16
- most plausible route to a training-speed win that still matters for inference

Cons:

- less novel than a fully direct-discrete method
- may still keep some floating-point latent state in the near term

### 4.2 Option B: shadow-free first

Push the direct-discrete branch harder and try to make it stable on the wide nonlinear
benchmark.

Pros:

- most novel optimization idea in the repo
- strongest long-term fit with “true low-bit training”
- naturally attractive for stable discrete-state kernels if it works

Cons:

- highest scientific risk
- currently unstable on the harder benchmark
- not the best immediate route to a believable GPU speed story

### 4.3 Option C: kernel-first systems push

Focus mainly on GPU operators and inference kernels before changing the training rule.

Pros:

- could produce concrete systems artifacts quickly
- reuses existing binary Triton groundwork

Cons:

- risks optimizing a representation that is not yet the best training path
- could produce systems wins without solving the core training objective

## 5. Recommended Direction

The recommended direction is:

1. preserve the current projected / STE references
2. use decision-grade GPU timing and memory measurement
3. use `refresh_projected K=8` as the new default GPU-first prototype
4. treat corrected `refresh_projected K=16` as the current fit-oriented refresh baseline
5. treat first-block-only low-rank control as the best current quality-oriented
   architecture extension, not the new default
6. compare new variants against dense BF16 and the current projected baseline
7. use the operator profiler to attack the always-on surrogate path first, and treat
   projection-hook work as a secondary target at the current `K=16` operating point
8. only then push layer-selective refresh or GPU inference kernels for the winning
   variant

## 6. Suggested First Milestone

The first milestone should be one decision-grade experiment on the wide nonlinear
benchmark where a new low-bit training variant:

- keeps RMSE in the projected / STE quality regime
- improves GPU step time or end-to-end runtime relative to the current projected path
- produces a stable discrete state that a future GPU inference kernel could reuse

## 7. Recommended Execution Order

1. use corrected `refresh_projected K=16, every1, density0.001` as the fit-oriented
   training baseline
2. lock the wide benchmark config and tuned dense BF16 / projected references
3. if exploring architecture, start from first-block-only low-rank control
4. require multi-seed repeated GPU benchmarks before claiming any speed win
5. do not keep pushing `K` upward blindly; corrected `K=32` already regressed on seed `42`
6. do not keep spending time on naive two-layer selective refresh schedules; the first
   `(16,8)` and `(32,8)` screens were negative
7. use `src/benchmark_refresh_projected_training_ops.py` to decompose the refresh path
   before changing kernels or training rules
8. prioritize cheaper non-refresh surrogate forward/backward work or representation
   changes before projection-pruner rewrites
9. revisit projection-hook micro-optimizations or GPU inference operators only after
   that first operator target is clearer
10. revisit pure shadow-free or structured variants only if the projected-refresh path
   and the leanest control variants both plateau

## 8. Decision Note For Future Sessions

If there is only enough time for one substantial change, choose training-rule and
GPU-measurement work over more CPU sparsity work.
