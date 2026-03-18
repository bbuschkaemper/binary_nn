# GPU-First Low-Bit Training Idea

Last updated: 2026-03-18

This document defines the current working research hypothesis for the repository.

## Document Role

Use this file when you need to understand the main *new* idea we want to test next.
It is a hypothesis, not a validated result.

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
