# Roadmap

Last updated: 2026-03-18

This document records the current next-step plan and the open design decisions.
It is meant to be actionable rather than historical.

## Document Role

Use this file after `CURRENT_STATUS.md`. It tells you what should happen next
and what decisions still need to be made.

## 1. What Was Just Finished

The latest completed workstream delivered:

- a direct-discrete `ShadowFreeTernaryLinear` prototype
- an STE `TernaryLinear` baseline beside it
- an STE-to-shadow-free handoff path with optional target-density projection
- sparse CPU inference caches for both ternary families
- BF16 support in the shared training path
- a nonlinear residual benchmark mode for harder sanity checks
- a new ternary comparison entry point with artifact export
- one decision-grade shadow-free proof of concept on the original linear task
- one decision-grade STE follow-up on the harder nonlinear task

That means the repo is no longer binary-only. It now has a real ternary branch
with one positive result and one meaningful constraint.

## 2. Immediate Next Steps

These are the highest-priority next tasks if work resumes without changing the
overall direction.

### 2.0 Preserve the current decision-grade ternary artifacts

- keep `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json` as the current
  shadow-free reference
- keep `/mnt/binary_nn/artifacts/2026-03-18-ste-nonlinear-followup.json` as the
  current harder-task ternary reference
- do not overwrite those names casually; future iterations should use new dated
  artifact names

### 2.1 Improve the harder-task ternary story

- treat `target_kind="nonlinear_residual"` as the default ternary stress test
- reject new ternary claims if they only work on the easy linear benchmark and
  collapse to a near-zero-density residual branch
- focus on closing the quality gap while lowering density on the nonlinear task
- treat the new `projected` handoff family as the current best bridge between
  quality and sparsity on that task

### 2.2 Turn sparse CPU wins into a more hardware-aligned path

- the current shadow-free CPU win uses cached sparse execution, not a packed
  ternary kernel
- the new `projected` handoff reaches density `0.35` on the harder nonlinear
  task, but sparse CPU inference is still slower than dense there
- the next systems step is to move from unstructured sparse tensor execution to
  block-sparse or bit-packed ternary kernels
- keep measuring dense versus sparse variants separately so the source of any win
  stays visible

### 2.3 Revisit GPU training speed honestly

- current ternary work does not yet show a training-speed advantage on GPU
- if GPU speed becomes the next focus, prioritize:
  - fused evidence accumulation for `ShadowFreeTernaryLinear`
  - structured updates that reduce memory traffic
  - benchmarking step time directly instead of only total runtime
- avoid claiming pretraining wins until that path is actually measured

### 2.4 Tune wide dense BF16 references before drawing wider conclusions

- the wide `256`-feature pilot showed that a dense BF16 MLP can look worse than
  the ternary path if it is under-tuned
- future wide comparisons must retune the dense baseline before using those runs
  as evidence

## 3. Open Design Decision

The main open decision now is how to push the ternary branch forward.

There are three plausible paths.

### 3.1 Option A: Shadow-free first

Keep pushing the direct-discrete branch immediately.

Pros:

- most novel optimization idea in the repo
- strongest long-term fit with the repo's original research direction
- naturally creates sparse states that are attractive for CPU inference

Cons:

- currently works best only on the easy linear benchmark
- degrades badly on the harder nonlinear task
- still lacks a GPU training-speed win

### 3.2 Option B: STE ternary first

Use the STE ternary branch as the main ternary baseline and delay direct-discrete
optimization work.

Pros:

- better quality behavior on the nonlinear benchmark
- simpler to tune and reason about
- closer to a practical BitNet-like training baseline

Cons:

- less novel
- currently too dense to win on CPU sparse inference
- risks turning into another quantization-aware branch without solving the
  discrete-optimization problem

### 3.3 Option C: Hybrid bootstrap path

Use STE ternary as a warm-start or quality anchor, then convert or consolidate
into a shadow-free sparse state.

Pros:

- matches the evidence now in hand
- keeps the quality-friendly branch and the sparse/direct branch connected
- offers the best chance of getting both quality and CPU efficiency

Cons:

- more engineering than committing to one branch immediately
- requires careful experiment design to avoid mixing two effects
- naive free-running consolidation is already known to be too destructive on the
  harder nonlinear task

## 4. Recommended Direction

Current recommendation: **Option C, but specifically the projected handoff
variant rather than free-running consolidation**.

Reasoning:

- the shadow-free branch already proves that direct-discrete sparse CPU wins are
  plausible
- the STE branch already shows that ternary quality on the harder nonlinear
  benchmark can get close to dense
- naive direct-discrete consolidation reached useful sparsity but collapsed to
  RMSE `25.8914` on the nonlinear task
- density-projected handoff reached RMSE `18.4060` at density `0.35`, which is
  the best current quality at a sparse-ready density
- the remaining blocker is now either lower density without another quality cliff
  or a better sparse/packed ternary inference kernel

## 5. Suggested First Milestone

If work resumes now, the first milestone should be:

1. keep the new `projected` handoff as the harder-task sparse baseline
2. sweep projection densities and recovery schedules around the current `0.35`
   operating point
3. compare:
   - dense BF16 baseline
   - STE ternary
   - free-running hybrid consolidation
   - density-projected handoff
4. only then decide whether the next effort should go into lower-density model
   schedules or a packed ternary CPU kernel

## 6. Recommended Execution Order

If work resumes now, the recommended order is:

1. keep the binary path intact as the stable published baseline
2. improve the nonlinear residual benchmark until it is the accepted ternary
   quality gate
3. treat the density-projected STE-to-shadow-free handoff as the current handoff
   baseline
4. add lower-density or structured-sparsity controls so CPU speed can improve
   without another accuracy cliff
5. benchmark CPU latency again with the model-side density held fixed
6. only then spend effort on custom CUDA update kernels or a packed ternary CPU
   kernel

This sequence keeps the repo scientifically clean:

- first make the ternary claim robust on a meaningful task
- then improve the systems path
- only then claim anything about training efficiency

## 7. Decision Note For Future Sessions

If a future session begins without user input on the next ternary choice, default
to:

- keeping the binary baseline untouched
- treating the shadow-free linear result as a real but narrow proof of concept
- treating the nonlinear residual benchmark as the primary ternary quality gate
- treating the density-projected handoff as the current best sparse-friendly
  nonlinear baseline
- avoiding strong claims about GPU training speed until step-time evidence exists
