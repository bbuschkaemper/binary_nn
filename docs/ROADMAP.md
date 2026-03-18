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
- treat the `projected` handoff family as the current best bridge between quality
  and sparsity on that task, with the replicated wide frontier now at density `0.05`
  and a lower-density single-seed candidate already at `0.02`
- on the wider `256`-feature nonlinear benchmark, use the tuned dense BF16
  reference (`75` epochs, `3e-4`) rather than the old under-tuned pilot

### 2.2 Turn CPU inference gains into a real speed win

- the current shadow-free CPU win still comes from extreme sparsification plus a
  sparse execution path, not from a general packed ternary kernel
- the projected frontier improved from density `0.35` to a replicated `0.05`
  point without giving back the wide-benchmark quality win, and a single-seed
  `0.02` follow-up still beat dense while showing the first meaningful quality bend
- forced sparse CSR is still slower than cached dense inference at `0.20`, `0.15`,
  `0.10`, `0.05`, and `0.02` on the cleaner runs
- the newer density sweeps strengthen the model-side story, but still do not create
  a robust new CPU speed claim
- the first genuinely packed lookup prototype was strongly negative at both density
  `0.35` and `0.20`; do not treat lookup-packed execution as the likely production
  direction anymore
- the biggest systems improvement so far still comes from caching exact dense ternary
  weights in eval mode; that remains the CPU baseline new systems work must beat
- keep measuring dense, cached-dense ternary, and any new structured or sparse path
  separately so the source of any win stays visible

### 2.3 Revisit GPU training speed honestly

- current ternary work still does not show a robust training-speed advantage on GPU
- one wide projected comparison run finished faster than its paired dense run, but
  standalone dense timing varied a lot, so this is not yet a claim
- if GPU speed becomes the next focus, prioritize:
  - repeated runs and step-time measurement, not one-off wall-clock totals
  - fused evidence accumulation for `ShadowFreeTernaryLinear`
  - structured updates that reduce memory traffic
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
- density-projected handoff reached RMSE `18.4060` at density `0.35` on the
  small nonlinear benchmark, `9.3466` at density `0.35` on the wide benchmark,
  `9.4578` in the replicated `0.05` regime, and `9.7952` at a lower-density
  single-seed `0.02` probe
- on the wide benchmark, projected now beats the tuned dense reference and the wide
  STE point while also moving materially lower in density
- the remaining blocker is still the systems side: cached dense ternary CPU
  inference is the best projected execution path so far, while sparse CSR and the
  lookup-packed prototype both remain slower

## 5. Suggested First Milestone

If work resumes now, the first milestone should be:

1. treat replicated projected `0.05` as the stable ternary frontier on the wide
   nonlinear benchmark
2. treat projected `0.02` as an exploratory bend-check, not yet as the stable
   default frontier
3. keep cached dense ternary inference as the CPU default for both points
4. if model-side work continues, first decide whether `0.02` is worth replicating or
   whether `0.05` is the better stopping point for systems work
5. bias future systems work toward structured sparsity or truly different packed
   kernels, not lookup-packed prototypes

## 6. Recommended Execution Order

If work resumes now, the recommended order is:

1. keep the binary path intact as the stable published baseline
2. treat the nonlinear residual benchmark as the accepted ternary quality gate
3. treat the wide `256`-feature projected `0.05` handoff result as the current
   replicated ternary frontier to beat, while treating `0.02` as the next bend-check
   candidate
4. treat cached dense projected inference as the CPU baseline that any new sparse
   kernel must beat
5. replicate runtime on that wide benchmark so the GPU speed story is either real
   or ruled out cleanly
6. pursue lower-density or structured-sparsity projected variants and genuinely
   different packed kernels only if they can outperform cached dense inference

This sequence keeps the repo scientifically clean:

- first make the ternary claim robust on a meaningful task
- then compare new systems work against the true best current CPU path
- only then claim anything about training efficiency

## 7. Decision Note For Future Sessions

If a future session begins without user input on the next ternary choice, default
to:

- keeping the binary baseline untouched
- treating the shadow-free linear result as a real but narrow proof of concept
- treating the nonlinear residual benchmark as the primary ternary quality gate
- treating the density-projected handoff as the current best sparse-friendly
  nonlinear baseline, especially the replicated wide `0.05` point on the
  `256`-feature benchmark
- treating the wide `0.02` point as the first meaningful bend-check candidate rather
  than as an already-promoted frontier
- treating cached dense projected inference, not forced sparse CSR or lookup-packed
  execution, as the current default CPU path on that frontier
- avoiding strong claims about GPU training speed until repeated timing or
  step-time evidence exists
