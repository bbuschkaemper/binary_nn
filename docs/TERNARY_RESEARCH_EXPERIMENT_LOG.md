# Ternary Research Experiment Log

Last updated: 2026-03-18

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

### 2.4 Shared comparison tooling

Implemented in:

- `src/run_ternary_research_comparison.py`

Role:

- compare dense BF16 baseline versus one ternary family
- write JSON plus CPU benchmark CSV artifacts under `/mnt`
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

## 5. Main Conclusions For The New Goal

The ternary workstream now supports five strong conclusions.

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

5. **The next training idea should combine the projected quality anchor with a more
   stable discrete-state schedule.**
   That is why the repo is now shifting toward refresh-scheduled projected ideas.

## 6. Best Decision-Grade Artifacts

- shadow-free proof of concept:
  - `/mnt/binary_nn/artifacts/2026-03-18-shadowfree-poc.json`
- wide STE reference:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-followup.json`
- projected frontier:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed7.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0001-seed123.json`
- supporting ultra-low seed-`42` checks:
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density0005-seed42.json`
  - `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density00025-seed42.json`

## 7. Recommended Next Experiments

1. add repeated GPU timing and memory measurement to the ternary comparison flow
2. prototype a refresh-scheduled projected ternary variant
3. compare that variant against dense BF16 and the current projected baseline on the
   wide nonlinear benchmark
4. add activation-side low-bit experiments once the refresh-scheduled baseline is clear
5. benchmark operator-level GPU inference for the resulting representation
6. revisit pure shadow-free or new structured variants only if the projected-refresh
   path plateaus
