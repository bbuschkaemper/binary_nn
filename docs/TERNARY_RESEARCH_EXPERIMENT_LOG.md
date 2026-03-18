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
- one selected ternary branch (`shadowfree`, `ste`, `hybrid`, or `projected`)
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

### 4.5 Negative result: naive hybrid consolidation on the nonlinear benchmark

Command surface:

- `src/run_ternary_research_comparison.py --model-family hybrid --target-kind nonlinear_residual ...`

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-hybrid-nonlinear-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-hybrid-nonlinear-followup-cpu.csv`

Measured result:

Dense BF16 baseline:

- hidden dims `(64, 32)`
- RMSE `15.6111`
- total runtime `7.0792s`

Hybrid handoff with free-running direct-discrete consolidation:

- STE warm-start epochs `50`
- shadow-free consolidation epochs `25`
- consolidation learning rate `1e-3`
- update interval `4`
- RMSE `25.8914`
- total runtime `12.0098s`
- nonzero density `0.1105`

CPU inference result:

- sparse CPU is slower than dense at batch `32`
- sparse CPU becomes faster than dense at batch `128` and `512`
- best measured sparse speedups:
  - batch `128`: `1.17x`
  - batch `512`: `1.36x`

Interpretation:

- the direct-discrete consolidation rule can force sparsity on the harder task
- in its current form it destroys too much quality to be the default handoff
  recipe

### 4.6 Density-projected handoff follow-up on the nonlinear benchmark

Command surface:

- `src/run_ternary_research_comparison.py --model-family projected --target-kind nonlinear_residual ...`

Decision artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-projected-nonlinear-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-projected-nonlinear-followup-cpu.csv`

Measured result:

Dense BF16 baseline:

- hidden dims `(64, 32)`
- RMSE `15.6111`
- total runtime `6.5965s`

Projected handoff:

- STE warm-start epochs `50`
- target density `0.35`
- recovery epochs `25`
- recovery learning rate `3e-4`
- projected-update interval `100000` to keep ternary state fixed during recovery
- RMSE `18.4060`
- total runtime `8.2129s`
- nonzero density `0.3500`

CPU inference result:

- sparse CPU is still slower than dense at every measured batch
- the projected model is the best current quality result observed at a
  sparse-friendly density, but the current sparse kernel does not yet capitalize
  on it

Interpretation:

- density-aware projection is a better harder-task bridge than free-running
  shadow-free consolidation
- the remaining bottleneck is either lower density without another accuracy cliff
  or a better sparse/packed ternary CPU kernel


### 4.7 Negative follow-up: local projected refinements did not beat the baseline

Several targeted follow-ups were tried immediately after the first projected
result. None improved the projected `0.35` nonlinear point.

Projection-structure variants:

- per-row `0.35` projection: RMSE `19.6784`, density `0.3719`
- per-row front-loaded `(0.7, 0.25)` layer densities: RMSE `19.4911`, density
  `0.4609`
- per-row dense-first `(1.0, 0.15)` layer densities: RMSE `20.7850`, density
  `0.4469`

Calibration variants:

- teacher-calibrated projected `0.35`: RMSE `18.6087`, density `0.3500`
- teacher-calibrated projected `0.25`: RMSE `21.0718`, density `0.2500`

Systems variant:

- a naive index-based CPU kernel prototype for `ShadowFreeTernaryLinear` was
  correct but much slower than both dense execution and the current
  `torch.sparse.mm` path

Interpretation:

- the easy local tweaks around the projected branch have been tested and mostly
  ruled out
- future progress is more likely to come from better kernels, repeated wide
  validation, or more structural sparsity changes than from another small
  projection heuristic

### 4.8 Wide nonlinear follow-up with a tuned dense baseline

The earlier wide pilot was explicitly revisited with a tuned dense reference.

Benchmark surface:

- `n_features=256`
- `n_informative=256`
- `nonlinear_pair_count=64`
- `nonlinear_scale=1.5`
- hidden dims `(256, 128)`

Dense BF16 tuning sweep:

- `75` epochs, `1e-3`: RMSE `10.0910`, total `7.0151s`
- `100` epochs, `1e-3`: RMSE `10.0333`, total `16.8387s`
- `75` epochs, `3e-4`: RMSE `10.0171`, total `6.6501s`
- `100` epochs, `3e-4`: RMSE `10.0171`, total `17.0500s`

Interpretation:

- the tuned dense reference for this wide benchmark is the `75` epoch, `3e-4`
  configuration
- future wide comparisons should use that tuned reference instead of the older
  under-tuned pilot

Wide STE artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-ste-followup.json`
- RMSE `9.7710`
- density about `0.7461`

Wide projected artifact:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-followup.json`
- RMSE `9.3466`
- density about `0.3500`

CPU inference result:

- wide STE is much slower than dense in both dense and sparse CPU modes
- wide projected is still slower than dense, but much closer than wide STE
- best projected sparse-to-dense ratios still stay below `1.0x`, reaching only
  about `0.665x` at batch `2048`

Important runtime caution:

- one paired projected comparison run reported projected total runtime below its
  paired dense run
- the standalone dense tuning sweep also produced a much faster dense run
- that means the wide benchmark is a real quality win and a better systems
  frontier, but not yet a stable GPU speed claim

Interpretation:

- projected handoff is now the strongest ternary quality-versus-density point in
  the repo
- the main blocker has moved even more clearly to the systems path rather than a
  lack of quality

### 4.9 CPU systems follow-up on the wide projected frontier

A dedicated systems pass was run after the wide projected result to answer a more
specific question: at density `0.35`, what CPU execution path is actually best on
that model?

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-indexed-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-csr-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-cached-dense-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-final-followup.json`

What was tested:

1. a cached indexed-gather ternary kernel
2. a CSR sparse backend for the existing sparse path
3. cached exact dense ternary weights in eval mode

Result summary:

- the indexed-gather path was correct but catastrophically slower on the real wide
  projected model, especially at batch `512` and `2048`
- CSR made the sparse path more realistic and much easier to reason about, but a
  forced sparse path still lost clearly to dense execution at density `0.35`
- caching exact dense ternary weights in eval mode was the biggest practical CPU
  improvement and is now the best current CPU path for the projected frontier

Final forced comparison on the wide projected point:

- tuned dense BF16 baseline latency: `0.1775 / 0.2450 / 0.4441 / 1.0431 ms`
- projected cached-dense latency: `0.2498 / 0.3883 / 0.7959 / 1.1286 ms`
- projected forced sparse CSR latency: `0.6591 / 0.7706 / 1.5097 / 2.6216 ms`
- all three latency series above are ordered by batch `32 / 128 / 512 / 2048`
- projected RMSE stayed at `9.3466` with density about `0.3500`

Interpretation:

- the systems bottleneck on the projected branch was not only sparse matmul; a lot
  of cost was coming from rebuilding ternary weights every forward pass
- after removing that rebuild cost, cached dense ternary inference beats the forced
  sparse path across the full tested batch range at density `0.35`
- future sparse or packed kernels now need to beat cached dense projected
  inference, not the older reconstruct-every-forward path
- the sparse default threshold was lowered so projected `0.35` models now stay on
  cached dense inference unless sparse execution is forced explicitly

### 4.10 Lower-density and packed-kernel dual follow-up on the wide projected frontier

After the cached-dense CPU update, two remaining directions were tested directly:

1. lower the projected target density further on the wide benchmark
2. try a genuinely packed CPU prototype instead of a sparse tensor path

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density025-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density020-followup.json`
- `/mnt/binary_nn/artifacts/2026-03-18-dual-eval-lean.json`
- `/mnt/binary_nn/artifacts/2026-03-18-dual-eval-lean.csv`

Quality follow-up:

- projected `0.25`: RMSE `9.3268`, density `0.25`
- projected `0.20`: RMSE `9.3308`, density `0.20`

Interpretation:

- both lower-density points stayed in essentially the same quality regime as the old
  projected `0.35` point
- the `0.20` point is now the strongest observed projected quality-versus-density
  result in the repo because it preserves the wide quality win while lowering
  density materially

Packed-kernel evaluation surface:

- clean lean CPU timing focused on batch `128` and `2048`
- variants compared per projected model:
  - cached dense
  - forced sparse CSR
  - packed lookup with block size `4`
  - packed lookup with block size `8`

Wide projected `0.35` CPU results:

- cached dense: `0.4819 / 1.1263 ms`
- forced sparse CSR: `0.9765 / 2.3463 ms`
- packed lookup `b4`: `4.4172 / 230.4038 ms`
- packed lookup `b8`: `5.2148 / 100.0477 ms`

Wide projected `0.20` CPU results:

- cached dense: `0.3466 / 1.0618 ms`
- forced sparse CSR: `0.6607 / 2.7884 ms`
- packed lookup `b4`: `2.8692 / 219.2001 ms`
- packed lookup `b8`: `4.0883 / 80.7821 ms`

All latency pairs above are ordered by batch `128 / 2048`.

Interpretation:

- the lower-density model-side move was positive
- the packed lookup prototype was decisively negative at both densities
- sparse CSR also remained slower than cached dense even after moving the projected
  density down to `0.20`
- this means the repo now has a better model frontier, but not a better CPU kernel
  frontier
- the lean benchmark used intentionally short timing loops after the packed path
  proved expensive, so any near-crossover between dense BF16 and cached-dense
  projected execution should be treated as suggestive rather than decision-grade

### 4.11 Replication of projected `0.20` and a lower-density `0.15` follow-up

The new projected `0.20` point was then tested across additional seeds before
pushing density lower again.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density020-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density020-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density015-seed42.json`

Replication results for projected `0.20`:

- seed `7`: dense RMSE `9.9809`, projected RMSE `9.4234`, density `0.20`
- seed `123`: dense RMSE `9.6383`, projected RMSE `9.0070`, density `0.20`

Interpretation:

- projected `0.20` is not a one-seed artifact
- across both new seeds it still beat the paired dense BF16 baseline by a useful
  margin
- this upgrades projected `0.20` from a promising point to the current best
  **replicated** projected frontier in the repo

Lower-density follow-up at projected `0.15`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.3295`, density `0.15`
- cached dense projected latency: `0.3695 / 1.0463 ms` at batch `128 / 2048`
- forced sparse CSR latency: `0.7057 / 2.0630 ms` at batch `128 / 2048`

Interpretation:

- the projected frontier likely moves at least as low as `0.15` on the wide
  benchmark without losing the quality regime
- however, `0.15` is still only a single-seed result and should not replace the
  replicated `0.20` point as the default recommendation yet
- sparse CSR still loses to cached dense inference even at `0.15`

### 4.12 Replication of projected `0.15` and a lower-density `0.10` follow-up

After projected `0.15` looked strong on seed `42`, that point was replicated on the
same additional seeds already used for the projected `0.20` check.

Artifacts:

- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density015-seed7.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density015-seed123.json`
- `/mnt/binary_nn/artifacts/2026-03-18-wide-projected-density010-seed42.json`

Replication results for projected `0.15`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.3295`, density `0.15`
- seed `7`: dense RMSE `9.9809`, projected RMSE `9.4330`, density `0.15`
- seed `123`: dense RMSE `9.6383`, projected RMSE `9.0266`, density `0.15`

Interpretation:

- projected `0.15` is not a one-seed artifact
- across all three checked seeds it stayed in essentially the same quality regime as
  the earlier projected `0.20` runs
- this upgrades projected `0.15` to the current best **replicated** projected
  frontier in the repo

Lower-density follow-up at projected `0.10`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.3453`, density `0.10`
- cached dense projected latency: `0.3495 / 1.0411 ms` at batch `128 / 2048`
- forced sparse CSR latency: `0.5716 / 1.7894 ms` at batch `128 / 2048`

Interpretation:

- the projected frontier likely moves at least as low as `0.10` on the wide
  benchmark without losing the quality regime
- however, `0.10` is still only a single-seed result and should not replace the
  replicated `0.15` point as the default recommendation yet
- cached dense remains the best CPU execution path even at `0.10`

### 4.13 Replication of projected `0.10` and a lower-density `0.05` follow-up

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

Interpretation:

- projected `0.10` is not a one-seed artifact
- across all three checked seeds it stayed in the same broad quality regime as the
  earlier projected `0.15` runs
- this upgrades projected `0.10` to the current best **replicated** projected
  frontier in the repo

Lower-density follow-up at projected `0.05`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.4578`, density `0.05`
- cached dense projected latency: `0.4216 / 1.1203 ms` at batch `128 / 2048`
- forced sparse CSR latency: `0.7145 / 1.6468 ms` at batch `128 / 2048`

Timing caution on the `0.10` replications:

- the new seed `7` and seed `123` CPU timing blocks showed obvious outliers, so they
  should not be treated as decision-grade systems evidence
- the model-quality result is still decision-grade because the RMSE pattern is stable
  across seeds

Interpretation:

- the projected frontier likely moves at least as low as `0.05` on the wide
  benchmark without losing the quality regime
- however, `0.05` is still only a single-seed result and should not replace the
  replicated `0.10` point as the default recommendation yet
- cached dense remains the best current CPU execution path; the noisy `0.10` timing
  replications do not justify changing that conclusion

### 4.14 Replication of projected `0.05` and a lower-density `0.02` probe

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

Interpretation:

- projected `0.05` is not a one-seed artifact
- across all three checked seeds it still beat the paired dense BF16 baseline by a
  meaningful margin
- this upgrades projected `0.05` to the current best **replicated** projected
  frontier in the repo

Lower-density probe at projected `0.02`:

- seed `42`: dense RMSE `10.0171`, projected RMSE `9.7952`, density `0.02`
- cached dense projected latency: `0.4013 / 1.0639 ms` at batch `128 / 2048`
- forced sparse CSR latency: `0.6281 / 1.4430 ms` at batch `128 / 2048`

Interpretation:

- projected `0.02` still beats the tuned dense baseline on seed `42`
- however, it is the first lower-density probe that shows a meaningful quality bend
  relative to the `0.05` frontier
- cached dense remains the best current CPU execution path there as well

## 5. Main Conclusions

The ternary workstream now supports seven clear conclusions.

### 5.1 Sparse CPU wins are possible

The shadow-free branch proved that cached sparse CPU inference can beat a dense
BF16 baseline on the established repo benchmark while matching or exceeding its
accuracy.

### 5.2 Direct-discrete optimization is not solved yet

The same shadow-free branch fails on the harder nonlinear benchmark and collapses
without the shortcut. That means the direct-discrete update rule is still a live
research problem, not a finished method.

### 5.3 STE remains the simplest harder-task quality anchor

The STE ternary branch is still the simplest quality-oriented ternary baseline on
the harder nonlinear benchmarks, but it is too dense to deliver CPU sparse wins
and much slower than dense in the wider benchmark.

### 5.4 Density-aware projection is the best current sparse bridge

The projected handoff remains the best bridge between quality and sparsity on the
nonlinear benchmarks.

On the small nonlinear benchmark it improved the quality-versus-density trade-off
relative to free-running hybrid consolidation.

On the wider `256`-feature benchmark it surpassed both the tuned dense BF16
reference and the wide STE point on RMSE, the replicated frontier moved down to
`0.05` while keeping RMSE in the `9.18-9.53` regime across seeds, and a single-seed
follow-up reached `0.02` with RMSE `9.7952`.

### 5.5 The main blocker is now the systems path

The latest local projection tweaks were negative, and the projected branch now
looks good enough on quality that the remaining problem is mainly systems work:
CPU sparse inference still loses to dense, and GPU runtime needs repeated
validation before any speed claim is trustworthy.

### 5.6 Cached dense ternary inference is the current CPU default on the projected frontier

Once exact ternary weights were cached in eval mode, projected CPU inference got
much closer to the tuned dense BF16 baseline and consistently beat the forced
sparse CSR path on the wide projected frontier.

That stayed true even after the projected density moved down to `0.05`, and it
still held on the single-seed `0.02` follow-up.

That means the next CPU target is no longer just “make sparse happen.” It is “beat
cached dense projected inference with either lower density, structure, or a truly
better packed kernel.”

### 5.7 Lookup-packed ternary kernels are currently a negative direction

The first genuinely packed lookup prototype was much slower than cached dense
inference at both density `0.35` and `0.20`.

That does not rule out all packed kernels, but it does rule out spending more time
on this lookup-table variant unless another structural change makes it qualitatively
different.

## 6. Best Current Research Framing

The most defensible current framing is:

- shadow-free ternary is a promising sparse residual path with real CPU wins on
  easy linear-style workloads
- STE ternary is still the simplest quality anchor on harder tasks
- naive free-running hybrid consolidation is too destructive on the harder task
- density-projected handoff is the current best staged method for moving from a
  quality-friendly ternary solution toward a sparse shadow-free state
- on the wide tuned benchmark, projected is now the strongest overall ternary
  frontier in the repo, with the current best replicated point at density `0.05`
  and the strongest lower-density single-seed point at density `0.02`
- cached dense ternary inference is the best current CPU execution path on that
  frontier; forced sparse CSR and the lookup-packed prototype are both slower there

## 7. Recommended Next Experiments

1. Decide whether the observed quality bend at projected `0.02` is worth a full
   replication, or whether projected `0.05` is the better stopping point for the
   model-side frontier.
2. Search for structured projected variants or new CPU execution paths that can beat
   the cached-dense baseline built around the replicated `0.05` point.
3. If kernel work continues, benchmark every new sparse or packed path against both
   tuned dense BF16 and cached-dense projected inference.
4. Avoid the current lookup-packed kernel family unless a new structural idea makes
   it qualitatively different from the failed prototype.
5. Avoid strong claims about GPU training acceleration until repeated timing or
   step-time evidence exists.
