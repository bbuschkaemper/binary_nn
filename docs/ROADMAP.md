# Roadmap

Last updated: 2026-03-18

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

The latest work established four important facts.

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

## 3. Immediate Next Steps

### 3.1 Build a decision-grade GPU measurement path

Before making new speed claims, the repo needs repeated GPU-side measurements.

Add or standardize:

- repeated run averages, not one-off wall-clock totals
- mean step time and step-time variance
- forward/backward or fit-time breakdowns when practical
- peak GPU memory
- clear separation between training runtime and inference runtime

### 3.2 Freeze the current quality anchor

The default quality test should now be:

- the wide nonlinear benchmark
- tuned dense BF16 baseline
- projected / STE ternary reference runs as the low-bit anchor

The main baseline set should be:

- dense BF16
- STE ternary
- projected ternary at the current strong operating point

### 3.3 Prototype a new training idea: refresh-scheduled projected ternary

The most promising next local hypothesis is:

- keep a cached discrete ternary state stable for `K` optimizer steps
- update the latent or auxiliary state every step if needed
- rebuild the ternary state only on refresh boundaries
- optionally apply projection or density control only on refresh boundaries too

This should be tested first because it is the cleanest way to combine:

- projected-family quality
- shadow-free-style discrete-state motivation
- a path toward fewer GPU-side representation changes and better kernel alignment

### 3.4 Add activation-side low-bit experiments

The repo currently focuses mostly on low-bit weights.

The next training-idea pass should treat activation precision as a first-class knob:

- start with easy-to-toggle activation quantization in forward
- keep evaluation fair against the same dense reference
- do not mix too many novel changes at once before the refresh-scheduled baseline is
  understood

### 3.5 Prepare a GPU inference path that matches the training representation

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
2. add decision-grade GPU timing and memory measurement
3. implement refresh-scheduled projected ternary training
4. compare it against dense BF16 and the current projected baseline
5. only then add activation-side low-bit experiments and GPU inference kernels

## 6. Suggested First Milestone

The first milestone should be one decision-grade experiment on the wide nonlinear
benchmark where a new low-bit training variant:

- keeps RMSE in the projected / STE quality regime
- improves GPU step time or end-to-end runtime relative to the current projected path
- produces a stable discrete state that a future GPU inference kernel could reuse

## 7. Recommended Execution Order

1. extend comparison tooling with GPU timing and memory stats
2. lock the wide benchmark config and tuned dense BF16 reference
3. add refresh-window logic and cached-state logic to the projected training path
4. run ablations on refresh interval, projection interval, and density schedule
5. add activation-side low-bit toggles only after the refresh path is understood
6. benchmark operator-level GPU inference for the resulting representation
7. revisit pure shadow-free or structured variants only if the projected-refresh path
   plateaus

## 8. Decision Note For Future Sessions

If there is only enough time for one substantial change, choose training-rule and
GPU-measurement work over more CPU sparsity work.
