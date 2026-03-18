# Binary Regression Experiment Log

Last updated: 2026-03-17

This document records the concrete ideas tested so far for the regression
proof-of-concept, the measurements that came out of those tests, and the
current best binary configurations.

## Document Role

Use this file when you need the detailed story: what was tried, what worked,
what failed, and what numbers were observed.

## 1. Goal

The working objective for this repository is not just to make a binary-weight
model train at all. The stronger goal is:

- get a binary-weight model that is accuracy-competitive with the dense
  baseline on the generated regression task
- reduce training and inference wall-clock time versus the dense baseline when
  possible
- keep the path compatible with later work on actual packed or specialized
  low-bit kernels

## 2. Initial Baseline State

The repository started with:

- a dense Lightning regression baseline in `src/run_regression_baseline.py`
- a synthetic `scikit-learn` regression dataset with 10 features
- a dense MLP baseline that already performs very well on this task

The first binary attempt used:

- `BinaryLinear` layers with a sign straight-through estimator
- latent real-valued weights that are clipped into `[-1, 1]`
- per-output absmean scaling in the forward pass
- batch normalization and `Hardtanh` activations in the binary hidden path
- a dense output head for regression

That first version learned the task, but it was clearly behind the dense model
and was also slower in wall-clock time on the current PyTorch implementation.

## 3. Steps Taken

### 3.1 Shared experiment runner

The dense and binary experiments were refactored to share one Lightning
training and evaluation path in `src/regression_experiment.py`.

This gave us:

- consistent metrics across runs
- a consistent naive baseline
- a common place to capture fit, test, predict, and total runtime
- a reusable comparison harness

### 3.2 Added dense-vs-binary comparison script

`src/run_regression_comparison.py` was added to compare:

- dense metrics
- binary metrics
- signed `binary - dense` deltas
- runtime deltas
- parameter count deltas

Separate knobs were also added for:

- dense hidden dimensions
- binary hidden dimensions
- dense learning rate
- binary learning rate
- dense epochs
- binary epochs

This made it possible to search for a real quality-time frontier rather than
assuming one architecture should be compared with one single default.

### 3.3 Swept binary width and learning rate

Observed result from the first focused sweep at `4096` samples and `50` epochs:

- the original binary defaults were under-tuned
- increasing binary learning rate from `1e-3` to `2e-3` mattered a lot
- larger plain binary MLPs improved accuracy, but also increased runtime and
  often still underperformed the dense baseline

One especially important finding was:

- plain binary `(64, 32)` with learning rate `2e-3` reached roughly
  `RMSE 21.42` and `R2 0.9910`

That was much better than the original plain-binary defaults, but still not
good enough to call the binary model dense-competitive.

### 3.4 Tested a binary residual architecture

The next idea was to add a cheap dense linear shortcut on top of the binary
path.

Reasoning:

- the regression target is strongly linear
- forcing the binary stack to relearn the full linear map is unnecessary
- a dense shortcut can absorb the easy linear structure while the binary branch
  models the residual corrections

This turned out to be the first change that materially shifted the frontier.

### 3.5 Ablated batch normalization and hidden size

The residual binary model was then tested with:

- with and without batch norm
- very small hidden sizes such as `(8, 8)` and `(16,)`
- larger hidden sizes such as `(16, 16)` and `(32, 16)`

The main finding was:

- batch norm improved accuracy in some cases, but often at a large runtime cost
- a very small residual binary path could already be highly competitive
- the best tradeoff for this task was not a large binary network, but a small
  residual binary branch with a tuned learning rate

### 3.6 Literature-guided activation-shaping check

After the residual shortcut worked, the next literature-guided idea came from:

- `Bi-Real Net`, which strongly supports identity shortcuts and improved binary
  optimization
- `ReActNet`, which argues that activation distribution shaping can matter a
  lot for binary networks

The concrete low-cost variants tested here were:

- plain `Hardtanh`
- learnable bias before `Hardtanh`
- `PReLU`

On this regression task, none of those activation-shaping variants beat the
current default residual binary model. The plain `Hardtanh` version remained
best on both accuracy and runtime among the tested variants.

That means the current bottleneck in this repository is no longer obviously the
activation nonlinearity. The shortcut and optimizer settings mattered more.

### 3.7 Added a reusable sweep harness

To move beyond manual one-off command history, `src/run_binary_regression_sweep.py`
was added to:

- run a dense reference once
- sweep a curated binary grid over hidden dims, learning rates, and epoch
  counts
- print the best-RMSE candidates, fastest candidates, and the Pareto frontier

This makes it easier to keep the search scientific rather than anecdotal.

### 3.8 Added a Triton packed inference path

The next systems step was to separate training-time and inference-time goals.

Training remains on the existing PyTorch path so that:

- latent real-valued weights still receive gradients normally
- the current optimization behavior remains unchanged
- experiments stay fully compatible with PyTorch modules and Lightning

Inference now has an additional experimental path:

- binary weights are packed into 32-bit sign words
- per-output scales are stored separately
- an eval-only Triton kernel computes the packed binary linear map directly on
  CUDA tensors

This is intentionally inference-only for now. The purpose is to test the
kernel and representation idea without destabilizing training.

The new synthetic benchmark entry point is:

- `src/benchmark_packed_binary_kernels.py`

That benchmark uses larger matrix shapes than the small regression task so the
custom-kernel story can be judged in a more realistic throughput setting.

First measured benchmark results on the repository's `NVIDIA L4` machine:

- shape `(256, 1024, 1024)`: about `2.33x` Triton speedup, max abs diff about
  `0.0017`
- shape `(512, 2048, 2048)`: about `2.38x` Triton speedup, max abs diff about
  `0.0017`

This is the first direct evidence in the repo that the packed custom-kernel
direction can produce the kind of runtime gains that the generic PyTorch path
could not.

### 3.9 Added exportable sweeps and end-to-end inference benchmarking

The next gap after the packed-kernel proof of concept was experiment hygiene and
model-level measurement.

Two upgrades were added:

- `src/run_binary_regression_sweep.py` can now export full sweep results to
  JSON and CSV
- `src/benchmark_model_inference.py` benchmarks whole dense and binary
  regressors, not just isolated binary linear kernels, and includes trained
  model quality metrics in the exported artifact

The model benchmark also includes two important ablations:

- binary shortcut on vs off
- Triton packed inference on vs off

That makes it possible to separate three effects that were previously mixed
together:

- architecture gain from the dense residual shortcut
- systems gain from the packed Triton inference path
- remaining gap between microkernel wins and end-to-end model wins

The dense-vs-binary comparison workflow now also includes the model-level
inference benchmark section by default, so the normal report includes both task
quality metrics and trained-model latency records.

### 3.10 Integrated trained-model inference benchmarking into the comparison workflow

The next refinement was to reduce duplication between:

- the standalone trained-model inference benchmark
- the dense-vs-binary comparison report

That was done by introducing shared inference benchmarking utilities and using
them in both entry points.

This changed the workflow in two important ways:

- the comparison report now includes trained-model inference latency records by
  default
- the standalone model benchmark now exports quality metrics and latency from
  the same trained models, rather than only benchmarking randomly initialized
  models on synthetic inputs

This is a meaningful quality improvement to the research process because it
reduces the chance of drawing conclusions from inconsistent benchmarking paths.

First measured end-to-end inference results on the repository's `NVIDIA L4`
machine with `input_dim=1024` and binary hidden dims `(1024,)`:

- dense `(1024, 1024)` at batch `512`: about `0.2107ms`
- binary with shortcut, no Triton at batch `512`: about `0.1521ms`
- binary with shortcut and Triton at batch `512`: about `0.1108ms`
- dense `(1024, 1024)` at batch `2048`: about `0.7788ms`
- binary with shortcut, no Triton at batch `2048`: about `0.5287ms`
- binary with shortcut and Triton at batch `2048`: about `0.2611ms`

Interpretation:

- the Triton microkernel speedup survives end to end in the model benchmark
- the residual shortcut is still worth keeping in the binary model
- the repo now has both microkernel and model-level evidence that custom packed
  kernels materially improve binary inference throughput

Additional compact comparison-run validation on the regression workflow:

- dense at `10` epochs on `1024` samples: about `0.1850ms` model-level latency
  for benchmark batch `128`, with `RMSE 49.27` and `R2 0.9419`
- binary at `10` epochs on `1024` samples, shortcut on, Triton off: about
  `0.2892ms`
- binary at `10` epochs on `1024` samples, shortcut on, Triton on: about
  `0.1850ms`

This short-run result should not be treated as a quality conclusion because the
binary model is clearly undertrained at `10` epochs. It is mainly a validation
that the comparison workflow now reports trained-model latency and quality in a
single run.

### 3.11 Refreshed decision-grade artifact bundle and larger-width benchmarking

The next step after marking the saved `smoke/` bundle as validation-only was to
rerun the main artifact flows under a non-smoke namespace:

- `/mnt/binary_nn/artifacts/2026-03-17-decision/binary_sweep_*`
- `/mnt/binary_nn/artifacts/2026-03-17-decision/model_benchmark_default_*`
- `/mnt/binary_nn/artifacts/2026-03-17-decision/model_benchmark_wide_*`
- `/mnt/binary_nn/artifacts/2026-03-17-decision/kernel_benchmark_large_*`

The binary sweep reconfirmed the earlier frontier rather than overturning it.

Dense reference on the standard regression task:

- hidden dims `(64, 32)`
- learning rate `1e-3`
- epochs `75`
- `RMSE 14.9243`
- total runtime `6.6829s`

Best binary quality point from the refreshed sweep:

- hidden dims `(8,)`
- learning rate `3e-3`
- epochs `75`
- `RMSE 12.4447`
- total runtime `8.5400s`

Best balanced frontier point from the refreshed sweep:

- hidden dims `(16,)`
- learning rate `3e-3`
- epochs `75`
- `RMSE 12.7292`
- total runtime `6.4044s`

Fast speed-oriented point from the refreshed sweep:

- hidden dims `(8,)`
- learning rate `3e-3`
- epochs `40`
- `RMSE 15.1069`
- total runtime `3.5248s`

Interpretation:

- the binary shortcut architecture remains the right baseline
- the `(8,)` and `(16,)` configurations still dominate the larger hidden-dim
  options on this task
- the repo now has a fresh non-smoke binary frontier artifact to use for later
  ternary or int2 comparisons

The trained-model inference benchmark was then rerun in two modes.

Default small-model benchmark, using the original `10`-feature task but larger
benchmark batch sizes:

- dense latency stayed around `0.18ms` from batch `512` through `16384`
- binary with shortcut and Triton stayed around `0.178ms` to `0.189ms`
- binary without shortcut and Triton was fastest, around `0.126ms` to `0.133ms`,
  but much worse on quality with `RMSE 57.08`

Interpretation:

- the tiny trained-model benchmark is mostly overhead-bound
- it is still useful for validating that the shortcut and Triton ablations are
  exported correctly
- it is not a strong proxy for realistic systems scaling

To address that limitation, `src/benchmark_model_inference.py` was extended to
accept `--features` and `--informative-features`, making it possible to run a
wider trained-model benchmark without code edits.

Wider trained-model benchmark on `NVIDIA L4` with:

- `features=1024`
- dense hidden dims `(1024, 1024)`
- binary hidden dims `(1024, 1024)`
- benchmark batches `512`, `2048`, `8192`, `16384`

Measured results for the binary model with shortcut:

- batch `512`: Triton `0.2406ms` vs no-Triton `0.4891ms`, about `2.03x` faster
- batch `2048`: Triton `0.5729ms` vs no-Triton `0.8206ms`, about `1.43x` faster
- batch `8192`: Triton `3.5930ms` vs no-Triton `5.1903ms`, about `1.44x` faster
- batch `16384`: Triton `12.7212ms` vs no-Triton `9.1373ms`, a regression

Quality on that wider benchmark was also striking:

- dense `RMSE 577.75`, `R2 0.9074`
- binary with shortcut `RMSE 18.11`, `R2 0.9999`
- binary without shortcut `RMSE 227.91`, `R2 0.9856`

Interpretation:

- the shortcut is still a major quality feature in the wider setting
- Triton clearly helps through moderate and fairly large batch sizes
- the current implementation has a high-batch scaling cliff that now deserves
  profiling before any major kernel rewrite or representation pivot

The larger packed-kernel benchmark reinforced the systems story directly.

Measured packed-kernel results on `NVIDIA L4`:

- `(512, 2048, 2048)`: `2.65x` speedup, max abs diff `0.001817`
- `(1024, 4096, 4096)`: `2.54x` speedup, max abs diff `0.001816`
- `(2048, 4096, 4096)`: `2.15x` speedup, max abs diff `0.001986`
- `(1024, 8192, 8192)`: `2.72x` speedup, max abs diff `0.001882`

This refresh substantially strengthens the repo's current evidence base:

- the binary training frontier is stable
- the packed Triton kernel keeps winning by roughly `2x` to `2.7x` on larger
  synthetic shapes
- end-to-end model wins are real on wider models, but not monotonic at the
  largest tested batch size

### 3.12 Isolated the high-batch regression and corrected dense benchmarking precision

The next questions after the wider benchmark refresh were:

1. is the batch-`16384` Triton regression a full-model effect or a kernel-local one?
2. were the wide dense baselines being under-measured on the `NVIDIA L4`
   because float32 matmul precision was left at the default setting?

To answer the first question, the wide binary shortcut model was profiled again
at the exact regressing layer shape:

- batch size `16384`
- input dim `1024`
- binary layer out dim `1024`

Measured result from the isolated profile:

- full model, no Triton: `8.63ms`
- full model, Triton: `12.71ms`
- first `BinaryLinear`, no Triton: `3.66ms`
- first `BinaryLinear`, Triton: `5.30ms`
- direct packed-kernel reference path: `3.53ms`
- direct packed-kernel Triton path: `5.29ms`

Interpretation:

- the regression is kernel-local at `(16384, 1024, 1024)`
- packed-weight caching is not the main problem here
- the current Triton kernel loses directly to the reference GEMM-backed path at
  that specific high-batch, moderate-width operating point

That means the next systems optimization should focus on:

- kernel tiling or autotune coverage for very large `M` with moderate `K` and `N`
- not on model-level bookkeeping or shortcut logic

To answer the second question, the wide dense baseline was benchmarked on the
same trained weights under three precision settings:

- `highest`
- `high`
- `medium`

Measured dense latencies for hidden dims `(1024, 1024)`:

- batch `512`: `0.2120ms` at `highest`, `0.1826ms` at `high`, `0.1823ms` at `medium`
- batch `2048`: `0.7767ms` at `highest`, `0.3487ms` at `high`, `0.3667ms` at `medium`
- batch `8192`: `4.2243ms` at `highest`, `1.7806ms` at `high`, `1.7040ms` at `medium`
- batch `16384`: `7.9693ms` at `highest`, `3.7374ms` at `high`, `3.6229ms` at `medium`

Interpretation:

- the earlier wide dense results were materially under-measured
- on `NVIDIA L4`, leaving float32 matmul precision at the default setting makes
  the dense wide baseline look much slower than it should
- future wide model comparisons in this repo should set matmul precision
  explicitly, preferably `high` or `medium`

After adding a `--matmul-precision` option to
`src/benchmark_model_inference.py`, the full wide benchmark was rerun with
`--matmul-precision medium`.

Updated wide-model comparison under `medium` precision:

- dense `(1024, 1024)` at batch `512`: `0.1861ms`
- dense `(1024, 1024)` at batch `2048`: `0.2900ms`
- dense `(1024, 1024)` at batch `8192`: `1.5427ms`
- dense `(1024, 1024)` at batch `16384`: `3.6230ms`

- binary shortcut with Triton at batch `512`: `0.2415ms`
- binary shortcut with Triton at batch `2048`: `0.5634ms`
- binary shortcut with Triton at batch `8192`: `3.6516ms`
- binary shortcut with Triton at batch `16384`: `12.6919ms`

This materially changes the systems interpretation:

- the binary shortcut model still dominates on quality in the wide regression
  setting
- the packed Triton path still helps the binary model relative to binary
  no-Triton at small batches
- but once dense matmul precision is configured fairly, the wide dense baseline
  is faster than the quality-preserving binary shortcut model across all tested
  batches on this machine

That means the repo's current systems story is now more precise:

- binary training quality is strong
- packed binary kernels are strong on larger synthetic shapes
- but the current wide end-to-end model path is not yet beating a properly
  configured dense baseline on `NVIDIA L4`

### 3.13 Expanded Triton autotune coverage and normalized the full wide GEMM baseline

After isolating the kernel-local regression, two follow-up changes were made.

First, the packed Triton kernel autotune space was expanded to include larger
output tiles and to key autotuning on `K` as well as `M` and `N`.

The new candidate grid now includes:

- `BLOCK_M` in `{16, 32, 64, 128}`
- `BLOCK_N` in `{16, 32, 64}`
- explicit `num_stages=2`
- autotune key `("M", "N", "K")`

The goal was to help the regime with very large batch dimension `M` and more
moderate matrix widths.

The widened benchmark was then rerun at `--matmul-precision medium`.

Post-retune wide-model results on `NVIDIA L4`:

- dense at batch `512`: `0.2031ms`
- dense at batch `2048`: `0.2914ms`
- dense at batch `8192`: `1.6395ms`
- dense at batch `16384`: `3.6349ms`

- binary shortcut, no Triton at batch `512`: `0.4986ms`
- binary shortcut, Triton at batch `512`: `0.2465ms`

- binary shortcut, no Triton at batch `2048`: `0.4908ms`
- binary shortcut, Triton at batch `2048`: `0.4354ms`

- binary shortcut, no Triton at batch `8192`: `2.1317ms`
- binary shortcut, Triton at batch `8192`: `2.9550ms`

- binary shortcut, no Triton at batch `16384`: `4.5615ms`
- binary shortcut, Triton at batch `16384`: `12.6656ms`

Interpretation:

- the autotune expansion materially improved the Triton path at batch `2048`
  relative to the earlier `medium` run
- it also improved the Triton path at batch `8192`, but not enough to beat the
  no-Triton baseline there
- it did not fix the catastrophic loss at batch `16384`

The kernel-local regression was then rechecked directly after the autotune
change at random shape `(16384, 1024, 1024)`:

- reference path: `3.2743ms`
- Triton path: `5.0160ms`
- speedup: `0.65x`

So the current conclusion remains:

- the extra autotune coverage improves the medium-large regime
- the kernel still loses directly at the highest tested batch size
- fixing the `16384 x 1024 x 1024` operating point likely needs deeper kernel
  work than just adding a few more tile options

Second, the binary no-Triton path was benchmarked under explicit matmul
precision settings, just as the dense baseline already was.

Wide binary no-Triton results by precision:

- shortcut batch `2048`: `0.8768ms` at `highest`, `0.4866ms` at `high`,
  `0.4826ms` at `medium`
- shortcut batch `8192`: `4.7805ms` at `highest`, `2.1846ms` at `high`,
  `2.1858ms` at `medium`
- shortcut batch `16384`: `8.8401ms` at `highest`, `4.5263ms` at `high`,
  `4.5032ms` at `medium`

- no-shortcut batch `2048`: `1.0507ms` at `highest`, `0.4699ms` at `high`,
  `0.4706ms` at `medium`
- no-shortcut batch `8192`: `4.9240ms` at `highest`, `2.0629ms` at `high`,
  `2.0733ms` at `medium`
- no-shortcut batch `16384`: `8.5893ms` at `highest`, `4.2681ms` at `high`,
  `4.2631ms` at `medium`

Interpretation:

- the binary no-Triton path benefits from Tensor Core-friendly precision almost
  as much as the dense baseline does
- fair wide-model systems comparisons in this repo must therefore normalize
  precision for both dense and no-Triton binary paths, not just dense alone

### 3.14 Added wide comparison controls and a conservative Triton fallback policy

The next pragmatic step was to improve the normal comparison workflow and to
stop paying a known bad inference cost in the widest regressing case.

Two implementation changes were made:

- `src/run_regression_comparison.py` now accepts `--features` and
  `--informative-features`, so wide regression experiments can be run through
  the comparison entry point instead of only through the standalone model
  benchmark
- `BinaryLinear` now uses a conservative fallback heuristic that disables Triton
  when the layer sees the currently known losing shape regime:
  very large batch size with `in_features <= 1024` and `out_features <= 1024`

The fallback is intentionally narrow. It is not a general performance model. It
only exists to avoid the specific high-batch loss that has already been
measured repeatedly.

After that change, a wide comparison run was executed with:

- `features=1024`
- dense hidden dims `(1024, 1024)`
- binary hidden dims `(1024, 1024)`
- `--matmul-precision medium`
- inference batches `512`, `2048`, `8192`, `16384`

Measured comparison-run inference results:

- dense at batch `512`: `0.3145ms`
- dense at batch `2048`: `0.2896ms`
- dense at batch `8192`: `1.6347ms`
- dense at batch `16384`: `3.6114ms`

- binary shortcut, no Triton at batch `512`: `0.4759ms`
- binary shortcut, Triton-enabled at batch `512`: `0.2366ms`

- binary shortcut, no Triton at batch `2048`: `0.4721ms`
- binary shortcut, Triton-enabled at batch `2048`: `0.4432ms`

- binary shortcut, no Triton at batch `8192`: `2.1540ms`
- binary shortcut, Triton-enabled at batch `8192`: `2.9375ms`

- binary shortcut, no Triton at batch `16384`: `4.5720ms`
- binary shortcut, Triton-enabled at batch `16384`: `4.5199ms`

Interpretation:

- the fallback removes the worst known penalty at batch `16384`
- the wide comparison workflow is now capable of producing decision-grade wide
  artifacts directly
- the current Triton path still helps at smaller batches, is marginal at `2048`,
  loses at `8192`, and is now effectively bypassed at the highest tested batch

That changes the practical systems recommendation:

- keep Triton available because it still helps in part of the regime
- but do not force it unconditionally for large-batch wide inference
- the next kernel iteration should aim to replace this heuristic with a real
  performance-aware specialization or a better kernel implementation

## 4. Best Measured Configurations So Far

All numbers below are from the same generated regression task with:

- `n_samples=4096`
- `noise=12.0`
- `batch_size=128`
- `seed=42`

### 4.1 Dense reference

Dense baseline:

- hidden dims: `(64, 32)`
- learning rate: `1e-3`
- epochs: `75`
- RMSE: `14.9243`
- R2: `0.9956`
- parameter count: `2817`
- total measured runtime: `8.5327s`

### 4.2 Quality-oriented binary operating point

Residual binary model:

- hidden dims: `(8,)`
- learning rate: `3e-3`
- epochs: `75`
- RMSE: `12.4451`
- R2: `0.9970`
- parameter count: `108`
- total measured runtime: `6.2923s`

Interpretation:

- accuracy is slightly better than the dense baseline on this task
- parameter count is dramatically smaller
- runtime is now very close to the dense baseline even at the quality-oriented setting

### 4.3 Speed-oriented binary operating point

Residual binary model:

- hidden dims: `(8,)`
- learning rate: `3e-3`
- epochs: `40`
- RMSE: `15.1090`
- R2: `0.9955`
- parameter count: `108`
- total measured runtime: `3.4380s`

Interpretation:

- much faster than the dense baseline in total training runtime
- still clearly better than a naive baseline
- now only slightly behind the dense baseline in quality

This is the first configuration that provides a real speed advantage in the
current codebase.

## 5. Current Conclusion

The strongest conclusion so far is:

- a plain binary MLP was not good enough
- a residual binary model with a dense shortcut is meaningfully better
- tuning the binary learning rate separately from the dense baseline is
  necessary
- the current frontier is now genuinely strong: one operating point is
  slightly better than dense on quality, and another is close to dense while
  cutting total runtime by more than half

In other words, the binary model is no longer just a weaker dense imitation.
It now has two viable operating modes:

- a quality-oriented mode that is dense-competitive on accuracy
- a speed-oriented mode that is materially faster while remaining very close to
  the dense baseline

## 6. Why Runtime Is Still Hard

Even when the binary model uses far fewer parameters, the runtime win is not
automatic.

The current `BinaryLinear` implementation still relies on generic PyTorch
floating-point execution plus extra operations for:

- sign binarization
- scaling
- normalization or activation handling

That means the model is still not using the kind of packed or specialized
binary kernels that would make the systems story really compelling.

So the current results should be read like this:

- model quality is now in a good enough regime to justify further work
- the next serious runtime improvements will likely require kernel-level or
  representation-level changes, not only optimizer tuning

## 7. Recommended Next Steps

The most useful next experiments are:

1. Add a sweep script that emits a Pareto table of RMSE vs total runtime.
2. Add an option to toggle the dense shortcut on and off in the comparison
   script for easier ablation.
3. Track the residual binary model in more detail across `20`, `30`, `40`,
   `50`, and `75` epochs to define a deployment-style early-stop policy.
4. Investigate whether activations or normalization can be simplified further
   without losing the current gains.
5. Start exploring a packed or specialized execution path, because the current
   implementation bottleneck is increasingly runtime overhead rather than model
   quality.
6. Use the sweep harness regularly to refresh the Pareto frontier after every
  architectural change rather than relying on one or two hand-picked runs.
7. Extend the Triton packed path beyond the current eval-only binary linear
  proof of concept once the benchmark shows a stable advantage.
