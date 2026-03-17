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
