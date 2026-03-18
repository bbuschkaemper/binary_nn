# binary_nn

## Setup

### Devcontainer

1. On the devcontainer host machine, create a named volume for persistent data storage called `data`.
For example, to create a volume that binds to a local directory `/path/to/local/data`, run:

    ```bash
    docker volume create --name data --opt type=none --opt device=/path/to/local/data --opt o=bind
    ```

2. Create a `.env` file in the repository directory containing the keys of `.env.example`

3. Install the NVIDIA Container Toolkit using the script `setup-gpu-host.sh`.

4. Open the workspace in a devcontainer using VSCode.

## Regression Baseline

Run the initial dense regression baseline from the repository root:

```bash
python src/run_regression_baseline.py --samples 4096 --epochs 75
```

The script generates a 10-feature regression dataset with `scikit-learn`, trains a small dense neural network in `torch`, and prints test metrics alongside a naive mean-prediction baseline.

The training loop is implemented with PyTorch Lightning so the same experiment can be reused for later comparisons against 1-bit layers with minimal trainer changes.

## Binary Regression

Run the first binary-weight regression experiment from the repository root:

```bash
python src/run_binary_regression.py --samples 4096 --epochs 75
```

This version keeps the Lightning training loop but uses a binary residual regressor: a small binary hidden path trained with a straight-through estimator and latent-weight clipping, plus a dense linear shortcut that helps on regression-style targets.

At evaluation time, `BinaryLinear` now attempts an eval-only packed Triton
inference path automatically for 2D CUDA inputs under `torch.no_grad()`. The
training path remains standard PyTorch so optimization behavior is unchanged.

## Dense vs Binary Comparison

Run both experiments back to back on the same generated dataset:

```bash
python src/run_regression_comparison.py --samples 4096 --epochs 75
```

The comparison script prints the dense metrics, the binary metrics, and the signed deltas for test loss, MSE, MAE, RMSE, and $R^2$ using `binary - dense`. It also supports separate dense and binary learning rates and epoch budgets so you can search for a better quality-time tradeoff.

You can also compare different model widths directly, for example a denser binary model against the default dense baseline:

```bash
python src/run_regression_comparison.py --samples 4096 --dense-epochs 75 --binary-epochs 40 --dense-hidden-dims 64 32 --binary-hidden-dims 8 --dense-learning-rate 1e-3 --binary-learning-rate 3e-3
```

The report includes parameter counts and wall-clock fit, test, predict, and total times so you can see whether a wider binary network still has a runtime advantage on your hardware.

It now also writes a comparison bundle under `/mnt/binary_nn/artifacts/`, plus
inference benchmark JSON, CSV, summary JSON, and frontier CSV when the
inference benchmark section is enabled.

## Ternary Research Comparison

Run the ternary research comparison entry point from the repository root:

```bash
python src/run_ternary_research_comparison.py --model-family ste
```

This script trains a dense BF16-capable reference plus one ternary branch and
then benchmarks CPU inference with sparse execution disabled and enabled. The
available ternary families are:

- `shadowfree` for the direct-discrete sparse residual path
- `ste` for the quality-oriented latent-weight ternary baseline
- `hybrid` for free-running STE-to-shadow-free consolidation
- `projected` for target-density STE-to-shadow-free projection plus recovery

For the harder stress test used in the current ternary research work, run:

```bash
python src/run_ternary_research_comparison.py --model-family projected --target-kind nonlinear_residual
```

The script writes a comparison JSON plus CPU benchmark CSV under
`/mnt/binary_nn/artifacts/`.

## Binary Sweep

Run a curated binary sweep and print the current Pareto frontier:

```bash
python src/run_binary_regression_sweep.py
```

This runs the dense reference once, sweeps several binary configurations, and
reports the best-accuracy, fastest, and non-dominated binary candidates.

You can also export sweep results and disable the dense residual shortcut for
ablation:

```bash
python src/run_binary_regression_sweep.py --disable-binary-shortcut --json-out sweep.json --csv-out sweep.csv
```

Sweep artifacts now default to /mnt under /mnt/binary_nn/artifacts/, including
full JSON and CSV outputs plus a summary JSON and frontier CSV.

## Triton Kernel Benchmark

Benchmark the packed Triton inference path for binary linear layers on larger
synthetic shapes:

```bash
python src/benchmark_packed_binary_kernels.py
```

This keeps training on the standard PyTorch path but benchmarks an eval-only,
packed-sign inference kernel implemented with Triton.

The kernel benchmark now also writes JSON, CSV, summary JSON, and frontier CSV
artifacts under `/mnt/binary_nn/artifacts/` by default.

On the current `NVIDIA L4` test machine, the first benchmark run showed about
`2.33x` speedup at shape `(256, 1024, 1024)` and about `2.38x` at shape
`(512, 2048, 2048)`, with max absolute output differences around `0.0017`.

## Model Inference Benchmark

Benchmark full dense and binary regressor inference on larger synthetic inputs,
including binary shortcut ablations and Triton on/off:

```bash
python src/benchmark_model_inference.py --json-out model-benchmark.json --csv-out model-benchmark.csv
```

This benchmark now trains the compared models on the regression task first,
then emits both quality metrics and end-to-end latency records so architecture
and systems tradeoffs can be judged in one artifact.

For larger, more realistic trained-model benchmarks, you can now increase the
synthetic regression width directly from the CLI:

```bash
python src/benchmark_model_inference.py \
    --features 1024 \
    --informative-features 1024 \
    --dense-hidden-dims 1024,1024 \
    --binary-hidden-dims 1024,1024 \
    --benchmark-batch-sizes 512 2048 8192 16384
```

By default it now also runs the binary shortcut on or off and Triton on or off
ablation matrix, then writes the full records, a summary JSON, and a frontier
CSV to /mnt under /mnt/binary_nn/artifacts/.

For fair wide-model benchmarking on Tensor Core GPUs, set matmul precision
explicitly:

```bash
python src/benchmark_model_inference.py \
    --matmul-precision medium \
    --features 1024 \
    --informative-features 1024 \
    --dense-hidden-dims 1024,1024 \
    --binary-hidden-dims 1024,1024 \
    --benchmark-batch-sizes 512 2048 8192 16384
```

The latest decision-grade bundle was written to
`/mnt/binary_nn/artifacts/2026-03-17-decision/`.

Two takeaways from that refresh on `NVIDIA L4`:

- on the original small `10`-feature benchmark, the model path is largely
    overhead-bound, so latency remains almost flat even as benchmark batch size
    increases
- on the earlier wide `1024`-feature benchmark, Triton improved the binary model
    relative to binary no-Triton through batch `8192`, but that comparison left
    dense matmul precision at the default setting

After rerunning the wide benchmark with `--matmul-precision medium`, the dense
baseline became much faster:

- dense `(1024, 1024)` reached about `0.1861ms` at batch `512`, `0.2900ms` at
    `2048`, `1.5427ms` at `8192`, and `3.6230ms` at `16384`
- binary shortcut with Triton reached about `0.2415ms`, `0.5634ms`, `3.6516ms`,
    and `12.6919ms` at those same batch sizes

The binary no-Triton path also needs the same precision normalization. On the
same `NVIDIA L4`, moving from the default `highest` setting to `high` or
`medium` roughly halves wide binary no-Triton latency at the larger batches.

At batch `16384`, the current Triton path still regresses badly. Targeted
profiling showed that this is already visible at the isolated binary-layer
kernel shape `(16384, 1024, 1024)`, where the Triton kernel itself loses to the
reference path. That is now the main systems debugging target.

An initial autotune expansion improved Triton at mid-range wide batches, but it
still does not recover the `16384 x 1024 x 1024` case, so the next step is a
deeper kernel iteration rather than another documentation-only benchmark pass.

The dense-vs-binary comparison script now also includes a model-level inference
benchmark section by default. You can disable it with:

```bash
python src/run_regression_comparison.py --skip-inference-benchmark
```

The comparison workflow now also supports wide experiments directly:

```bash
python src/run_regression_comparison.py \
    --features 1024 \
    --informative-features 1024 \
    --dense-hidden-dims 1024 1024 \
    --binary-hidden-dims 1024 1024 \
    --matmul-precision medium \
    --inference-benchmark-batch-sizes 512 2048 8192 16384
```

The current code also includes a conservative runtime fallback that disables
Triton for the known losing large-batch shape regime. That means a `triton=true`
record at the highest tested wide batch can reflect a fallback to the reference
path rather than actual Triton kernel execution.

## Experiment Notes

Ongoing experiment ideas, steps taken, and measured findings are tracked in:

- `docs/BINARY_REGRESSION_EXPERIMENT_LOG.md`
- `docs/TERNARY_RESEARCH_EXPERIMENT_LOG.md`

For session-to-session handoff and planning, also see:

- `docs/README.md`
- `docs/CURRENT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
