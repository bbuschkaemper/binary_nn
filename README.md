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

By default it now also runs the binary shortcut on or off and Triton on or off
ablation matrix, then writes the full records, a summary JSON, and a frontier
CSV to /mnt under /mnt/binary_nn/artifacts/.

The latest run also wrote machine-readable outputs to `/mnt/binary_nn/artifacts/` and showed
that the Triton path survives end to end at the model level. For example, on an
`NVIDIA L4` with `input_dim=1024` and `hidden=1024`, the binary model with
shortcut and Triton reached about `0.1108ms` vs `0.1521ms` without Triton at
batch `512`, and about `0.2611ms` vs `0.5287ms` at batch `2048`.

The dense-vs-binary comparison script now also includes a model-level inference
benchmark section by default. You can disable it with:

```bash
python src/run_regression_comparison.py --skip-inference-benchmark
```

## Experiment Notes

Ongoing experiment ideas, steps taken, and measured findings are tracked in
`docs/BINARY_REGRESSION_EXPERIMENT_LOG.md`.

For session-to-session handoff and planning, also see:

- `docs/README.md`
- `docs/CURRENT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
