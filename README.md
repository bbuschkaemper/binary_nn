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

This version keeps the Lightning training loop but swaps the hidden layers to binary-weight layers trained with a straight-through estimator, latent-weight clipping, batch normalization, and a dense output head for regression.

## Dense vs Binary Comparison

Run both experiments back to back on the same generated dataset:

```bash
python src/run_regression_comparison.py --samples 4096 --epochs 75
```

The comparison script prints the dense metrics, the binary metrics, and the signed deltas for test loss, MSE, MAE, RMSE, and $R^2$ using `binary - dense`.

You can also compare different model widths directly, for example a denser binary model against the default dense baseline:

```bash
python src/run_regression_comparison.py --samples 4096 --epochs 75 --dense-hidden-dims 64 32 --binary-hidden-dims 64 64
```

The report includes parameter counts and wall-clock fit, test, predict, and total times so you can see whether a wider binary network still has a runtime advantage on your hardware.
