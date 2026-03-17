from __future__ import annotations

import argparse

from regression_data import RegressionDataConfig
from regression_experiment import TrainingConfig, train_regression_model
from regression_models import BinaryRegressor


DEFAULT_BINARY_HIDDEN_DIMS = (8,)
DEFAULT_BINARY_LEARNING_RATE = 3e-3


def train_binary_regression(
    data_config: RegressionDataConfig | None = None,
    training_config: TrainingConfig | None = None,
):
    resolved_training_config = training_config or TrainingConfig(
        hidden_dims=DEFAULT_BINARY_HIDDEN_DIMS,
        learning_rate=DEFAULT_BINARY_LEARNING_RATE,
    )
    return train_regression_model(
        model_builder=lambda input_dim, hidden_dims: BinaryRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ),
        data_config=data_config,
        training_config=resolved_training_config,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary-weight regression network on generated data."
    )
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_BINARY_LEARNING_RATE
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    result = train_binary_regression(
        data_config=RegressionDataConfig(
            n_samples=args.samples,
            n_features=10,
            n_informative=10,
            noise=args.noise,
            batch_size=args.batch_size,
            random_state=args.seed,
        ),
        training_config=TrainingConfig(
            hidden_dims=DEFAULT_BINARY_HIDDEN_DIMS,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    )

    print(f"Device: {result.device}")
    print(f"Test RMSE: {result.test_metrics.rmse:.4f}")
    print(f"Test MAE:  {result.test_metrics.mae:.4f}")
    print(f"Test R2:   {result.test_metrics.r2:.4f}")
    print(f"Naive RMSE: {result.naive_test_metrics.rmse:.4f}")


if __name__ == "__main__":
    main()
