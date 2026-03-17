from __future__ import annotations

import argparse

from regression_data import (
    RegressionDataConfig,
)
from regression_experiment import TrainingConfig, train_regression_model
from regression_models import DenseRegressor


def train_regression_baseline(
    data_config: RegressionDataConfig | None = None,
    training_config: TrainingConfig | None = None,
):
    return train_regression_model(
        model_builder=lambda input_dim, hidden_dims: DenseRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ),
        data_config=data_config,
        training_config=training_config,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a dense regression baseline on generated data."
    )
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    result = train_regression_baseline(
        data_config=RegressionDataConfig(
            n_samples=args.samples,
            n_features=10,
            n_informative=10,
            noise=args.noise,
            batch_size=args.batch_size,
            random_state=args.seed,
        ),
        training_config=TrainingConfig(
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
