from __future__ import annotations

import argparse

from regression_data import RegressionDataConfig
from regression_experiment import TrainingConfig, train_regression_model
from regression_models import TernaryRegressor


DEFAULT_TERNARY_HIDDEN_DIMS = (64, 32)
DEFAULT_TERNARY_LEARNING_RATE = 3e-3


def train_ternary_regression(
    data_config: RegressionDataConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    use_input_shortcut: bool = True,
    threshold_scale: float = 0.5,
):
    resolved_training_config = training_config or TrainingConfig(
        hidden_dims=DEFAULT_TERNARY_HIDDEN_DIMS,
        learning_rate=DEFAULT_TERNARY_LEARNING_RATE,
    )
    return train_regression_model(
        model_builder=lambda input_dim, hidden_dims: TernaryRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_input_shortcut=use_input_shortcut,
            threshold_scale=threshold_scale,
        ),
        data_config=data_config,
        training_config=resolved_training_config,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a ternary STE residual regressor."
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument(
        "--informative-features",
        type=int,
        default=None,
        help="Number of informative features for the generated task. Defaults to the full feature count.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument(
        "--target-kind",
        choices=("linear", "nonlinear_residual"),
        default="linear",
    )
    parser.add_argument("--nonlinear-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-pair-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TERNARY_LEARNING_RATE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_TERNARY_HIDDEN_DIMS),
    )
    parser.add_argument(
        "--disable-input-shortcut",
        action="store_true",
        help="Disable the dense residual shortcut.",
    )
    parser.add_argument("--threshold-scale", type=float, default=0.5)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Lightning precision setting, for example 32-true or bf16-mixed.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    informative_features = (
        args.informative_features
        if args.informative_features is not None
        else args.features
    )
    result = train_ternary_regression(
        data_config=RegressionDataConfig(
            n_samples=args.samples,
            n_features=args.features,
            n_informative=informative_features,
            noise=args.noise,
            target_kind=args.target_kind,
            nonlinear_scale=args.nonlinear_scale,
            nonlinear_pair_count=args.nonlinear_pair_count,
            batch_size=args.batch_size,
            random_state=args.seed,
        ),
        training_config=TrainingConfig(
            hidden_dims=tuple(args.hidden_dims),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            accelerator=args.accelerator,
            precision=args.precision,
        ),
        use_input_shortcut=not args.disable_input_shortcut,
        threshold_scale=args.threshold_scale,
    )

    model = result.model.model
    density = (
        model.ternary_nonzero_density()
        if hasattr(model, "ternary_nonzero_density")
        else float("nan")
    )
    print(f"Device: {result.device}")
    print(f"Test RMSE: {result.test_metrics.rmse:.4f}")
    print(f"Test MAE:  {result.test_metrics.mae:.4f}")
    print(f"Test R2:   {result.test_metrics.r2:.4f}")
    print(f"Naive RMSE: {result.naive_test_metrics.rmse:.4f}")
    print(f"Ternary nonzero density: {density:.4f}")


if __name__ == "__main__":
    main()
