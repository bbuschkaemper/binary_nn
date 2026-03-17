from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from regression_data import RegressionDataConfig
from regression_experiment import RegressionRunResult, TrainingConfig
from run_binary_regression import DEFAULT_BINARY_HIDDEN_DIMS, train_binary_regression
from run_regression_baseline import train_regression_baseline


DEFAULT_DENSE_HIDDEN_DIMS = (64, 32)


@dataclass(slots=True)
class RegressionComparisonResult:
    dense_result: RegressionRunResult
    binary_result: RegressionRunResult

    @property
    def test_loss_delta(self) -> float:
        return self.binary_result.test_loss - self.dense_result.test_loss

    @property
    def mse_delta(self) -> float:
        return self.binary_result.test_metrics.mse - self.dense_result.test_metrics.mse

    @property
    def mae_delta(self) -> float:
        return self.binary_result.test_metrics.mae - self.dense_result.test_metrics.mae

    @property
    def rmse_delta(self) -> float:
        return (
            self.binary_result.test_metrics.rmse - self.dense_result.test_metrics.rmse
        )

    @property
    def r2_delta(self) -> float:
        return self.binary_result.test_metrics.r2 - self.dense_result.test_metrics.r2

    @property
    def fit_time_delta(self) -> float:
        return (
            self.binary_result.runtime.fit_seconds
            - self.dense_result.runtime.fit_seconds
        )

    @property
    def test_time_delta(self) -> float:
        return (
            self.binary_result.runtime.test_seconds
            - self.dense_result.runtime.test_seconds
        )

    @property
    def predict_time_delta(self) -> float:
        return (
            self.binary_result.runtime.predict_seconds
            - self.dense_result.runtime.predict_seconds
        )

    @property
    def total_time_delta(self) -> float:
        return (
            self.binary_result.runtime.total_seconds
            - self.dense_result.runtime.total_seconds
        )

    @property
    def parameter_count_delta(self) -> int:
        return (
            self.binary_result.runtime.parameter_count
            - self.dense_result.runtime.parameter_count
        )


def compare_dense_and_binary_regression(
    data_config: RegressionDataConfig | None = None,
    dense_training_config: TrainingConfig | None = None,
    binary_training_config: TrainingConfig | None = None,
) -> RegressionComparisonResult:
    resolved_dense_training_config = dense_training_config or TrainingConfig(
        hidden_dims=DEFAULT_DENSE_HIDDEN_DIMS
    )
    resolved_binary_training_config = binary_training_config or TrainingConfig(
        hidden_dims=DEFAULT_BINARY_HIDDEN_DIMS
    )

    dense_result = train_regression_baseline(
        data_config=data_config,
        training_config=resolved_dense_training_config,
    )
    binary_result = train_binary_regression(
        data_config=data_config,
        training_config=resolved_binary_training_config,
    )

    return RegressionComparisonResult(
        dense_result=dense_result,
        binary_result=binary_result,
    )


def _format_metric_line(
    label: str,
    dense_value: float,
    binary_value: float,
    delta_value: float,
) -> str:
    return (
        f"{label:<10} dense={dense_value:>9.4f}  "
        f"binary={binary_value:>9.4f}  "
        f"delta={delta_value:>+9.4f}"
    )


def _format_runtime_line(
    label: str,
    dense_value: float,
    binary_value: float,
    delta_value: float,
) -> str:
    return (
        f"{label:<10} dense={dense_value:>9.4f}s "
        f"binary={binary_value:>9.4f}s "
        f"delta={delta_value:>+9.4f}s"
    )


def _format_int_line(
    label: str,
    dense_value: int,
    binary_value: int,
    delta_value: int,
) -> str:
    return (
        f"{label:<10} dense={dense_value:>9d}  "
        f"binary={binary_value:>9d}  "
        f"delta={delta_value:>+9d}"
    )


def _format_hidden_dims(hidden_dims: Sequence[int]) -> str:
    return " x ".join(str(hidden_dim) for hidden_dim in hidden_dims)


def _print_result_block(title: str, result: RegressionRunResult) -> None:
    metrics = result.test_metrics
    runtime = result.runtime
    print(title)
    print(
        f"  Hidden dims:{_format_hidden_dims(result.training_config.hidden_dims):>12}"
    )
    print(f"  Device:    {result.device}")
    print(f"  Parameters:{runtime.parameter_count:>12d}")
    print(f"  Fit time:  {runtime.fit_seconds:.4f}s")
    print(f"  Test time: {runtime.test_seconds:.4f}s")
    print(f"  Predict:   {runtime.predict_seconds:.4f}s")
    print(f"  Total:     {runtime.total_seconds:.4f}s")
    print(f"  Test loss: {result.test_loss:.4f}")
    print(f"  Test RMSE: {metrics.rmse:.4f}")
    print(f"  Test MAE:  {metrics.mae:.4f}")
    print(f"  Test R2:   {metrics.r2:.4f}")
    print(f"  Naive RMSE:{result.naive_test_metrics.rmse: .4f}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dense and binary regression experiments back to back."
    )
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dense-hidden-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_DENSE_HIDDEN_DIMS),
    )
    parser.add_argument(
        "--binary-hidden-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_BINARY_HIDDEN_DIMS),
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    data_config = RegressionDataConfig(
        n_samples=args.samples,
        n_features=10,
        n_informative=10,
        noise=args.noise,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    dense_training_config = TrainingConfig(
        hidden_dims=tuple(args.dense_hidden_dims),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    binary_training_config = TrainingConfig(
        hidden_dims=tuple(args.binary_hidden_dims),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    comparison = compare_dense_and_binary_regression(
        data_config=data_config,
        dense_training_config=dense_training_config,
        binary_training_config=binary_training_config,
    )

    _print_result_block("Dense baseline", comparison.dense_result)
    print()
    _print_result_block("Binary baseline", comparison.binary_result)
    print()
    print("Deltas (binary - dense)")
    print(
        _format_metric_line(
            "test_loss",
            comparison.dense_result.test_loss,
            comparison.binary_result.test_loss,
            comparison.test_loss_delta,
        )
    )
    print(
        _format_metric_line(
            "mse",
            comparison.dense_result.test_metrics.mse,
            comparison.binary_result.test_metrics.mse,
            comparison.mse_delta,
        )
    )
    print(
        _format_metric_line(
            "mae",
            comparison.dense_result.test_metrics.mae,
            comparison.binary_result.test_metrics.mae,
            comparison.mae_delta,
        )
    )
    print(
        _format_metric_line(
            "rmse",
            comparison.dense_result.test_metrics.rmse,
            comparison.binary_result.test_metrics.rmse,
            comparison.rmse_delta,
        )
    )
    print(
        _format_metric_line(
            "r2",
            comparison.dense_result.test_metrics.r2,
            comparison.binary_result.test_metrics.r2,
            comparison.r2_delta,
        )
    )
    print()
    print("Runtime and size deltas (binary - dense)")
    print(
        _format_int_line(
            "params",
            comparison.dense_result.runtime.parameter_count,
            comparison.binary_result.runtime.parameter_count,
            comparison.parameter_count_delta,
        )
    )
    print(
        _format_runtime_line(
            "fit_time",
            comparison.dense_result.runtime.fit_seconds,
            comparison.binary_result.runtime.fit_seconds,
            comparison.fit_time_delta,
        )
    )
    print(
        _format_runtime_line(
            "test_time",
            comparison.dense_result.runtime.test_seconds,
            comparison.binary_result.runtime.test_seconds,
            comparison.test_time_delta,
        )
    )
    print(
        _format_runtime_line(
            "predict",
            comparison.dense_result.runtime.predict_seconds,
            comparison.binary_result.runtime.predict_seconds,
            comparison.predict_time_delta,
        )
    )
    print(
        _format_runtime_line(
            "total",
            comparison.dense_result.runtime.total_seconds,
            comparison.binary_result.runtime.total_seconds,
            comparison.total_time_delta,
        )
    )


if __name__ == "__main__":
    main()
