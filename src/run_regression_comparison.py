from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from model_inference_benchmarking import (
    InferenceBenchmarkConfig,
    ModelInferenceBenchmarkRecord,
    benchmark_regression_run_result,
    build_model_inference_summary,
    model_inference_record_to_dict,
    write_model_inference_benchmark_csv,
    write_model_inference_benchmark_json,
    write_model_inference_frontier_csv,
    write_model_inference_summary_json,
)
from output_paths import ARTIFACTS_DIRNAME, resolve_output_path
from regression_data import RegressionDataConfig
from regression_experiment import RegressionRunResult, TrainingConfig
from run_binary_regression import (
    DEFAULT_BINARY_HIDDEN_DIMS,
    DEFAULT_BINARY_LEARNING_RATE,
    train_binary_regression,
)
from run_regression_baseline import train_regression_baseline


DEFAULT_DENSE_HIDDEN_DIMS = (64, 32)
DEFAULT_DENSE_LEARNING_RATE = 1e-3


@dataclass(slots=True)
class RegressionComparisonResult:
    dense_result: RegressionRunResult
    binary_result: RegressionRunResult
    inference_benchmark_records: list[ModelInferenceBenchmarkRecord] | None = None

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


def regression_run_result_to_dict(result: RegressionRunResult) -> dict[str, object]:
    return {
        "device": result.device,
        "training_config": {
            "hidden_dims": list(result.training_config.hidden_dims),
            "learning_rate": result.training_config.learning_rate,
            "weight_decay": result.training_config.weight_decay,
            "epochs": result.training_config.epochs,
            "seed": result.training_config.seed,
        },
        "data_config": asdict(result.data_config),
        "history": result.history,
        "test_loss": result.test_loss,
        "test_metrics": asdict(result.test_metrics),
        "naive_test_metrics": asdict(result.naive_test_metrics),
        "runtime": asdict(result.runtime),
    }


def regression_comparison_result_to_dict(
    comparison: RegressionComparisonResult,
) -> dict[str, object]:
    result = {
        "dense_result": regression_run_result_to_dict(comparison.dense_result),
        "binary_result": regression_run_result_to_dict(comparison.binary_result),
        "deltas": {
            "test_loss": comparison.test_loss_delta,
            "mse": comparison.mse_delta,
            "mae": comparison.mae_delta,
            "rmse": comparison.rmse_delta,
            "r2": comparison.r2_delta,
            "fit_seconds": comparison.fit_time_delta,
            "test_seconds": comparison.test_time_delta,
            "predict_seconds": comparison.predict_time_delta,
            "total_seconds": comparison.total_time_delta,
            "parameter_count": comparison.parameter_count_delta,
        },
    }
    if comparison.inference_benchmark_records is not None:
        result["inference_benchmark"] = {
            "records": [
                model_inference_record_to_dict(record)
                for record in comparison.inference_benchmark_records
            ],
            "summary": build_model_inference_summary(
                comparison.inference_benchmark_records
            ),
        }
    return result


def write_regression_comparison_json(
    comparison: RegressionComparisonResult,
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(regression_comparison_result_to_dict(comparison), indent=2) + "\n",
        encoding="utf-8",
    )


def compare_dense_and_binary_regression(
    data_config: RegressionDataConfig | None = None,
    dense_training_config: TrainingConfig | None = None,
    binary_training_config: TrainingConfig | None = None,
    binary_use_input_shortcut: bool = True,
    inference_benchmark_config: InferenceBenchmarkConfig | None = None,
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
        use_input_shortcut=binary_use_input_shortcut,
    )

    inference_benchmark_records = None
    if inference_benchmark_config is not None:
        inference_benchmark_records = benchmark_regression_run_result(
            dense_result,
            model_name="dense",
            use_input_shortcut=False,
            benchmark_config=inference_benchmark_config,
            benchmark_triton_variants=False,
        )
        inference_benchmark_records.extend(
            benchmark_regression_run_result(
                binary_result,
                model_name="binary",
                use_input_shortcut=binary_use_input_shortcut,
                benchmark_config=inference_benchmark_config,
                benchmark_triton_variants=True,
            )
        )

    return RegressionComparisonResult(
        dense_result=dense_result,
        binary_result=binary_result,
        inference_benchmark_records=inference_benchmark_records,
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
    print(f"  Epochs:    {result.training_config.epochs}")
    print(f"  Learn rate:{result.training_config.learning_rate:>12.4g}")
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


def _print_inference_benchmark_records(
    records: list[ModelInferenceBenchmarkRecord],
) -> None:
    print("Model inference benchmark")
    for record in records:
        print(
            f"  {record.model_name:<6} batch={record.batch_size:<5} hidden={list(record.hidden_dims)} "
            f"shortcut={record.use_input_shortcut} triton={record.use_triton_packed_inference} "
            f"latency={record.latency_ms:.4f}ms rmse={record.test_rmse:.4f} r2={record.test_r2:.4f}"
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dense and binary regression experiments back to back."
    )
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dense-epochs", type=int, default=None)
    parser.add_argument("--binary-epochs", type=int, default=None)
    parser.add_argument("--dense-learning-rate", type=float, default=None)
    parser.add_argument("--binary-learning-rate", type=float, default=None)
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
    parser.add_argument(
        "--disable-binary-shortcut",
        action="store_true",
        help="Disable the dense residual shortcut inside the binary model.",
    )
    parser.add_argument(
        "--skip-inference-benchmark",
        action="store_true",
        help="Skip the end-to-end model inference benchmark section.",
    )
    parser.add_argument(
        "--inference-benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[128],
        help="Batch sizes to use for the model-level inference latency benchmark.",
    )
    parser.add_argument("--inference-benchmark-iterations", type=int, default=50)
    parser.add_argument("--inference-benchmark-warmup", type=int, default=10)
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=Path("regression_comparison_summary.json"),
        help="Write the comparison bundle JSON under /mnt by default.",
    )
    parser.add_argument(
        "--inference-json-out",
        type=Path,
        default=Path("regression_comparison_inference.json"),
        help="Write inference benchmark records JSON when benchmarking is enabled.",
    )
    parser.add_argument(
        "--inference-csv-out",
        type=Path,
        default=Path("regression_comparison_inference.csv"),
        help="Write inference benchmark records CSV when benchmarking is enabled.",
    )
    parser.add_argument(
        "--inference-summary-json-out",
        type=Path,
        default=Path("regression_comparison_inference_summary.json"),
        help="Write inference benchmark summary JSON when benchmarking is enabled.",
    )
    parser.add_argument(
        "--inference-frontier-csv-out",
        type=Path,
        default=Path("regression_comparison_inference_frontier.csv"),
        help="Write inference benchmark frontier CSV when benchmarking is enabled.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    dense_epochs = args.dense_epochs or args.epochs
    binary_epochs = args.binary_epochs or args.epochs
    dense_learning_rate = (
        args.dense_learning_rate
        if args.dense_learning_rate is not None
        else (
            args.learning_rate
            if args.learning_rate is not None
            else DEFAULT_DENSE_LEARNING_RATE
        )
    )
    binary_learning_rate = (
        args.binary_learning_rate
        if args.binary_learning_rate is not None
        else (
            args.learning_rate
            if args.learning_rate is not None
            else DEFAULT_BINARY_LEARNING_RATE
        )
    )

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
        epochs=dense_epochs,
        learning_rate=dense_learning_rate,
        seed=args.seed,
    )
    binary_training_config = TrainingConfig(
        hidden_dims=tuple(args.binary_hidden_dims),
        epochs=binary_epochs,
        learning_rate=binary_learning_rate,
        seed=args.seed,
    )

    inference_benchmark_config = None
    if not args.skip_inference_benchmark:
        inference_benchmark_config = InferenceBenchmarkConfig(
            batch_sizes=tuple(args.inference_benchmark_batch_sizes),
            iterations=args.inference_benchmark_iterations,
            warmup=args.inference_benchmark_warmup,
            seed=args.seed,
        )

    comparison = compare_dense_and_binary_regression(
        data_config=data_config,
        dense_training_config=dense_training_config,
        binary_training_config=binary_training_config,
        binary_use_input_shortcut=not args.disable_binary_shortcut,
        inference_benchmark_config=inference_benchmark_config,
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
    if comparison.inference_benchmark_records:
        print()
        _print_inference_benchmark_records(comparison.inference_benchmark_records)

    summary_json_out = resolve_output_path(
        args.summary_json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="regression_comparison_summary.json",
    )
    write_regression_comparison_json(comparison, summary_json_out)
    print()
    print(f"Wrote comparison summary to {summary_json_out}")

    if comparison.inference_benchmark_records:
        inference_json_out = resolve_output_path(
            args.inference_json_out,
            default_subdir=ARTIFACTS_DIRNAME,
            default_name="regression_comparison_inference.json",
        )
        inference_csv_out = resolve_output_path(
            args.inference_csv_out,
            default_subdir=ARTIFACTS_DIRNAME,
            default_name="regression_comparison_inference.csv",
        )
        inference_summary_json_out = resolve_output_path(
            args.inference_summary_json_out,
            default_subdir=ARTIFACTS_DIRNAME,
            default_name="regression_comparison_inference_summary.json",
        )
        inference_frontier_csv_out = resolve_output_path(
            args.inference_frontier_csv_out,
            default_subdir=ARTIFACTS_DIRNAME,
            default_name="regression_comparison_inference_frontier.csv",
        )

        write_model_inference_benchmark_json(
            comparison.inference_benchmark_records, inference_json_out
        )
        write_model_inference_benchmark_csv(
            comparison.inference_benchmark_records, inference_csv_out
        )
        write_model_inference_summary_json(
            comparison.inference_benchmark_records, inference_summary_json_out
        )
        write_model_inference_frontier_csv(
            comparison.inference_benchmark_records, inference_frontier_csv_out
        )
        print(f"Wrote comparison inference JSON to {inference_json_out}")
        print(f"Wrote comparison inference CSV to {inference_csv_out}")
        print(
            f"Wrote comparison inference summary to {inference_summary_json_out}"
        )
        print(
            f"Wrote comparison inference frontier CSV to {inference_frontier_csv_out}"
        )


if __name__ == "__main__":
    main()
