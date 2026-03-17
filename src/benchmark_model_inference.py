from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model_inference_benchmarking import (
    InferenceBenchmarkConfig,
    benchmark_regression_run_result,
    write_model_inference_benchmark_csv,
    write_model_inference_benchmark_json,
    write_model_inference_frontier_csv,
    write_model_inference_summary_json,
)
from output_paths import ARTIFACTS_DIRNAME, resolve_output_path
from regression_data import RegressionDataConfig
from regression_experiment import TrainingConfig
from run_binary_regression import train_binary_regression
from run_regression_baseline import train_regression_baseline


def _parse_hidden_dims(spec: str) -> tuple[int, ...]:
    return tuple(int(part) for part in spec.split(",") if part.strip())


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train dense and binary regressors, then benchmark end-to-end inference with quality metrics."
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-hidden-dims", type=str, default="64,32")
    parser.add_argument("--binary-hidden-dims", type=str, default="8")
    parser.add_argument("--dense-epochs", type=int, default=75)
    parser.add_argument("--binary-epochs", type=int, default=75)
    parser.add_argument("--dense-learning-rate", type=float, default=1e-3)
    parser.add_argument("--binary-learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--benchmark-batch-sizes", nargs="+", type=int, default=[128, 512, 2048]
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--skip-shortcut-ablation",
        action="store_true",
        help="Skip training the binary no-shortcut variant in the ablation matrix.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("trained_model_benchmark.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("trained_model_benchmark.csv"),
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=Path("trained_model_benchmark_summary.json"),
    )
    parser.add_argument(
        "--frontier-csv-out",
        type=Path,
        default=Path("trained_model_benchmark_frontier.csv"),
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
    benchmark_config = InferenceBenchmarkConfig(
        batch_sizes=tuple(args.benchmark_batch_sizes),
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
    )

    dense_result = train_regression_baseline(
        data_config=data_config,
        training_config=TrainingConfig(
            hidden_dims=_parse_hidden_dims(args.dense_hidden_dims),
            epochs=args.dense_epochs,
            learning_rate=args.dense_learning_rate,
            seed=args.seed,
        ),
    )
    binary_result = train_binary_regression(
        data_config=data_config,
        training_config=TrainingConfig(
            hidden_dims=_parse_hidden_dims(args.binary_hidden_dims),
            epochs=args.binary_epochs,
            learning_rate=args.binary_learning_rate,
            seed=args.seed,
        ),
        use_input_shortcut=True,
    )

    records = benchmark_regression_run_result(
        dense_result,
        model_name="dense",
        use_input_shortcut=False,
        benchmark_config=benchmark_config,
        benchmark_triton_variants=False,
    )
    records.extend(
        benchmark_regression_run_result(
            binary_result,
            model_name="binary",
            use_input_shortcut=True,
            benchmark_config=benchmark_config,
            benchmark_triton_variants=True,
        )
    )

    if not args.skip_shortcut_ablation:
        binary_no_shortcut_result = train_binary_regression(
            data_config=data_config,
            training_config=TrainingConfig(
                hidden_dims=_parse_hidden_dims(args.binary_hidden_dims),
                epochs=args.binary_epochs,
                learning_rate=args.binary_learning_rate,
                seed=args.seed,
            ),
            use_input_shortcut=False,
        )
        records.extend(
            benchmark_regression_run_result(
                binary_no_shortcut_result,
                model_name="binary",
                use_input_shortcut=False,
                benchmark_config=benchmark_config,
                benchmark_triton_variants=True,
            )
        )

    print("End-to-end model inference benchmark")
    print(f"Benchmark device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()
    for record in records:
        print(
            f"model={record.model_name:<6} batch={record.batch_size:<5} hidden={list(record.hidden_dims)} "
            f"shortcut={record.use_input_shortcut} triton={record.use_triton_packed_inference} "
            f"latency={record.latency_ms:.4f}ms rmse={record.test_rmse:.4f} r2={record.test_r2:.4f}"
        )

    json_out = resolve_output_path(
        args.json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="trained_model_benchmark.json",
    )
    csv_out = resolve_output_path(
        args.csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="trained_model_benchmark.csv",
    )
    summary_json_out = resolve_output_path(
        args.summary_json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="trained_model_benchmark_summary.json",
    )
    frontier_csv_out = resolve_output_path(
        args.frontier_csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="trained_model_benchmark_frontier.csv",
    )

    write_model_inference_benchmark_json(records, json_out)
    write_model_inference_benchmark_csv(records, csv_out)
    write_model_inference_summary_json(records, summary_json_out)
    write_model_inference_frontier_csv(records, frontier_csv_out)

    print()
    print(f"Wrote model benchmark JSON to {json_out}")
    print(f"Wrote model benchmark CSV to {csv_out}")
    print(f"Wrote model benchmark summary to {summary_json_out}")
    print(f"Wrote model benchmark frontier CSV to {frontier_csv_out}")


if __name__ == "__main__":
    main()
