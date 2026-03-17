from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from output_paths import ARTIFACTS_DIRNAME, resolve_output_path
from regression_data import RegressionDataConfig
from regression_experiment import RegressionRunResult
from regression_experiment import TrainingConfig
from run_binary_regression import train_binary_regression
from run_regression_baseline import train_regression_baseline


DEFAULT_BINARY_HIDDEN_DIMS_GRID = ["8", "16", "16,16", "32,16"]
DEFAULT_BINARY_LEARNING_RATES = [2e-3, 3e-3]
DEFAULT_BINARY_EPOCHS = [30, 40, 50, 75]
DEFAULT_DENSE_HIDDEN_DIMS = (64, 32)
DEFAULT_DENSE_LEARNING_RATE = 1e-3
DEFAULT_DENSE_EPOCHS = 75


@dataclass(slots=True)
class BinarySweepSummary:
    hidden_dims: tuple[int, ...]
    learning_rate: float
    epochs: int
    use_input_shortcut: bool
    rmse: float
    r2: float
    total_seconds: float
    parameter_count: int


def parse_hidden_dims(spec: str) -> tuple[int, ...]:
    return tuple(int(part) for part in spec.split(",") if part.strip())


def dominates(left: BinarySweepSummary, right: BinarySweepSummary) -> bool:
    return (
        left.rmse <= right.rmse
        and left.total_seconds <= right.total_seconds
        and (left.rmse < right.rmse or left.total_seconds < right.total_seconds)
    )


def pareto_frontier(
    candidates: list[BinarySweepSummary],
) -> list[BinarySweepSummary]:
    frontier: list[BinarySweepSummary] = []
    for candidate in candidates:
        if any(
            dominates(other, candidate) for other in candidates if other != candidate
        ):
            continue
        frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.total_seconds, item.rmse))


def _format_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    return " x ".join(str(hidden_dim) for hidden_dim in hidden_dims)


def _print_candidate(prefix: str, candidate: BinarySweepSummary) -> None:
    print(
        f"{prefix:<10} hidden={_format_hidden_dims(candidate.hidden_dims):<8} "
        f"lr={candidate.learning_rate:<7.4g} epochs={candidate.epochs:<3d} "
        f"shortcut={str(candidate.use_input_shortcut):<5} "
        f"rmse={candidate.rmse:<8.4f} r2={candidate.r2:<7.4f} "
        f"total={candidate.total_seconds:<7.4f}s params={candidate.parameter_count}"
    )


def sweep_summary_to_record(summary: BinarySweepSummary) -> dict[str, object]:
    return {
        "hidden_dims": list(summary.hidden_dims),
        "learning_rate": summary.learning_rate,
        "epochs": summary.epochs,
        "use_input_shortcut": summary.use_input_shortcut,
        "rmse": summary.rmse,
        "r2": summary.r2,
        "total_seconds": summary.total_seconds,
        "parameter_count": summary.parameter_count,
    }


def _run_result_to_record(result: RegressionRunResult) -> dict[str, object]:
    return {
        "hidden_dims": list(result.training_config.hidden_dims),
        "learning_rate": result.training_config.learning_rate,
        "epochs": result.training_config.epochs,
        "use_input_shortcut": False,
        "rmse": result.test_metrics.rmse,
        "r2": result.test_metrics.r2,
        "total_seconds": result.runtime.total_seconds,
        "parameter_count": result.runtime.parameter_count,
    }


def build_sweep_summary(
    dense_result: RegressionRunResult,
    summaries: list[BinarySweepSummary],
) -> dict[str, object]:
    best_rmse = sorted(summaries, key=lambda item: (item.rmse, item.total_seconds))[:5]
    fastest = sorted(summaries, key=lambda item: (item.total_seconds, item.rmse))[:5]
    frontier = pareto_frontier(summaries)
    return {
        "dense_reference": _run_result_to_record(dense_result),
        "candidate_count": len(summaries),
        "best_rmse_candidates": [
            sweep_summary_to_record(candidate) for candidate in best_rmse
        ],
        "fastest_candidates": [
            sweep_summary_to_record(candidate) for candidate in fastest
        ],
        "pareto_frontier": [
            sweep_summary_to_record(candidate) for candidate in frontier
        ],
    }


def _write_json(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep binary regression configurations and report a Pareto frontier."
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--binary-hidden-dims-grid",
        nargs="+",
        default=DEFAULT_BINARY_HIDDEN_DIMS_GRID,
        help="Comma-separated hidden-dim specs, e.g. 16 16,16 32,16",
    )
    parser.add_argument(
        "--binary-learning-rates",
        nargs="+",
        type=float,
        default=DEFAULT_BINARY_LEARNING_RATES,
    )
    parser.add_argument(
        "--binary-epochs-grid",
        nargs="+",
        type=int,
        default=DEFAULT_BINARY_EPOCHS,
    )
    parser.add_argument(
        "--disable-binary-shortcut",
        action="store_true",
        help="Disable the dense residual shortcut for all swept binary models.",
    )
    parser.add_argument("--json-out", type=Path, default=Path("binary_regression_sweep.json"))
    parser.add_argument("--csv-out", type=Path, default=Path("binary_regression_sweep.csv"))
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=Path("binary_regression_sweep_summary.json"),
    )
    parser.add_argument(
        "--frontier-csv-out",
        type=Path,
        default=Path("binary_regression_sweep_frontier.csv"),
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

    dense_result = train_regression_baseline(
        data_config=data_config,
        training_config=TrainingConfig(
            hidden_dims=DEFAULT_DENSE_HIDDEN_DIMS,
            epochs=DEFAULT_DENSE_EPOCHS,
            learning_rate=DEFAULT_DENSE_LEARNING_RATE,
            seed=args.seed,
        ),
    )

    print("Dense reference")
    print(
        f"dense      hidden={_format_hidden_dims(DEFAULT_DENSE_HIDDEN_DIMS):<8} "
        f"lr={DEFAULT_DENSE_LEARNING_RATE:<7.4g} epochs={DEFAULT_DENSE_EPOCHS:<3d} "
        f"rmse={dense_result.test_metrics.rmse:<8.4f} r2={dense_result.test_metrics.r2:<7.4f} "
        f"total={dense_result.runtime.total_seconds:<7.4f}s params={dense_result.runtime.parameter_count}"
    )
    print()

    summaries: list[BinarySweepSummary] = []
    configs = product(
        [parse_hidden_dims(spec) for spec in args.binary_hidden_dims_grid],
        args.binary_learning_rates,
        args.binary_epochs_grid,
    )
    for hidden_dims, learning_rate, epochs in configs:
        result = train_binary_regression(
            data_config=data_config,
            training_config=TrainingConfig(
                hidden_dims=hidden_dims,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=args.seed,
            ),
            use_input_shortcut=not args.disable_binary_shortcut,
        )
        summaries.append(
            BinarySweepSummary(
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                epochs=epochs,
                use_input_shortcut=not args.disable_binary_shortcut,
                rmse=result.test_metrics.rmse,
                r2=result.test_metrics.r2,
                total_seconds=result.runtime.total_seconds,
                parameter_count=result.runtime.parameter_count,
            )
        )

    print("Best RMSE candidates")
    for candidate in sorted(
        summaries, key=lambda item: (item.rmse, item.total_seconds)
    )[:5]:
        _print_candidate("binary", candidate)
    print()

    print("Fastest candidates")
    for candidate in sorted(
        summaries, key=lambda item: (item.total_seconds, item.rmse)
    )[:5]:
        _print_candidate("binary", candidate)
    print()

    print("Pareto frontier")
    for candidate in pareto_frontier(summaries):
        _print_candidate("frontier", candidate)

    records = [sweep_summary_to_record(summary) for summary in summaries]
    summary = build_sweep_summary(dense_result, summaries)

    json_out = resolve_output_path(
        args.json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="binary_regression_sweep.json",
    )
    csv_out = resolve_output_path(
        args.csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="binary_regression_sweep.csv",
    )
    summary_json_out = resolve_output_path(
        args.summary_json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="binary_regression_sweep_summary.json",
    )
    frontier_csv_out = resolve_output_path(
        args.frontier_csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="binary_regression_sweep_frontier.csv",
    )

    _write_json(records, json_out)
    _write_csv(records, csv_out)
    _write_json(summary, summary_json_out)
    _write_csv(summary["pareto_frontier"], frontier_csv_out)

    print()
    print(f"Wrote sweep records to {json_out}")
    print(f"Wrote sweep CSV to {csv_out}")
    print(f"Wrote sweep summary to {summary_json_out}")
    print(f"Wrote sweep frontier CSV to {frontier_csv_out}")


if __name__ == "__main__":
    main()
