from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

from regression_data import RegressionDataConfig
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
        f"rmse={candidate.rmse:<8.4f} r2={candidate.r2:<7.4f} "
        f"total={candidate.total_seconds:<7.4f}s params={candidate.parameter_count}"
    )


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
        )
        summaries.append(
            BinarySweepSummary(
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                epochs=epochs,
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


if __name__ == "__main__":
    main()
