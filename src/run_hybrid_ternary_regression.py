from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from regression_data import RegressionDataConfig
from regression_experiment import (
    RegressionRunResult,
    RegressionRuntime,
    TrainingConfig,
    train_regression_model,
)
from regression_models import (
    ControlledRefreshProjectedTernaryRegressor,
    RefreshScheduledProjectedTernaryRegressor,
    ShadowFreeTernaryRegressor,
)
from run_ternary_regression import (
    DEFAULT_TERNARY_HIDDEN_DIMS,
    DEFAULT_TERNARY_LEARNING_RATE,
    train_ternary_regression,
)


DEFAULT_HYBRID_WARM_START_EPOCHS = 50
DEFAULT_HYBRID_CONSOLIDATION_EPOCHS = 25
DEFAULT_HYBRID_CONSOLIDATION_LEARNING_RATE = 1e-3


def _parse_control_ranks(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    return tuple(int(part) for part in spec.split(",") if part.strip())


def _parse_refresh_intervals(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    return tuple(int(part) for part in spec.split(",") if part.strip())


@dataclass(slots=True)
class HybridTernaryRunResult:
    warm_start_result: RegressionRunResult
    consolidation_result: RegressionRunResult
    final_result: RegressionRunResult


def _combine_hybrid_runtime(
    warm_start_result: RegressionRunResult,
    consolidation_result: RegressionRunResult,
) -> RegressionRuntime:
    fit_seconds = (
        warm_start_result.runtime.fit_seconds
        + consolidation_result.runtime.fit_seconds
    )
    test_seconds = consolidation_result.runtime.test_seconds
    predict_seconds = consolidation_result.runtime.predict_seconds
    return RegressionRuntime(
        fit_seconds=fit_seconds,
        test_seconds=test_seconds,
        predict_seconds=predict_seconds,
        total_seconds=fit_seconds + test_seconds + predict_seconds,
        parameter_count=consolidation_result.runtime.parameter_count,
        stage_benchmarks=consolidation_result.runtime.stage_benchmarks,
    )


def _build_final_result(
    warm_start_result: RegressionRunResult,
    consolidation_result: RegressionRunResult,
) -> RegressionRunResult:
    return RegressionRunResult(
        model=consolidation_result.model,
        device=consolidation_result.device,
        history=consolidation_result.history,
        test_loss=consolidation_result.test_loss,
        test_metrics=consolidation_result.test_metrics,
        runtime=_combine_hybrid_runtime(warm_start_result, consolidation_result),
        naive_test_metrics=consolidation_result.naive_test_metrics,
        data_config=consolidation_result.data_config,
        training_config=consolidation_result.training_config,
    )


def train_hybrid_ternary_regression(
    data_config: RegressionDataConfig | None = None,
    warm_start_training_config: TrainingConfig | None = None,
    consolidation_training_config: TrainingConfig | None = None,
    *,
    use_input_shortcut: bool = True,
    threshold_scale: float = 0.5,
    consolidation_variant: str = "shadowfree",
    projection_target_density: float | None = None,
    projection_structure: str | None = None,
    projection_block_size: int | None = None,
    initial_density: float = 0.25,
    update_interval: int = 4,
    refresh_interval: int = 8,
    refresh_intervals: Sequence[int] | None = None,
    refresh_project_every_n_refreshes: int = 1,
    control_rank: int = 16,
    control_ranks: Sequence[int] | None = None,
    control_mode: str = "low_rank",
    activation_threshold_scale: float = 0.25,
    quantize_hidden_activations: bool = True,
    activation_std_multiplier: float = 1.0,
    prune_ratio: float = 0.25,
    flip_multiplier: float = 2.0,
) -> HybridTernaryRunResult:
    resolved_warm_start_training_config = warm_start_training_config or TrainingConfig(
        hidden_dims=DEFAULT_TERNARY_HIDDEN_DIMS,
        learning_rate=DEFAULT_TERNARY_LEARNING_RATE,
        epochs=DEFAULT_HYBRID_WARM_START_EPOCHS,
    )
    resolved_consolidation_training_config = (
        consolidation_training_config
        or TrainingConfig(
            hidden_dims=resolved_warm_start_training_config.hidden_dims,
            learning_rate=DEFAULT_HYBRID_CONSOLIDATION_LEARNING_RATE,
            epochs=DEFAULT_HYBRID_CONSOLIDATION_EPOCHS,
            seed=resolved_warm_start_training_config.seed,
            accelerator=resolved_warm_start_training_config.accelerator,
            devices=resolved_warm_start_training_config.devices,
            precision=resolved_warm_start_training_config.precision,
            enable_progress_bar=resolved_warm_start_training_config.enable_progress_bar,
        )
    )

    warm_start_result = train_ternary_regression(
        data_config=data_config,
        training_config=resolved_warm_start_training_config,
        use_input_shortcut=use_input_shortcut,
        threshold_scale=threshold_scale,
    )
    if consolidation_variant == "shadowfree":
        hybrid_model = ShadowFreeTernaryRegressor.from_ste_regressor(
            warm_start_result.model.model,
            target_density=projection_target_density,
            projection_structure=projection_structure,
            projection_block_size=projection_block_size,
            initial_density=initial_density,
            update_interval=update_interval,
            activation_std_multiplier=activation_std_multiplier,
            prune_ratio=prune_ratio,
            flip_multiplier=flip_multiplier,
        )
    elif consolidation_variant == "refresh_projected":
        hybrid_model = RefreshScheduledProjectedTernaryRegressor.from_ste_regressor(
            warm_start_result.model.model,
            target_density=projection_target_density,
            projection_structure=projection_structure,
            projection_block_size=projection_block_size,
            refresh_interval=refresh_interval,
            refresh_intervals=refresh_intervals,
            refresh_project_every_n_refreshes=refresh_project_every_n_refreshes,
        )
    elif consolidation_variant == "controlled_refresh_projected":
        hybrid_model = ControlledRefreshProjectedTernaryRegressor.from_ste_regressor(
            warm_start_result.model.model,
            target_density=projection_target_density,
            projection_structure=projection_structure,
            projection_block_size=projection_block_size,
            refresh_interval=refresh_interval,
            refresh_intervals=refresh_intervals,
            refresh_project_every_n_refreshes=refresh_project_every_n_refreshes,
            control_rank=control_rank,
            control_ranks=control_ranks,
            control_mode=control_mode,
            activation_threshold_scale=activation_threshold_scale,
            quantize_hidden_activations=quantize_hidden_activations,
        )
    else:
        raise ValueError(
            "consolidation_variant must be 'shadowfree', 'refresh_projected', or 'controlled_refresh_projected'."
        )
    consolidation_result = train_regression_model(
        model=hybrid_model,
        data_config=data_config,
        training_config=resolved_consolidation_training_config,
    )
    final_result = _build_final_result(warm_start_result, consolidation_result)
    return HybridTernaryRunResult(
        warm_start_result=warm_start_result,
        consolidation_result=consolidation_result,
        final_result=final_result,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a hybrid ternary regressor with an STE warm start and a shadow-free consolidation phase."
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
        default="nonlinear_residual",
    )
    parser.add_argument("--nonlinear-scale", type=float, default=18.0)
    parser.add_argument("--nonlinear-pair-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_TERNARY_HIDDEN_DIMS),
    )
    parser.add_argument(
        "--warm-start-epochs",
        type=int,
        default=DEFAULT_HYBRID_WARM_START_EPOCHS,
    )
    parser.add_argument(
        "--consolidation-epochs",
        type=int,
        default=DEFAULT_HYBRID_CONSOLIDATION_EPOCHS,
    )
    parser.add_argument(
        "--warm-start-learning-rate",
        type=float,
        default=DEFAULT_TERNARY_LEARNING_RATE,
    )
    parser.add_argument(
        "--consolidation-learning-rate",
        type=float,
        default=DEFAULT_HYBRID_CONSOLIDATION_LEARNING_RATE,
    )
    parser.add_argument("--threshold-scale", type=float, default=0.5)
    parser.add_argument(
        "--consolidation-variant",
        choices=("shadowfree", "refresh_projected", "controlled_refresh_projected"),
        default="shadowfree",
        help="Which low-bit consolidation model to use after the STE warm start.",
    )
    parser.add_argument(
        "--projection-target-density",
        type=float,
        default=None,
        help="Optionally prune the STE ternary state to this density before the consolidation phase.",
    )
    parser.add_argument(
        "--projection-structure",
        choices=("none", "row_block"),
        default="none",
        help="Optional structured projection rule to apply before the consolidation phase.",
    )
    parser.add_argument(
        "--projection-block-size",
        type=int,
        default=16,
        help="Contiguous block size for structured projection rules such as row_block.",
    )
    parser.add_argument("--initial-density", type=float, default=0.25)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument(
        "--refresh-intervals",
        type=str,
        default=None,
        help="Optional comma-separated per-layer refresh intervals for refresh_projected variants. Must match the hidden layer count.",
    )
    parser.add_argument("--refresh-project-every-n-refreshes", type=int, default=1)
    parser.add_argument(
        "--control-rank",
        type=int,
        default=16,
        help="Low-rank width for the dense control path in the controlled_refresh_projected variant.",
    )
    parser.add_argument(
        "--control-ranks",
        type=str,
        default=None,
        help="Optional comma-separated per-layer control ranks for the controlled_refresh_projected variant. Use 0 to disable control on a layer.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("low_rank", "scalar_gate"),
        default="low_rank",
        help="Dense control mechanism for the controlled_refresh_projected variant.",
    )
    parser.add_argument(
        "--activation-threshold-scale",
        type=float,
        default=0.25,
        help="Threshold scale for ternary hidden-activation quantization in the controlled_refresh_projected variant.",
    )
    parser.add_argument(
        "--disable-lowbit-activations",
        action="store_true",
        help="Keep hidden activations in BF16/FP32 instead of ternary in the controlled_refresh_projected variant.",
    )
    parser.add_argument("--activation-std-multiplier", type=float, default=1.0)
    parser.add_argument("--prune-ratio", type=float, default=0.25)
    parser.add_argument("--flip-multiplier", type=float, default=2.0)
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
    data_config = RegressionDataConfig(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=informative_features,
        noise=args.noise,
        target_kind=args.target_kind,
        nonlinear_scale=args.nonlinear_scale,
        nonlinear_pair_count=args.nonlinear_pair_count,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    warm_start_training_config = TrainingConfig(
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.warm_start_epochs,
        learning_rate=args.warm_start_learning_rate,
        seed=args.seed,
        accelerator=args.accelerator,
        precision=args.precision,
    )
    consolidation_training_config = TrainingConfig(
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.consolidation_epochs,
        learning_rate=args.consolidation_learning_rate,
        seed=args.seed,
        accelerator=args.accelerator,
        precision=args.precision,
    )
    result = train_hybrid_ternary_regression(
        data_config=data_config,
        warm_start_training_config=warm_start_training_config,
        consolidation_training_config=consolidation_training_config,
        threshold_scale=args.threshold_scale,
        consolidation_variant=args.consolidation_variant,
        projection_target_density=args.projection_target_density,
        projection_structure=(
            None if args.projection_structure == "none" else args.projection_structure
        ),
        projection_block_size=(
            None if args.projection_structure == "none" else args.projection_block_size
        ),
        initial_density=args.initial_density,
        update_interval=args.update_interval,
        refresh_interval=args.refresh_interval,
        refresh_intervals=_parse_refresh_intervals(args.refresh_intervals),
        refresh_project_every_n_refreshes=args.refresh_project_every_n_refreshes,
        control_rank=args.control_rank,
        control_ranks=_parse_control_ranks(args.control_ranks),
        control_mode=args.control_mode,
        activation_threshold_scale=args.activation_threshold_scale,
        quantize_hidden_activations=not args.disable_lowbit_activations,
        activation_std_multiplier=args.activation_std_multiplier,
        prune_ratio=args.prune_ratio,
        flip_multiplier=args.flip_multiplier,
    )
    density = result.final_result.model.model.ternary_nonzero_density()
    print(f"Warm-start RMSE: {result.warm_start_result.test_metrics.rmse:.4f}")
    print(f"Consolidated RMSE: {result.final_result.test_metrics.rmse:.4f}")
    print(f"Consolidated density: {density:.4f}")
    print(f"Combined total runtime: {result.final_result.runtime.total_seconds:.4f}s")


if __name__ == "__main__":
    main()
