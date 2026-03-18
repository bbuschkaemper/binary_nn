from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from output_paths import ARTIFACTS_DIRNAME, resolve_output_path
from regression_data import RegressionDataConfig
from regression_experiment import RegressionRunResult, TrainingConfig
from regression_models import ShadowFreeTernaryLinear, TernaryLinear
from run_regression_baseline import train_regression_baseline
from run_regression_comparison import regression_run_result_to_dict
from run_hybrid_ternary_regression import train_hybrid_ternary_regression
from run_shadowfree_ternary_regression import train_shadowfree_ternary_regression
from run_ternary_regression import train_ternary_regression


@dataclass(slots=True)
class CpuInferenceRecord:
    model_name: str
    model_family: str
    batch_size: int
    use_sparse_cpu_inference: bool
    latency_ms: float
    test_rmse: float
    test_r2: float
    parameter_count: int
    ternary_nonzero_density: float | None
    benchmark_device: str


def cpu_inference_record_to_dict(record: CpuInferenceRecord) -> dict[str, object]:
    return asdict(record)


def _parse_hidden_dims(spec: str) -> tuple[int, ...]:
    return tuple(int(part) for part in spec.split(",") if part.strip())


def _resolve_default_accelerator(requested: str | None) -> str:
    if requested not in {None, "auto"}:
        return requested
    return "gpu" if torch.cuda.is_available() else "cpu"


def _resolve_default_precision(accelerator: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    return "bf16-mixed" if accelerator == "gpu" else "32-true"


def _set_sparse_cpu_inference(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, (ShadowFreeTernaryLinear, TernaryLinear)):
            module.use_cpu_sparse_inference = enabled


def _clone_model_for_cpu_benchmark(run_result: RegressionRunResult) -> torch.nn.Module:
    model = copy.deepcopy(run_result.model.model)
    model.to("cpu")
    model.eval()
    return model


def _time_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    iterations: int,
    warmup: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
    return (time.perf_counter() - start) * 1000.0 / iterations


def _ternary_nonzero_density(model: torch.nn.Module) -> float | None:
    density_fn = getattr(model, "ternary_nonzero_density", None)
    if callable(density_fn):
        return float(density_fn())
    return None


def benchmark_model_on_cpu(
    run_result: RegressionRunResult,
    *,
    model_name: str,
    model_family: str,
    batch_sizes: tuple[int, ...],
    iterations: int,
    warmup: int,
    benchmark_sparse_variants: bool,
) -> list[CpuInferenceRecord]:
    model = _clone_model_for_cpu_benchmark(run_result)
    has_sparse_toggle = any(
        isinstance(module, (ShadowFreeTernaryLinear, TernaryLinear))
        for module in model.modules()
    )
    sparse_options = (
        (False, True) if benchmark_sparse_variants and has_sparse_toggle else (False,)
    )
    records: list[CpuInferenceRecord] = []

    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, run_result.data_config.n_features)
        for use_sparse in sparse_options:
            _set_sparse_cpu_inference(model, use_sparse)
            latency_ms = _time_model(
                model=model,
                inputs=inputs,
                iterations=iterations,
                warmup=warmup,
            )
            records.append(
                CpuInferenceRecord(
                    model_name=model_name,
                    model_family=model_family,
                    batch_size=batch_size,
                    use_sparse_cpu_inference=use_sparse,
                    latency_ms=latency_ms,
                    test_rmse=run_result.test_metrics.rmse,
                    test_r2=run_result.test_metrics.r2,
                    parameter_count=run_result.runtime.parameter_count,
                    ternary_nonzero_density=_ternary_nonzero_density(model),
                    benchmark_device="cpu",
                )
            )
    return records


def _best_sparse_speedup(records: list[CpuInferenceRecord]) -> dict[int, float]:
    dense_by_batch = {
        record.batch_size: record
        for record in records
        if record.model_name == "dense"
    }
    sparse_by_batch = {
        record.batch_size: record
        for record in records
        if record.model_name != "dense" and record.use_sparse_cpu_inference
    }
    speedups: dict[int, float] = {}
    for batch_size, dense_record in dense_by_batch.items():
        sparse_record = sparse_by_batch.get(batch_size)
        if sparse_record is None or sparse_record.latency_ms <= 0.0:
            continue
        speedups[batch_size] = dense_record.latency_ms / sparse_record.latency_ms
    return speedups


def _write_cpu_inference_csv(
    records: list[CpuInferenceRecord], output_path: Path
) -> None:
    serialized = [cpu_inference_record_to_dict(record) for record in records]
    if not serialized:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0].keys()))
        writer.writeheader()
        writer.writerows(serialized)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a dense BF16 baseline against a ternary research prototype."
    )
    parser.add_argument(
        "--model-family",
        choices=("shadowfree", "ste", "hybrid", "projected"),
        default="shadowfree",
        help="Which ternary research model to compare against the dense baseline.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-hidden-dims", type=str, default="64,32")
    parser.add_argument("--ternary-hidden-dims", type=str, default="64")
    parser.add_argument("--dense-epochs", type=int, default=75)
    parser.add_argument("--ternary-epochs", type=int, default=75)
    parser.add_argument("--dense-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ternary-learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--dense-accelerator",
        choices=("auto", "cpu", "gpu"),
        default=None,
    )
    parser.add_argument(
        "--ternary-accelerator",
        choices=("auto", "cpu", "gpu"),
        default=None,
    )
    parser.add_argument("--dense-precision", type=str, default=None)
    parser.add_argument("--ternary-precision", type=str, default=None)
    parser.add_argument(
        "--cpu-benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[32, 128, 512],
    )
    parser.add_argument("--cpu-benchmark-iterations", type=int, default=200)
    parser.add_argument("--cpu-benchmark-warmup", type=int, default=20)
    parser.add_argument("--initial-density", type=float, default=0.25)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--activation-std-multiplier", type=float, default=0.5)
    parser.add_argument("--prune-ratio", type=float, default=0.5)
    parser.add_argument("--flip-multiplier", type=float, default=1.5)
    parser.add_argument("--threshold-scale", type=float, default=0.5)
    parser.add_argument("--warm-start-epochs", type=int, default=50)
    parser.add_argument("--consolidation-epochs", type=int, default=25)
    parser.add_argument("--consolidation-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--projection-target-density",
        type=float,
        default=0.35,
        help="Target density for projected STE-to-shadow-free handoff.",
    )
    parser.add_argument(
        "--projected-update-interval",
        type=int,
        default=100000,
        help="Discrete-update interval for the projected family. Use a large value to freeze ternary state during recovery training.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("ternary_research_comparison.json"),
    )
    parser.add_argument(
        "--cpu-csv-out",
        type=Path,
        default=Path("ternary_research_cpu_benchmark.csv"),
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    dense_accelerator = _resolve_default_accelerator(args.dense_accelerator)
    ternary_accelerator = _resolve_default_accelerator(args.ternary_accelerator)
    dense_precision = _resolve_default_precision(
        dense_accelerator, args.dense_precision
    )
    ternary_precision = _resolve_default_precision(
        ternary_accelerator, args.ternary_precision
    )

    if dense_accelerator == "gpu" or ternary_accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

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
    dense_training_config = TrainingConfig(
        hidden_dims=_parse_hidden_dims(args.dense_hidden_dims),
        epochs=args.dense_epochs,
        learning_rate=args.dense_learning_rate,
        seed=args.seed,
        accelerator=dense_accelerator,
        precision=dense_precision,
    )
    ternary_hidden_dims_spec = args.ternary_hidden_dims
    if args.model_family in {"ste", "hybrid", "projected"} and ternary_hidden_dims_spec == "64":
        ternary_hidden_dims_spec = "64,32"

    ternary_training_config = TrainingConfig(
        hidden_dims=_parse_hidden_dims(ternary_hidden_dims_spec),
        epochs=args.ternary_epochs,
        learning_rate=args.ternary_learning_rate,
        seed=args.seed,
        accelerator=ternary_accelerator,
        precision=ternary_precision,
    )

    dense_result = train_regression_baseline(
        data_config=data_config,
        training_config=dense_training_config,
    )
    family_details_key: str | None = None
    family_details: dict[str, object] | None = None
    if args.model_family == "shadowfree":
        ternary_result = train_shadowfree_ternary_regression(
            data_config=data_config,
            training_config=ternary_training_config,
            use_input_shortcut=True,
            initial_density=args.initial_density,
            update_interval=args.update_interval,
            activation_std_multiplier=args.activation_std_multiplier,
            prune_ratio=args.prune_ratio,
            flip_multiplier=args.flip_multiplier,
        )
        ternary_model_name = "shadowfree_ternary"
    elif args.model_family == "ste":
        ternary_result = train_ternary_regression(
            data_config=data_config,
            training_config=ternary_training_config,
            use_input_shortcut=True,
            threshold_scale=args.threshold_scale,
        )
        ternary_model_name = "ste_ternary"
    elif args.model_family == "hybrid":
        warm_start_training_config = TrainingConfig(
            hidden_dims=ternary_training_config.hidden_dims,
            epochs=args.warm_start_epochs,
            learning_rate=args.ternary_learning_rate,
            seed=args.seed,
            accelerator=ternary_accelerator,
            precision=ternary_precision,
        )
        consolidation_training_config = TrainingConfig(
            hidden_dims=ternary_training_config.hidden_dims,
            epochs=args.consolidation_epochs,
            learning_rate=args.consolidation_learning_rate,
            seed=args.seed,
            accelerator=ternary_accelerator,
            precision=ternary_precision,
        )
        hybrid_run = train_hybrid_ternary_regression(
            data_config=data_config,
            warm_start_training_config=warm_start_training_config,
            consolidation_training_config=consolidation_training_config,
            use_input_shortcut=True,
            threshold_scale=args.threshold_scale,
            initial_density=args.initial_density,
            update_interval=args.update_interval,
            activation_std_multiplier=args.activation_std_multiplier,
            prune_ratio=args.prune_ratio,
            flip_multiplier=args.flip_multiplier,
        )
        ternary_result = hybrid_run.final_result
        ternary_model_name = "hybrid_ternary"
        ternary_training_config = consolidation_training_config
        family_details_key = "hybrid_details"
        family_details = {
            "warm_start_training_config": asdict(warm_start_training_config),
            "consolidation_training_config": asdict(consolidation_training_config),
            "warm_start_result": regression_run_result_to_dict(
                hybrid_run.warm_start_result
            ),
            "consolidation_result": regression_run_result_to_dict(
                hybrid_run.consolidation_result
            ),
        }
    else:
        warm_start_training_config = TrainingConfig(
            hidden_dims=ternary_training_config.hidden_dims,
            epochs=args.warm_start_epochs,
            learning_rate=args.ternary_learning_rate,
            seed=args.seed,
            accelerator=ternary_accelerator,
            precision=ternary_precision,
        )
        consolidation_training_config = TrainingConfig(
            hidden_dims=ternary_training_config.hidden_dims,
            epochs=args.consolidation_epochs,
            learning_rate=args.consolidation_learning_rate,
            seed=args.seed,
            accelerator=ternary_accelerator,
            precision=ternary_precision,
        )
        projected_run = train_hybrid_ternary_regression(
            data_config=data_config,
            warm_start_training_config=warm_start_training_config,
            consolidation_training_config=consolidation_training_config,
            use_input_shortcut=True,
            threshold_scale=args.threshold_scale,
            projection_target_density=args.projection_target_density,
            initial_density=args.initial_density,
            update_interval=args.projected_update_interval,
            activation_std_multiplier=args.activation_std_multiplier,
            prune_ratio=args.prune_ratio,
            flip_multiplier=args.flip_multiplier,
        )
        ternary_result = projected_run.final_result
        ternary_model_name = "projected_ternary"
        ternary_training_config = consolidation_training_config
        family_details_key = "projection_details"
        family_details = {
            "projection_target_density": args.projection_target_density,
            "projected_update_interval": args.projected_update_interval,
            "warm_start_training_config": asdict(warm_start_training_config),
            "consolidation_training_config": asdict(consolidation_training_config),
            "warm_start_result": regression_run_result_to_dict(
                projected_run.warm_start_result
            ),
            "consolidation_result": regression_run_result_to_dict(
                projected_run.consolidation_result
            ),
        }

    cpu_records = benchmark_model_on_cpu(
        dense_result,
        model_name="dense",
        model_family="dense",
        batch_sizes=tuple(args.cpu_benchmark_batch_sizes),
        iterations=args.cpu_benchmark_iterations,
        warmup=args.cpu_benchmark_warmup,
        benchmark_sparse_variants=False,
    )
    cpu_records.extend(
        benchmark_model_on_cpu(
            ternary_result,
            model_name=ternary_model_name,
            model_family=args.model_family,
            batch_sizes=tuple(args.cpu_benchmark_batch_sizes),
            iterations=args.cpu_benchmark_iterations,
            warmup=args.cpu_benchmark_warmup,
            benchmark_sparse_variants=True,
        )
    )

    best_sparse_speedup = _best_sparse_speedup(cpu_records)
    result = {
        "model_family": args.model_family,
        "data_config": asdict(data_config),
        "dense_training_config": asdict(dense_training_config),
        "ternary_training_config": asdict(ternary_training_config),
        "dense_result": regression_run_result_to_dict(dense_result),
        "ternary_result": regression_run_result_to_dict(ternary_result),
        "cpu_inference_records": [
            cpu_inference_record_to_dict(record) for record in cpu_records
        ],
        "best_sparse_speedup_vs_dense_by_batch": best_sparse_speedup,
    }
    if family_details_key is not None and family_details is not None:
        result[family_details_key] = family_details

    json_out = resolve_output_path(
        args.json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="ternary_research_comparison.json",
    )
    cpu_csv_out = resolve_output_path(
        args.cpu_csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="ternary_research_cpu_benchmark.csv",
    )
    json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_cpu_inference_csv(cpu_records, cpu_csv_out)

    print("Dense baseline")
    print(
        f"  precision={dense_precision} accelerator={dense_accelerator} rmse={dense_result.test_metrics.rmse:.4f} "
        f"total={dense_result.runtime.total_seconds:.4f}s"
    )
    print(f"{ternary_model_name}")
    print(
        f"  precision={ternary_precision} accelerator={ternary_accelerator} rmse={ternary_result.test_metrics.rmse:.4f} "
        f"total={ternary_result.runtime.total_seconds:.4f}s density={_ternary_nonzero_density(ternary_result.model.model)}"
    )
    print("CPU inference benchmark")
    for record in cpu_records:
        print(
            f"  model={record.model_name:<18} batch={record.batch_size:<5} sparse={record.use_sparse_cpu_inference} "
            f"latency={record.latency_ms:.4f}ms rmse={record.test_rmse:.4f}"
        )
    if best_sparse_speedup:
        print("Best sparse speedup vs dense by batch")
        for batch_size, speedup in sorted(best_sparse_speedup.items()):
            print(f"  batch={batch_size:<5} speedup={speedup:.3f}x")
    print(f"Wrote comparison JSON to {json_out}")
    print(f"Wrote CPU benchmark CSV to {cpu_csv_out}")


if __name__ == "__main__":
    main()
