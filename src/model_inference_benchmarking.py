from __future__ import annotations

import copy
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from regression_experiment import RegressionRunResult
from regression_models import BinaryLinear


@dataclass(slots=True)
class InferenceBenchmarkConfig:
    batch_sizes: tuple[int, ...] = (128,)
    iterations: int = 50
    warmup: int = 10
    seed: int = 123


@dataclass(slots=True)
class ModelInferenceBenchmarkRecord:
    model_name: str
    batch_size: int
    input_dim: int
    hidden_dims: tuple[int, ...]
    use_input_shortcut: bool
    use_triton_packed_inference: bool
    latency_ms: float
    test_loss: float
    test_rmse: float
    test_mae: float
    test_r2: float
    parameter_count: int
    benchmark_device: str


def model_inference_record_to_dict(
    record: ModelInferenceBenchmarkRecord,
) -> dict[str, object]:
    result = asdict(record)
    result["hidden_dims"] = list(record.hidden_dims)
    return result


def dominates_model_inference(
    left: ModelInferenceBenchmarkRecord,
    right: ModelInferenceBenchmarkRecord,
) -> bool:
    return (
        left.test_rmse <= right.test_rmse
        and left.latency_ms <= right.latency_ms
        and (left.test_rmse < right.test_rmse or left.latency_ms < right.latency_ms)
    )


def model_inference_pareto_frontier(
    records: list[ModelInferenceBenchmarkRecord],
) -> list[ModelInferenceBenchmarkRecord]:
    frontier: list[ModelInferenceBenchmarkRecord] = []
    for candidate in records:
        if any(
            dominates_model_inference(other, candidate)
            for other in records
            if other != candidate
        ):
            continue
        frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item.latency_ms, item.test_rmse))


def build_binary_ablation_matrix(
    records: list[ModelInferenceBenchmarkRecord],
) -> list[dict[str, object]]:
    binary_records = [record for record in records if record.model_name == "binary"]
    grouped_by_batch: dict[int, list[ModelInferenceBenchmarkRecord]] = defaultdict(list)
    for record in binary_records:
        grouped_by_batch[record.batch_size].append(record)

    matrix_rows: list[dict[str, object]] = []
    for batch_size in sorted(grouped_by_batch):
        batch_records = grouped_by_batch[batch_size]
        by_variant = {
            (record.use_input_shortcut, record.use_triton_packed_inference): record
            for record in batch_records
        }
        for shortcut_enabled, use_triton in sorted(by_variant):
            record = by_variant[(shortcut_enabled, use_triton)]
            same_shortcut_baseline = by_variant.get((shortcut_enabled, False))
            no_shortcut_baseline = by_variant.get((False, use_triton))
            matrix_rows.append(
                {
                    "batch_size": batch_size,
                    "hidden_dims": list(record.hidden_dims),
                    "use_input_shortcut": shortcut_enabled,
                    "use_triton_packed_inference": use_triton,
                    "latency_ms": record.latency_ms,
                    "test_rmse": record.test_rmse,
                    "test_r2": record.test_r2,
                    "speedup_vs_same_shortcut_no_triton": (
                        same_shortcut_baseline.latency_ms / record.latency_ms
                        if same_shortcut_baseline is not None and record.latency_ms > 0.0
                        else None
                    ),
                    "latency_delta_vs_same_shortcut_no_triton_ms": (
                        record.latency_ms - same_shortcut_baseline.latency_ms
                        if same_shortcut_baseline is not None
                        else None
                    ),
                    "latency_delta_vs_no_shortcut_same_triton_ms": (
                        record.latency_ms - no_shortcut_baseline.latency_ms
                        if no_shortcut_baseline is not None
                        else None
                    ),
                    "rmse_delta_vs_no_shortcut_same_triton": (
                        record.test_rmse - no_shortcut_baseline.test_rmse
                        if no_shortcut_baseline is not None
                        else None
                    ),
                }
            )
    return matrix_rows


def build_model_inference_summary(
    records: list[ModelInferenceBenchmarkRecord],
) -> dict[str, object]:
    grouped_by_batch: dict[int, list[ModelInferenceBenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped_by_batch[record.batch_size].append(record)

    by_batch_size: list[dict[str, object]] = []
    for batch_size in sorted(grouped_by_batch):
        batch_records = grouped_by_batch[batch_size]
        fastest = sorted(batch_records, key=lambda item: (item.latency_ms, item.test_rmse))[:5]
        best_rmse = sorted(batch_records, key=lambda item: (item.test_rmse, item.latency_ms))[:5]
        frontier = model_inference_pareto_frontier(batch_records)
        by_batch_size.append(
            {
                "batch_size": batch_size,
                "fastest_candidates": [
                    model_inference_record_to_dict(record) for record in fastest
                ],
                "best_rmse_candidates": [
                    model_inference_record_to_dict(record) for record in best_rmse
                ],
                "pareto_frontier": [
                    model_inference_record_to_dict(record) for record in frontier
                ],
            }
        )

    return {
        "benchmark_device": records[0].benchmark_device if records else "unknown",
        "candidate_count": len(records),
        "by_batch_size": by_batch_size,
        "binary_ablation_matrix": build_binary_ablation_matrix(records),
    }


def write_model_inference_benchmark_json(
    records: list[ModelInferenceBenchmarkRecord],
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(
            [model_inference_record_to_dict(record) for record in records],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_model_inference_benchmark_csv(
    records: list[ModelInferenceBenchmarkRecord],
    output_path: Path,
) -> None:
    serialized = [model_inference_record_to_dict(record) for record in records]
    if not serialized:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0].keys()))
        writer.writeheader()
        writer.writerows(serialized)


def write_model_inference_summary_json(
    records: list[ModelInferenceBenchmarkRecord],
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(build_model_inference_summary(records), indent=2) + "\n",
        encoding="utf-8",
    )


def write_model_inference_frontier_csv(
    records: list[ModelInferenceBenchmarkRecord],
    output_path: Path,
) -> None:
    frontier_records: list[dict[str, object]] = []
    grouped_by_batch: dict[int, list[ModelInferenceBenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped_by_batch[record.batch_size].append(record)

    for batch_size in sorted(grouped_by_batch):
        for frontier_record in model_inference_pareto_frontier(grouped_by_batch[batch_size]):
            frontier_records.append(model_inference_record_to_dict(frontier_record))

    if not frontier_records:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier_records[0].keys()))
        writer.writeheader()
        writer.writerows(frontier_records)


def _set_binary_linear_triton_usage(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, BinaryLinear):
            module.use_triton_packed_inference = enabled


def _time_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    iterations: int,
    warmup: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def _make_benchmark_inputs(
    batch_size: int,
    input_dim: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        batch_size, input_dim, device=device, dtype=torch.float32, generator=generator
    )


def _resolve_benchmark_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _clone_model_for_benchmark(
    run_result: RegressionRunResult,
    device: torch.device,
) -> torch.nn.Module:
    model = copy.deepcopy(run_result.model.model)
    model.to(device)
    model.eval()
    return model


def benchmark_regression_run_result(
    run_result: RegressionRunResult,
    model_name: str,
    use_input_shortcut: bool,
    benchmark_config: InferenceBenchmarkConfig,
    *,
    benchmark_triton_variants: bool,
) -> list[ModelInferenceBenchmarkRecord]:
    device = _resolve_benchmark_device()
    model = _clone_model_for_benchmark(run_result, device)
    use_triton_options = (
        (False, True)
        if benchmark_triton_variants and device.type == "cuda"
        else (False,)
    )
    records: list[ModelInferenceBenchmarkRecord] = []

    for batch_size in benchmark_config.batch_sizes:
        inputs = _make_benchmark_inputs(
            batch_size=batch_size,
            input_dim=run_result.data_config.n_features,
            device=device,
            seed=benchmark_config.seed + batch_size,
        )
        for use_triton in use_triton_options:
            _set_binary_linear_triton_usage(model, use_triton)
            latency_ms = _time_model(
                model=model,
                inputs=inputs,
                iterations=benchmark_config.iterations,
                warmup=benchmark_config.warmup,
            )
            records.append(
                ModelInferenceBenchmarkRecord(
                    model_name=model_name,
                    batch_size=batch_size,
                    input_dim=run_result.data_config.n_features,
                    hidden_dims=run_result.training_config.hidden_dims,
                    use_input_shortcut=use_input_shortcut,
                    use_triton_packed_inference=use_triton,
                    latency_ms=latency_ms,
                    test_loss=run_result.test_loss,
                    test_rmse=run_result.test_metrics.rmse,
                    test_mae=run_result.test_metrics.mae,
                    test_r2=run_result.test_metrics.r2,
                    parameter_count=run_result.runtime.parameter_count,
                    benchmark_device=str(device),
                )
            )
    return records
