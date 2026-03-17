from __future__ import annotations

import copy
import csv
import json
import time
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
