from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from output_paths import ARTIFACTS_DIRNAME, resolve_output_path
from regression_models import RefreshScheduledProjectedTernaryLinear


@dataclass(slots=True)
class TrainingOpBenchmarkResult:
    mode: str
    batch_size: int
    in_features: int
    out_features: int
    refresh_interval: int
    refresh_target_density: float | None
    iterations: int
    warmup: int
    precision: str
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    post_step_ms: float
    total_ms: float


@dataclass(slots=True)
class ShapeSummary:
    batch_size: int
    in_features: int
    out_features: int
    refresh_interval: int
    precision: str
    dense_total_ms: float
    refresh_nonrefresh_total_ms: float
    refresh_refresh_total_ms: float
    estimated_refresh_mean_step_ms: float
    mean_step_gap_vs_dense_ms: float
    surrogate_overhead_vs_dense_ms: float
    refresh_step_hook_overhead_ms: float
    mean_refresh_hook_overhead_ms: float


def _parse_optional_density(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


def _result_to_dict(result: TrainingOpBenchmarkResult | ShapeSummary) -> dict[str, object]:
    return asdict(result)


def _write_json(records: list[dict[str, object]] | dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _parse_shapes(specs: list[str]) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for spec in specs:
        batch_size, in_features, out_features = (int(part) for part in spec.split(","))
        shapes.append((batch_size, in_features, out_features))
    return shapes


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision in {"bf16", "bf16-mixed", "bf16-true"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision in {"16", "16-mixed", "16-true"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_phase_ms(device: torch.device, fn) -> float:
    _synchronize(device)
    start = time.perf_counter()
    fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def _build_dense_layer(
    *,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, device=device, dtype=torch.float32)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _build_refresh_layer(
    *,
    in_features: int,
    out_features: int,
    refresh_interval: int,
    refresh_target_density: float | None,
    threshold_scale: float,
    device: torch.device,
) -> RefreshScheduledProjectedTernaryLinear:
    layer = RefreshScheduledProjectedTernaryLinear(
        in_features=in_features,
        out_features=out_features,
        threshold_scale=threshold_scale,
        refresh_interval=refresh_interval,
        refresh_target_density=refresh_target_density,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        layer.refresh_cached_state_(force_project=True)
    return layer


def _benchmark_dense_train_step(
    *,
    batch_size: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    precision: str,
    iterations: int,
    warmup: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingOpBenchmarkResult:
    layer = _build_dense_layer(
        in_features=in_features,
        out_features=out_features,
        device=device,
    )
    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    inputs = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    targets = torch.randn(batch_size, out_features, device=device, dtype=torch.float32)
    phase_totals = {"forward_ms": 0.0, "backward_ms": 0.0, "optimizer_ms": 0.0}

    for step_index in range(warmup + iterations):
        optimizer.zero_grad(set_to_none=True)

        forward_state: dict[str, torch.Tensor] = {}

        def run_forward() -> None:
            with _autocast_context(device, precision):
                predictions = layer(inputs)
                forward_state["loss"] = F.mse_loss(predictions, targets)

        forward_ms = _measure_phase_ms(device, run_forward)
        backward_ms = _measure_phase_ms(device, lambda: forward_state["loss"].backward())
        optimizer_ms = _measure_phase_ms(device, optimizer.step)

        if step_index < warmup:
            continue
        phase_totals["forward_ms"] += forward_ms
        phase_totals["backward_ms"] += backward_ms
        phase_totals["optimizer_ms"] += optimizer_ms

    forward_mean = phase_totals["forward_ms"] / iterations
    backward_mean = phase_totals["backward_ms"] / iterations
    optimizer_mean = phase_totals["optimizer_ms"] / iterations
    return TrainingOpBenchmarkResult(
        mode="dense_train_step",
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        refresh_interval=1,
        refresh_target_density=None,
        iterations=iterations,
        warmup=warmup,
        precision=precision,
        forward_ms=forward_mean,
        backward_ms=backward_mean,
        optimizer_ms=optimizer_mean,
        post_step_ms=0.0,
        total_ms=forward_mean + backward_mean + optimizer_mean,
    )


def _benchmark_refresh_train_step(
    *,
    mode: str,
    batch_size: int,
    in_features: int,
    out_features: int,
    refresh_interval: int,
    refresh_target_density: float | None,
    threshold_scale: float,
    device: torch.device,
    precision: str,
    iterations: int,
    warmup: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingOpBenchmarkResult:
    if mode not in {"refresh_nonrefresh_step", "refresh_refresh_step"}:
        raise ValueError(f"Unsupported refresh benchmark mode: {mode}")

    layer = _build_refresh_layer(
        in_features=in_features,
        out_features=out_features,
        refresh_interval=refresh_interval,
        refresh_target_density=refresh_target_density,
        threshold_scale=threshold_scale,
        device=device,
    )
    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    inputs = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    targets = torch.randn(batch_size, out_features, device=device, dtype=torch.float32)
    phase_totals = {
        "forward_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "post_step_ms": 0.0,
    }

    for step_index in range(warmup + iterations):
        optimizer.zero_grad(set_to_none=True)
        if mode == "refresh_nonrefresh_step":
            layer._steps_since_refresh = 0
        else:
            layer._steps_since_refresh = layer.refresh_interval - 1

        forward_state: dict[str, torch.Tensor] = {}

        def run_forward() -> None:
            with _autocast_context(device, precision):
                predictions = layer(inputs)
                forward_state["loss"] = F.mse_loss(predictions, targets)

        forward_ms = _measure_phase_ms(device, run_forward)
        backward_ms = _measure_phase_ms(device, lambda: forward_state["loss"].backward())
        optimizer_ms = _measure_phase_ms(device, optimizer.step)
        post_step_ms = _measure_phase_ms(device, layer.apply_discrete_updates_)

        if step_index < warmup:
            continue
        phase_totals["forward_ms"] += forward_ms
        phase_totals["backward_ms"] += backward_ms
        phase_totals["optimizer_ms"] += optimizer_ms
        phase_totals["post_step_ms"] += post_step_ms

    forward_mean = phase_totals["forward_ms"] / iterations
    backward_mean = phase_totals["backward_ms"] / iterations
    optimizer_mean = phase_totals["optimizer_ms"] / iterations
    post_step_mean = phase_totals["post_step_ms"] / iterations
    return TrainingOpBenchmarkResult(
        mode=mode,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        refresh_interval=refresh_interval,
        refresh_target_density=refresh_target_density,
        iterations=iterations,
        warmup=warmup,
        precision=precision,
        forward_ms=forward_mean,
        backward_ms=backward_mean,
        optimizer_ms=optimizer_mean,
        post_step_ms=post_step_mean,
        total_ms=forward_mean + backward_mean + optimizer_mean + post_step_mean,
    )


def _build_shape_summary(
    dense_result: TrainingOpBenchmarkResult,
    refresh_nonrefresh_result: TrainingOpBenchmarkResult,
    refresh_refresh_result: TrainingOpBenchmarkResult,
) -> ShapeSummary:
    refresh_interval = refresh_nonrefresh_result.refresh_interval
    refresh_step_hook_overhead_ms = (
        refresh_refresh_result.total_ms - refresh_nonrefresh_result.total_ms
    )
    estimated_refresh_mean_step_ms = (
        refresh_nonrefresh_result.total_ms * (refresh_interval - 1)
        + refresh_refresh_result.total_ms
    ) / refresh_interval
    return ShapeSummary(
        batch_size=dense_result.batch_size,
        in_features=dense_result.in_features,
        out_features=dense_result.out_features,
        refresh_interval=refresh_interval,
        precision=dense_result.precision,
        dense_total_ms=dense_result.total_ms,
        refresh_nonrefresh_total_ms=refresh_nonrefresh_result.total_ms,
        refresh_refresh_total_ms=refresh_refresh_result.total_ms,
        estimated_refresh_mean_step_ms=estimated_refresh_mean_step_ms,
        mean_step_gap_vs_dense_ms=estimated_refresh_mean_step_ms - dense_result.total_ms,
        surrogate_overhead_vs_dense_ms=(
            refresh_nonrefresh_result.total_ms - dense_result.total_ms
        ),
        refresh_step_hook_overhead_ms=refresh_step_hook_overhead_ms,
        mean_refresh_hook_overhead_ms=(
            refresh_step_hook_overhead_ms / refresh_interval
        ),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose refresh_projected training-step cost into dense, non-refresh, and refresh-step phases."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["128,256,256", "128,256,128"],
        help="Comma-separated benchmark shapes in the form batch,in,out.",
    )
    parser.add_argument("--refresh-interval", type=int, default=16)
    parser.add_argument(
        "--refresh-target-density",
        type=_parse_optional_density,
        default=0.001,
        help="Density target for refresh projection, or 'none' to disable projection.",
    )
    parser.add_argument("--threshold-scale", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("refresh_projected_training_ops.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("refresh_projected_training_ops.csv"),
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=Path("refresh_projected_training_ops_summary.json"),
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=Path("refresh_projected_training_ops_summary.csv"),
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark refresh_projected training ops.")

    device = torch.device("cuda")
    print("Refresh-projected training-op benchmark")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"refresh_interval={args.refresh_interval} density={args.refresh_target_density} precision={args.precision}"
    )
    print()

    detailed_results: list[TrainingOpBenchmarkResult] = []
    summaries: list[ShapeSummary] = []

    for batch_size, in_features, out_features in _parse_shapes(args.shapes):
        dense_result = _benchmark_dense_train_step(
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
            device=device,
            precision=args.precision,
            iterations=args.iterations,
            warmup=args.warmup,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        refresh_nonrefresh_result = _benchmark_refresh_train_step(
            mode="refresh_nonrefresh_step",
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
            refresh_interval=args.refresh_interval,
            refresh_target_density=args.refresh_target_density,
            threshold_scale=args.threshold_scale,
            device=device,
            precision=args.precision,
            iterations=args.iterations,
            warmup=args.warmup,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        refresh_refresh_result = _benchmark_refresh_train_step(
            mode="refresh_refresh_step",
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
            refresh_interval=args.refresh_interval,
            refresh_target_density=args.refresh_target_density,
            threshold_scale=args.threshold_scale,
            device=device,
            precision=args.precision,
            iterations=args.iterations,
            warmup=args.warmup,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        summary = _build_shape_summary(
            dense_result,
            refresh_nonrefresh_result,
            refresh_refresh_result,
        )

        detailed_results.extend(
            [dense_result, refresh_nonrefresh_result, refresh_refresh_result]
        )
        summaries.append(summary)

        print(f"shape=({batch_size}, {in_features}, {out_features})")
        print(
            f"  dense total={dense_result.total_ms:.4f}ms "
            f"(forward={dense_result.forward_ms:.4f} backward={dense_result.backward_ms:.4f} optimizer={dense_result.optimizer_ms:.4f})"
        )
        print(
            f"  refresh non-refresh total={refresh_nonrefresh_result.total_ms:.4f}ms "
            f"(forward={refresh_nonrefresh_result.forward_ms:.4f} backward={refresh_nonrefresh_result.backward_ms:.4f} "
            f"optimizer={refresh_nonrefresh_result.optimizer_ms:.4f} post={refresh_nonrefresh_result.post_step_ms:.4f})"
        )
        print(
            f"  refresh refresh-step total={refresh_refresh_result.total_ms:.4f}ms "
            f"(forward={refresh_refresh_result.forward_ms:.4f} backward={refresh_refresh_result.backward_ms:.4f} "
            f"optimizer={refresh_refresh_result.optimizer_ms:.4f} post={refresh_refresh_result.post_step_ms:.4f})"
        )
        print(
            f"  estimated refresh mean step={summary.estimated_refresh_mean_step_ms:.4f}ms "
            f"mean_gap_vs_dense={summary.mean_step_gap_vs_dense_ms:.4f}ms "
            f"surrogate_overhead={summary.surrogate_overhead_vs_dense_ms:.4f}ms "
            f"refresh_step_hook_overhead={summary.refresh_step_hook_overhead_ms:.4f}ms "
            f"mean_refresh_hook_overhead={summary.mean_refresh_hook_overhead_ms:.4f}ms"
        )

    detailed_records = [_result_to_dict(result) for result in detailed_results]
    summary_records = [_result_to_dict(summary) for summary in summaries]
    payload = {
        "benchmark_config": {
            "refresh_interval": args.refresh_interval,
            "refresh_target_density": args.refresh_target_density,
            "threshold_scale": args.threshold_scale,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "device": str(device),
            "shapes": args.shapes,
        },
        "results": detailed_records,
        "shape_summaries": summary_records,
    }

    json_out = resolve_output_path(
        args.json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="refresh_projected_training_ops.json",
    )
    csv_out = resolve_output_path(
        args.csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="refresh_projected_training_ops.csv",
    )
    summary_json_out = resolve_output_path(
        args.summary_json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="refresh_projected_training_ops_summary.json",
    )
    summary_csv_out = resolve_output_path(
        args.summary_csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="refresh_projected_training_ops_summary.csv",
    )

    _write_json(payload, json_out)
    _write_csv(detailed_records, csv_out)
    _write_json(summary_records, summary_json_out)
    _write_csv(summary_records, summary_csv_out)

    print()
    print(f"Wrote benchmark JSON to {json_out}")
    print(f"Wrote benchmark CSV to {csv_out}")
    print(f"Wrote benchmark summary JSON to {summary_json_out}")
    print(f"Wrote benchmark summary CSV to {summary_csv_out}")


if __name__ == "__main__":
    main()
