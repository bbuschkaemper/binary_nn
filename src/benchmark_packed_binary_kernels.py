from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from binary_kernels import (
    pack_binary_weight,
    packed_binary_linear_reference,
    packed_binary_linear_triton,
    supports_triton_packed_binary_linear,
)
from output_paths import ARTIFACTS_DIRNAME, resolve_output_path


@dataclass(slots=True)
class BenchmarkResult:
    batch_size: int
    in_features: int
    out_features: int
    torch_ms: float
    triton_ms: float
    speedup: float
    max_abs_diff: float


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, object]:
    return asdict(result)


def dominates_benchmark_result(left: BenchmarkResult, right: BenchmarkResult) -> bool:
    return (
        left.speedup >= right.speedup
        and left.max_abs_diff <= right.max_abs_diff
        and (left.speedup > right.speedup or left.max_abs_diff < right.max_abs_diff)
    )


def benchmark_result_frontier(
    results: list[BenchmarkResult],
) -> list[BenchmarkResult]:
    frontier: list[BenchmarkResult] = []
    for candidate in results:
        if any(
            dominates_benchmark_result(other, candidate)
            for other in results
            if other != candidate
        ):
            continue
        frontier.append(candidate)
    return sorted(frontier, key=lambda item: (-item.speedup, item.max_abs_diff))


def build_benchmark_summary(results: list[BenchmarkResult]) -> dict[str, object]:
    best_speedup = sorted(results, key=lambda item: (-item.speedup, item.max_abs_diff))[:5]
    lowest_error = sorted(results, key=lambda item: (item.max_abs_diff, -item.speedup))[:5]
    frontier = benchmark_result_frontier(results)
    return {
        "candidate_count": len(results),
        "best_speedup_candidates": [
            benchmark_result_to_dict(result) for result in best_speedup
        ],
        "lowest_error_candidates": [
            benchmark_result_to_dict(result) for result in lowest_error
        ],
        "pareto_frontier": [benchmark_result_to_dict(result) for result in frontier],
    }


def _write_json(
    records: list[dict[str, object]] | dict[str, object], output_path: Path
) -> None:
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _time_callable(fn, iterations: int, warmup: int) -> tuple[torch.Tensor, float]:
    output = None
    for _ in range(warmup):
        output = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
    assert output is not None
    return output, elapsed_ms


def _parse_shapes(specs: list[str]) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for spec in specs:
        batch_size, in_features, out_features = (int(part) for part in spec.split(","))
        shapes.append((batch_size, in_features, out_features))
    return shapes


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark packed Triton binary inference kernels against the PyTorch reference path."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["256,1024,1024", "512,2048,2048", "1024,4096,4096"],
        help="Comma-separated benchmark shapes in the form batch,in,out.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("packed_binary_kernel_benchmark.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("packed_binary_kernel_benchmark.csv"),
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=Path("packed_binary_kernel_benchmark_summary.json"),
    )
    parser.add_argument(
        "--frontier-csv-out",
        type=Path,
        default=Path("packed_binary_kernel_benchmark_frontier.csv"),
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to benchmark the Triton packed binary kernels."
        )

    device = torch.device("cuda")
    if not supports_triton_packed_binary_linear(torch.randn(1, 1, device=device)):
        raise RuntimeError(
            "Triton packed binary kernels are not available in the current environment."
        )

    print("Packed Triton binary benchmark")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print()

    results: list[BenchmarkResult] = []

    for batch_size, in_features, out_features in _parse_shapes(args.shapes):
        inputs = torch.randn(
            batch_size, in_features, device=device, dtype=torch.float32
        )
        weight = torch.empty(
            out_features, in_features, device=device, dtype=torch.float32
        )
        torch.nn.init.xavier_uniform_(weight)
        bias = torch.zeros(out_features, device=device, dtype=torch.float32)
        packed_weight = pack_binary_weight(weight)

        torch_output, torch_ms = _time_callable(
            lambda: packed_binary_linear_reference(inputs, packed_weight, bias),
            iterations=args.iterations,
            warmup=args.warmup,
        )
        triton_output, triton_ms = _time_callable(
            lambda: packed_binary_linear_triton(inputs, packed_weight, bias),
            iterations=args.iterations,
            warmup=args.warmup,
        )
        max_abs_diff = float(torch.max(torch.abs(torch_output - triton_output)).item())
        speedup = torch_ms / triton_ms if triton_ms > 0.0 else float("inf")

        result = BenchmarkResult(
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
            torch_ms=torch_ms,
            triton_ms=triton_ms,
            speedup=speedup,
            max_abs_diff=max_abs_diff,
        )
        results.append(result)
        print(
            f"shape=({result.batch_size}, {result.in_features}, {result.out_features}) "
            f"torch={result.torch_ms:.4f}ms triton={result.triton_ms:.4f}ms "
            f"speedup={result.speedup:.2f}x max_abs_diff={result.max_abs_diff:.6f}"
        )

    serialized = [benchmark_result_to_dict(result) for result in results]
    summary = build_benchmark_summary(results)
    json_out = resolve_output_path(
        args.json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="packed_binary_kernel_benchmark.json",
    )
    csv_out = resolve_output_path(
        args.csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="packed_binary_kernel_benchmark.csv",
    )
    summary_json_out = resolve_output_path(
        args.summary_json_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="packed_binary_kernel_benchmark_summary.json",
    )
    frontier_csv_out = resolve_output_path(
        args.frontier_csv_out,
        default_subdir=ARTIFACTS_DIRNAME,
        default_name="packed_binary_kernel_benchmark_frontier.csv",
    )

    _write_json(serialized, json_out)
    _write_csv(serialized, csv_out)
    _write_json(summary, summary_json_out)
    frontier_records = summary["pareto_frontier"]
    assert isinstance(frontier_records, list)
    _write_csv(frontier_records, frontier_csv_out)

    print()
    print(f"Wrote kernel benchmark JSON to {json_out}")
    print(f"Wrote kernel benchmark CSV to {csv_out}")
    print(f"Wrote kernel benchmark summary to {summary_json_out}")
    print(f"Wrote kernel benchmark frontier CSV to {frontier_csv_out}")


if __name__ == "__main__":
    main()
