from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from binary_kernels import (
    pack_binary_weight,
    packed_binary_linear_reference,
    packed_binary_linear_triton,
    supports_triton_packed_binary_linear,
)


@dataclass(slots=True)
class BenchmarkResult:
    batch_size: int
    in_features: int
    out_features: int
    torch_ms: float
    triton_ms: float
    speedup: float
    max_abs_diff: float


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
        print(
            f"shape=({result.batch_size}, {result.in_features}, {result.out_features}) "
            f"torch={result.torch_ms:.4f}ms triton={result.triton_ms:.4f}ms "
            f"speedup={result.speedup:.2f}x max_abs_diff={result.max_abs_diff:.6f}"
        )


if __name__ == "__main__":
    main()
