from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from binary_kernels import (
    pack_binary_weight,
    packed_binary_linear_reference,
    packed_binary_linear_triton,
    unpack_binary_weight,
)
from regression_models import BinaryLinear
from benchmark_packed_binary_kernels import (
    BenchmarkResult,
    benchmark_result_frontier,
    build_benchmark_summary,
)


def test_pack_and_unpack_binary_weight_round_trip_sign_and_scale() -> None:
    weight = torch.tensor(
        [[1.0, -2.0, 0.5, -0.5], [-1.0, -1.0, 2.0, 2.0]],
        dtype=torch.float32,
    )

    packed = pack_binary_weight(weight)
    unpacked = unpack_binary_weight(packed)
    expected = torch.sign(weight).masked_fill(weight == 0, 1.0) * weight.abs().mean(
        dim=1, keepdim=True
    )

    assert torch.allclose(unpacked, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton kernel validation.",
)
def test_triton_packed_binary_linear_matches_reference() -> None:
    inputs = torch.randn(64, 96, device="cuda", dtype=torch.float32)
    weight = torch.randn(48, 96, device="cuda", dtype=torch.float32)
    bias = torch.randn(48, device="cuda", dtype=torch.float32)
    packed = pack_binary_weight(weight)

    reference = packed_binary_linear_reference(inputs, packed, bias)
    triton_output = packed_binary_linear_triton(inputs, packed, bias)
    max_abs_diff = float(torch.max(torch.abs(triton_output - reference)).item())

    assert max_abs_diff < 2e-2


def test_kernel_benchmark_summary_includes_frontier() -> None:
    results = [
        BenchmarkResult(256, 1024, 1024, 1.5, 0.75, 2.0, 0.0020),
        BenchmarkResult(512, 2048, 2048, 2.4, 1.0, 2.4, 0.0030),
        BenchmarkResult(128, 512, 512, 0.9, 0.6, 1.5, 0.0010),
    ]

    frontier = benchmark_result_frontier(results)
    summary = build_benchmark_summary(results)

    assert len(frontier) >= 2
    assert summary["candidate_count"] == 3
    assert len(summary["pareto_frontier"]) == len(frontier)
    assert summary["best_speedup_candidates"][0]["speedup"] == 2.4


def test_binary_linear_disables_triton_for_known_losing_shape() -> None:
    layer = BinaryLinear(1024, 1024)
    inputs = torch.randn(16384, 1024)

    assert layer._known_triton_losing_shape(inputs) is True
    assert layer._should_use_packed_inference(inputs) is False


def test_binary_linear_keeps_triton_enabled_for_smaller_batches() -> None:
    layer = BinaryLinear(1024, 1024)
    inputs = torch.randn(2048, 1024, device="cpu")

    assert layer._known_triton_losing_shape(inputs) is False
