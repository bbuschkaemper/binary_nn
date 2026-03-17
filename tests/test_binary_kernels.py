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
