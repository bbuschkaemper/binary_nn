from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportInvalidTypeForm=false

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
else:
    try:
        import triton
        import triton.language as tl

        TRITON_AVAILABLE = True
    except ImportError:  # pragma: no cover - exercised only when Triton is absent.
        triton = None
        tl = None
        TRITON_AVAILABLE = False


PACKING_FACTOR = 32


@dataclass(slots=True)
class PackedBinaryWeight:
    packed_signs: torch.Tensor
    scales: torch.Tensor
    in_features: int


def pack_binary_weight(weight: torch.Tensor) -> PackedBinaryWeight:
    if weight.ndim != 2:
        raise ValueError(
            f"Expected a 2D weight tensor, got shape {tuple(weight.shape)}."
        )

    out_features, in_features = weight.shape
    positive_mask = (weight.detach() >= 0).to(torch.int64)
    padding = (-in_features) % PACKING_FACTOR
    if padding:
        positive_mask = F.pad(positive_mask, (0, padding))

    words_per_row = positive_mask.shape[1] // PACKING_FACTOR
    positive_mask = positive_mask.view(out_features, words_per_row, PACKING_FACTOR)
    bit_offsets = (
        1 << torch.arange(PACKING_FACTOR, device=weight.device, dtype=torch.int64)
    ).view(1, 1, -1)
    packed_signs = (
        torch.sum(positive_mask * bit_offsets, dim=-1).to(torch.int32).contiguous()
    )
    scales = weight.detach().abs().mean(dim=1).to(torch.float32).contiguous()
    return PackedBinaryWeight(
        packed_signs=packed_signs,
        scales=scales,
        in_features=in_features,
    )


def unpack_binary_weight(packed_weight: PackedBinaryWeight) -> torch.Tensor:
    packed_signs = packed_weight.packed_signs
    out_features, words_per_row = packed_signs.shape
    bit_offsets = torch.arange(
        PACKING_FACTOR, device=packed_signs.device, dtype=torch.int32
    )
    bits = ((packed_signs.unsqueeze(-1) >> bit_offsets) & 1).to(torch.float32)
    signs = bits.mul(2.0).sub(1.0).view(out_features, words_per_row * PACKING_FACTOR)
    signs = signs[:, : packed_weight.in_features]
    return signs * packed_weight.scales.unsqueeze(1)


def packed_binary_linear_reference(
    inputs: torch.Tensor,
    packed_weight: PackedBinaryWeight,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(inputs, unpack_binary_weight(packed_weight), bias)


def supports_triton_packed_binary_linear(inputs: torch.Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and inputs.is_cuda
        and inputs.ndim == 2
        and inputs.dtype in {torch.float16, torch.float32, torch.bfloat16}
    )


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _packed_binary_linear_kernel(
        input_ptr,
        packed_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K,
        stride_input_m,
        stride_input_k,
        stride_packed_n,
        stride_packed_word,
        stride_output_m,
        stride_output_n,
        HAS_BIAS: tl.constexpr,
        PACK_BITS: tl.constexpr,
        PACKED_WORDS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for word_idx in tl.static_range(0, PACKED_WORDS):
            offs_k = word_idx * PACK_BITS + tl.arange(0, PACK_BITS)
            valid_k = offs_k < K
            input_block = tl.load(
                input_ptr
                + offs_m[:, None] * stride_input_m
                + offs_k[None, :] * stride_input_k,
                mask=mask_m[:, None] & valid_k[None, :],
                other=0.0,
            ).to(tl.float32)

            packed_words = tl.load(
                packed_ptr + offs_n * stride_packed_n + word_idx * stride_packed_word,
                mask=mask_n,
                other=0,
            )
            bit_offsets = tl.arange(0, PACK_BITS)
            bits = (packed_words[None, :] >> bit_offsets[:, None]) & 1
            signs = tl.where(bits != 0, 1.0, -1.0).to(tl.float32)
            signs = tl.where(valid_k[:, None], signs, 0.0)
            acc += tl.dot(input_block, signs)

        scales = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        tl.store(
            output_ptr
            + offs_m[:, None] * stride_output_m
            + offs_n[None, :] * stride_output_n,
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def packed_binary_linear_triton(
    inputs: torch.Tensor,
    packed_weight: PackedBinaryWeight,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if not supports_triton_packed_binary_linear(inputs):
        return packed_binary_linear_reference(inputs, packed_weight, bias)

    if inputs.shape[1] != packed_weight.in_features:
        raise ValueError(
            f"Input feature mismatch: expected {packed_weight.in_features}, got {inputs.shape[1]}."
        )

    packed_signs = packed_weight.packed_signs
    output = torch.empty(
        (inputs.shape[0], packed_signs.shape[0]),
        device=inputs.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(inputs.shape[0], meta["BLOCK_M"]),
        triton.cdiv(packed_signs.shape[0], meta["BLOCK_N"]),
    )
    _packed_binary_linear_kernel[grid](
        inputs,
        packed_signs,
        packed_weight.scales,
        bias if bias is not None else packed_weight.scales,
        output,
        inputs.shape[0],
        packed_signs.shape[0],
        packed_weight.in_features,
        inputs.stride(0),
        inputs.stride(1),
        packed_signs.stride(0),
        packed_signs.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=bias is not None,
        PACK_BITS=PACKING_FACTOR,
        PACKED_WORDS=packed_signs.shape[1],
    )
    return output.to(inputs.dtype)
