from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class IndexedTernaryWeight:
    indices: torch.Tensor
    signs: torch.Tensor
    scales: torch.Tensor
    in_features: int


@dataclass(slots=True)
class PackedTernaryLookupWeight:
    positive_masks: torch.Tensor
    negative_masks: torch.Tensor
    scales: torch.Tensor
    in_features: int
    block_size: int

    @property
    def padded_in_features(self) -> int:
        return int(self.positive_masks.shape[1] * self.block_size)


def _resolve_scales(weight: torch.Tensor, scales: torch.Tensor | None) -> torch.Tensor:
    out_features = weight.shape[0]
    if scales is None:
        return torch.ones(out_features, device=weight.device, dtype=torch.float32)

    resolved_scales = scales.reshape(-1).detach().to(torch.float32)
    if resolved_scales.shape[0] != out_features:
        raise ValueError(
            f"Scale shape mismatch: expected {out_features}, got {resolved_scales.shape[0]}."
        )
    return resolved_scales


def pack_ternary_weight(
    weight: torch.Tensor,
    scales: torch.Tensor | None = None,
) -> IndexedTernaryWeight:
    if weight.ndim != 2:
        raise ValueError(
            f"Expected a 2D ternary weight tensor, got shape {tuple(weight.shape)}."
        )

    sign_weight = weight.detach().to(torch.int8)
    out_features, in_features = sign_weight.shape
    nonzero_counts = (sign_weight != 0).sum(dim=1)
    max_nonzero = int(nonzero_counts.max().item()) if nonzero_counts.numel() else 0

    indices = torch.zeros(
        (out_features, max_nonzero),
        dtype=torch.int64,
        device=sign_weight.device,
    )
    signs = torch.zeros(
        (out_features, max_nonzero),
        dtype=torch.float32,
        device=sign_weight.device,
    )
    for row_idx in range(out_features):
        row_indices = torch.nonzero(sign_weight[row_idx], as_tuple=False).squeeze(1)
        count = int(row_indices.numel())
        if count == 0:
            continue
        indices[row_idx, :count] = row_indices
        signs[row_idx, :count] = sign_weight[row_idx, row_indices].to(torch.float32)

    return IndexedTernaryWeight(
        indices=indices.contiguous(),
        signs=signs.contiguous(),
        scales=_resolve_scales(sign_weight, scales).contiguous(),
        in_features=in_features,
    )


def unpack_ternary_weight(indexed_weight: IndexedTernaryWeight) -> torch.Tensor:
    out_features = indexed_weight.indices.shape[0]
    dense = torch.zeros(
        (out_features, indexed_weight.in_features),
        dtype=torch.float32,
        device=indexed_weight.indices.device,
    )
    if indexed_weight.indices.shape[1] > 0:
        dense.scatter_add_(1, indexed_weight.indices, indexed_weight.signs)
    return dense * indexed_weight.scales.unsqueeze(1)


def indexed_ternary_linear_reference(
    inputs: torch.Tensor,
    indexed_weight: IndexedTernaryWeight,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(inputs, unpack_ternary_weight(indexed_weight), bias)


def indexed_ternary_linear_cpu(
    inputs: torch.Tensor,
    indexed_weight: IndexedTernaryWeight,
    bias: torch.Tensor | None,
    *,
    output_block_size: int = 64,
) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got shape {tuple(inputs.shape)}.")
    if inputs.shape[1] != indexed_weight.in_features:
        raise ValueError(
            f"Input feature mismatch: expected {indexed_weight.in_features}, got {inputs.shape[1]}."
        )
    if output_block_size <= 0:
        raise ValueError("output_block_size must be positive.")

    if inputs.device.type != "cpu" or inputs.is_sparse:
        return indexed_ternary_linear_reference(inputs, indexed_weight, bias)

    out_features = indexed_weight.indices.shape[0]
    outputs = torch.empty(
        (inputs.shape[0], out_features),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    indices = indexed_weight.indices.to(device=inputs.device)
    signs = indexed_weight.signs.to(device=inputs.device, dtype=inputs.dtype)

    if indices.shape[1] == 0:
        outputs.zero_()
    else:
        for start in range(0, out_features, output_block_size):
            end = min(out_features, start + output_block_size)
            gathered = inputs[:, indices[start:end]]
            outputs[:, start:end] = (
                gathered * signs[start:end].unsqueeze(0)
            ).sum(dim=-1)

    outputs.mul_(
        indexed_weight.scales.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(0)
    )
    if bias is not None:
        outputs.add_(bias.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(0))
    return outputs


@lru_cache(maxsize=None)
def _subset_basis(block_size: int) -> torch.Tensor:
    if block_size <= 0 or block_size > 8:
        raise ValueError("block_size must be in the range [1, 8].")

    masks = torch.arange(1 << block_size, dtype=torch.int64).unsqueeze(1)
    bit_offsets = torch.arange(block_size, dtype=torch.int64).unsqueeze(0)
    return ((masks >> bit_offsets) & 1).to(torch.float32)


def pack_packed_ternary_lookup_weight(
    weight: torch.Tensor,
    scales: torch.Tensor | None = None,
    *,
    block_size: int = 8,
) -> PackedTernaryLookupWeight:
    if weight.ndim != 2:
        raise ValueError(
            f"Expected a 2D ternary weight tensor, got shape {tuple(weight.shape)}."
        )
    if block_size <= 0 or block_size > 8:
        raise ValueError("block_size must be in the range [1, 8].")

    sign_weight = weight.detach().to(torch.int8)
    out_features, in_features = sign_weight.shape
    padding = (-in_features) % block_size
    if padding:
        sign_weight = F.pad(sign_weight, (0, padding))

    blocks = sign_weight.view(out_features, -1, block_size)
    bit_offsets = (
        1 << torch.arange(block_size, device=sign_weight.device, dtype=torch.int64)
    ).view(1, 1, -1)
    positive_masks = torch.sum((blocks > 0).to(torch.int64) * bit_offsets, dim=-1)
    negative_masks = torch.sum((blocks < 0).to(torch.int64) * bit_offsets, dim=-1)

    return PackedTernaryLookupWeight(
        positive_masks=positive_masks.to(torch.uint8).contiguous(),
        negative_masks=negative_masks.to(torch.uint8).contiguous(),
        scales=_resolve_scales(sign_weight[:, :in_features], scales).contiguous(),
        in_features=in_features,
        block_size=block_size,
    )


def unpack_packed_ternary_lookup_weight(
    packed_weight: PackedTernaryLookupWeight,
) -> torch.Tensor:
    positive_masks = packed_weight.positive_masks
    negative_masks = packed_weight.negative_masks
    out_features, num_blocks = positive_masks.shape
    bit_offsets = torch.arange(
        packed_weight.block_size,
        dtype=torch.int64,
        device=positive_masks.device,
    )
    positive_bits = ((positive_masks.to(torch.int64).unsqueeze(-1) >> bit_offsets) & 1).to(torch.float32)
    negative_bits = ((negative_masks.to(torch.int64).unsqueeze(-1) >> bit_offsets) & 1).to(torch.float32)
    dense = (positive_bits - negative_bits).view(
        out_features, num_blocks * packed_weight.block_size
    )
    dense = dense[:, : packed_weight.in_features]
    return dense * packed_weight.scales.unsqueeze(1)


def packed_ternary_lookup_linear_reference(
    inputs: torch.Tensor,
    packed_weight: PackedTernaryLookupWeight,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    dense_weight = unpack_packed_ternary_lookup_weight(packed_weight).to(
        device=inputs.device,
        dtype=inputs.dtype,
    )
    return F.linear(inputs, dense_weight, bias)


def packed_ternary_lookup_linear_cpu(
    inputs: torch.Tensor,
    packed_weight: PackedTernaryLookupWeight,
    bias: torch.Tensor | None,
    *,
    output_block_size: int = 64,
) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got shape {tuple(inputs.shape)}.")
    if inputs.shape[1] != packed_weight.in_features:
        raise ValueError(
            f"Input feature mismatch: expected {packed_weight.in_features}, got {inputs.shape[1]}."
        )
    if output_block_size <= 0:
        raise ValueError("output_block_size must be positive.")

    if inputs.device.type != "cpu" or inputs.is_sparse:
        return packed_ternary_lookup_linear_reference(inputs, packed_weight, bias)

    inputs_float = inputs.to(torch.float32)
    padding = packed_weight.padded_in_features - packed_weight.in_features
    if padding:
        inputs_float = F.pad(inputs_float, (0, padding))

    num_blocks = packed_weight.positive_masks.shape[1]
    blocks = inputs_float.view(inputs_float.shape[0], num_blocks, packed_weight.block_size)
    subset_basis = _subset_basis(packed_weight.block_size).to(device=inputs.device)
    subset_sums = torch.matmul(blocks, subset_basis.transpose(0, 1)).transpose(1, 2)

    outputs = torch.empty(
        (inputs.shape[0], packed_weight.positive_masks.shape[0]),
        dtype=torch.float32,
        device=inputs.device,
    )
    positive_masks = packed_weight.positive_masks.to(device=inputs.device, dtype=torch.long)
    negative_masks = packed_weight.negative_masks.to(device=inputs.device, dtype=torch.long)
    batch_size = inputs.shape[0]
    out_features = positive_masks.shape[0]

    for start in range(0, out_features, output_block_size):
        end = min(out_features, start + output_block_size)
        positive_indices = positive_masks[start:end].unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        negative_indices = negative_masks[start:end].unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        positive_terms = torch.take_along_dim(subset_sums, positive_indices, dim=1)
        negative_terms = torch.take_along_dim(subset_sums, negative_indices, dim=1)
        outputs[:, start:end] = (positive_terms - negative_terms).sum(dim=-1)

    outputs.mul_(packed_weight.scales.to(device=inputs.device).unsqueeze(0))
    if bias is not None:
        outputs.add_(bias.to(device=inputs.device, dtype=torch.float32).unsqueeze(0))
    return outputs.to(inputs.dtype)
