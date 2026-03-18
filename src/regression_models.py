from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

from binary_kernels import (
    PackedBinaryWeight,
    pack_binary_weight,
    packed_binary_linear_triton,
)
from ternary_kernels import (
    IndexedTernaryWeight,
    indexed_ternary_linear_cpu,
    pack_ternary_weight,
)

# All observed projected frontier points still favor cached dense CPU inference over
# the sparse CSR path, so projected models should not select sparse by default.
PROJECTED_SPARSE_INFERENCE_DENSITY_THRESHOLD = 0.0


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    linear_layer: Callable[[int, int], nn.Module] = nn.Linear,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(linear_layer(previous_dim, hidden_dim))
        layers.append(nn.ReLU())
        previous_dim = hidden_dim

    layers.append(linear_layer(previous_dim, output_dim))
    return nn.Sequential(*layers)


class BinarySignSTE(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return inputs.sign().masked_fill(inputs == 0, 1.0)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor]:
        (inputs,) = ctx.saved_tensors
        (grad_output,) = grad_outputs
        surrogate_mask = (inputs.abs() <= 1).to(grad_output.dtype)
        return (grad_output * surrogate_mask,)


class TernaryQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        positive = inputs >= threshold
        negative = inputs <= -threshold
        outputs = torch.zeros_like(inputs)
        outputs = outputs.masked_fill(positive, 1.0)
        outputs = outputs.masked_fill(negative, -1.0)
        return outputs

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (inputs,) = ctx.saved_tensors
        (grad_output,) = grad_outputs
        surrogate_mask = (inputs.abs() <= 1).to(grad_output.dtype)
        return (grad_output * surrogate_mask, None)


class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.use_triton_packed_inference = True
        self._cached_packed_weight: PackedBinaryWeight | None = None
        self._cached_weight_version = -1
        self._cached_weight_device: torch.device | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def clip_weights_(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(-1.0, 1.0)

    def _refresh_packed_weight_cache(self) -> None:
        self._cached_packed_weight = pack_binary_weight(self.weight)
        self._cached_weight_version = self.weight._version
        self._cached_weight_device = self.weight.device

    def _known_triton_losing_shape(self, inputs: torch.Tensor) -> bool:
        return (
            inputs.shape[0] >= 16384
            and self.in_features <= 1024
            and self.out_features <= 1024
        )

    def _should_use_packed_inference(self, inputs: torch.Tensor) -> bool:
        return (
            self.use_triton_packed_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.is_cuda
            and inputs.ndim == 2
            and not self._known_triton_losing_shape(inputs)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._should_use_packed_inference(inputs):
            if (
                self._cached_packed_weight is None
                or self._cached_weight_version != self.weight._version
                or self._cached_weight_device != self.weight.device
            ):
                self._refresh_packed_weight_cache()
            assert self._cached_packed_weight is not None
            return packed_binary_linear_triton(
                inputs, self._cached_packed_weight, self.bias
            )

        binary_weight = cast(torch.Tensor, BinarySignSTE.apply(self.weight))
        scale = self.weight.abs().mean(dim=1, keepdim=True).detach().clamp_min(1e-6)
        return F.linear(inputs, binary_weight * scale, self.bias)


class TernaryLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        threshold_scale: float = 0.75,
        use_cpu_sparse_inference: bool = True,
        sparse_inference_density_threshold: float = 0.15,
        use_cpu_index_inference: bool = False,
        index_inference_min_density: float = 0.15,
        index_inference_density_threshold: float = 0.5,
        index_inference_min_batch_size: int = 128,
        index_inference_output_block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_scale = threshold_scale
        self.use_cpu_sparse_inference = use_cpu_sparse_inference
        self.sparse_inference_density_threshold = sparse_inference_density_threshold
        self.use_cpu_index_inference = use_cpu_index_inference
        self.index_inference_min_density = index_inference_min_density
        self.index_inference_density_threshold = index_inference_density_threshold
        self.index_inference_min_batch_size = index_inference_min_batch_size
        self.index_inference_output_block_size = index_inference_output_block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._cached_sparse_weight: torch.Tensor | None = None
        self._cached_weight_version = -1
        self._cached_weight_device: torch.device | None = None
        self._cached_dense_weight: torch.Tensor | None = None
        self._cached_dense_weight_version = -1
        self._cached_dense_weight_device: torch.device | None = None
        self._cached_indexed_weight: IndexedTernaryWeight | None = None
        self._cached_indexed_weight_version = -1
        self._cached_indexed_weight_device: torch.device | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def clip_weights_(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(-1.0, 1.0)

    def _scale(self) -> torch.Tensor:
        return self.weight.abs().mean(dim=1, keepdim=True).detach().clamp_min(1e-6)

    def _prune_quantized_weight_to_density(
        self, quantized_weight: torch.Tensor, target_density: float
    ) -> torch.Tensor:
        if not 0.0 < target_density <= 1.0:
            raise ValueError("target_density must be in the range (0, 1].")

        nonzero_mask = quantized_weight != 0
        active_count = int(nonzero_mask.sum().item())
        if active_count == 0:
            return quantized_weight

        target_active_count = min(
            active_count,
            max(1, int(round(target_density * quantized_weight.numel()))),
        )
        if target_active_count >= active_count:
            return quantized_weight

        active_scores = self.weight.detach().abs()[nonzero_mask]
        keep_positions = torch.topk(
            active_scores,
            k=target_active_count,
            largest=True,
            sorted=False,
        ).indices
        flat_quantized = quantized_weight.clone().view(-1)
        nonzero_indices = torch.nonzero(
            nonzero_mask.view(-1), as_tuple=False
        ).squeeze(1)
        prune_mask = torch.ones(
            active_count,
            dtype=torch.bool,
            device=quantized_weight.device,
        )
        prune_mask[keep_positions] = False
        flat_quantized[nonzero_indices[prune_mask]] = 0
        return flat_quantized.view_as(quantized_weight)

    def _prune_quantized_weight_to_row_block_density(
        self,
        quantized_weight: torch.Tensor,
        target_density: float,
        block_size: int,
    ) -> torch.Tensor:
        if block_size <= 0:
            raise ValueError("projection_block_size must be positive.")
        if block_size == 1:
            return self._prune_quantized_weight_to_density(
                quantized_weight,
                target_density,
            )
        if not 0.0 < target_density <= 1.0:
            raise ValueError("target_density must be in the range (0, 1].")

        nonzero_mask = quantized_weight != 0
        active_count = int(nonzero_mask.sum().item())
        if active_count == 0:
            return quantized_weight

        target_active_count = min(
            active_count,
            max(1, int(round(target_density * quantized_weight.numel()))),
        )
        if target_active_count >= active_count:
            return quantized_weight

        weight_abs = self.weight.detach().abs()
        block_candidates: list[tuple[float, int, int, int, int]] = []
        for row_idx in range(quantized_weight.shape[0]):
            row_nonzero_mask = nonzero_mask[row_idx]
            if not row_nonzero_mask.any():
                continue
            for start in range(0, quantized_weight.shape[1], block_size):
                end = min(start + block_size, quantized_weight.shape[1])
                block_nonzero_mask = row_nonzero_mask[start:end]
                block_active_count = int(block_nonzero_mask.sum().item())
                if block_active_count == 0:
                    continue
                block_score = float(
                    weight_abs[row_idx, start:end][block_nonzero_mask].sum().item()
                )
                block_candidates.append(
                    (block_score, block_active_count, row_idx, start, end)
                )

        if not block_candidates:
            return torch.zeros_like(quantized_weight)

        block_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        structured_quantized_weight = torch.zeros_like(quantized_weight)
        selected_active_count = 0
        kept_any = False
        for _, block_active_count, row_idx, start, end in block_candidates:
            if selected_active_count >= target_active_count:
                break
            if (
                kept_any
                and selected_active_count + block_active_count > target_active_count
            ):
                continue
            structured_quantized_weight[row_idx, start:end] = quantized_weight[
                row_idx, start:end
            ]
            selected_active_count += block_active_count
            kept_any = True

        if not kept_any:
            _, _, row_idx, start, end = block_candidates[0]
            structured_quantized_weight[row_idx, start:end] = quantized_weight[
                row_idx, start:end
            ]

        return structured_quantized_weight

    def quantized_weight(self) -> torch.Tensor:
        scale = self._scale()
        threshold = scale * self.threshold_scale
        return cast(torch.Tensor, TernaryQuantizeSTE.apply(self.weight, threshold))

    def export_shadowfree_state(
        self,
        *,
        target_density: float | None = None,
        projection_structure: str | None = None,
        projection_block_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        quantized_weight = self.quantized_weight().detach().to(torch.int8)
        if target_density is not None:
            if projection_structure is None:
                quantized_weight = self._prune_quantized_weight_to_density(
                    quantized_weight,
                    target_density,
                )
            elif projection_structure == "row_block":
                if projection_block_size is None:
                    raise ValueError(
                        "projection_block_size is required for row_block projection."
                    )
                quantized_weight = self._prune_quantized_weight_to_row_block_density(
                    quantized_weight,
                    target_density,
                    projection_block_size,
                )
            else:
                raise ValueError(
                    f"Unsupported projection_structure: {projection_structure!r}"
                )
        scale = self._scale().detach().squeeze(1).to(torch.float32)
        bias = self.bias.detach().clone() if self.bias is not None else None
        return quantized_weight, scale, bias

    def nonzero_density(self) -> float:
        with torch.no_grad():
            quantized = self.quantized_weight()
            return float((quantized != 0).float().mean().item())

    def _should_use_index_inference(self, inputs: torch.Tensor) -> bool:
        density = self.nonzero_density()
        return (
            self.use_cpu_sparse_inference
            and self.use_cpu_index_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
            and inputs.shape[0] >= self.index_inference_min_batch_size
            and self.index_inference_min_density <= density <= self.index_inference_density_threshold
        )

    def _should_use_sparse_inference(self, inputs: torch.Tensor) -> bool:
        return (
            self.use_cpu_sparse_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
            and self.nonzero_density() <= self.sparse_inference_density_threshold
        )

    def _refresh_indexed_weight_cache(self) -> None:
        quantized = self.quantized_weight().detach().to(torch.int8)
        self._cached_indexed_weight = pack_ternary_weight(
            quantized,
            self._scale().detach().squeeze(1),
        )
        self._cached_indexed_weight_version = self.weight._version
        self._cached_indexed_weight_device = self.weight.device

    def _get_indexed_weight(self) -> IndexedTernaryWeight:
        if (
            self._cached_indexed_weight is None
            or self._cached_indexed_weight_version != self.weight._version
            or self._cached_indexed_weight_device != self.weight.device
        ):
            self._refresh_indexed_weight_cache()
        assert self._cached_indexed_weight is not None
        return self._cached_indexed_weight

    def _should_use_cached_dense_inference(self, inputs: torch.Tensor) -> bool:
        return (
            not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
        )

    def _refresh_dense_weight_cache(self) -> None:
        quantized = self.quantized_weight().detach().to(torch.float32)
        self._cached_dense_weight = (quantized * self._scale()).contiguous()
        self._cached_dense_weight_version = self.weight._version
        self._cached_dense_weight_device = self.weight.device

    def _get_dense_weight(self) -> torch.Tensor:
        if (
            self._cached_dense_weight is None
            or self._cached_dense_weight_version != self.weight._version
            or self._cached_dense_weight_device != self.weight.device
        ):
            self._refresh_dense_weight_cache()
        assert self._cached_dense_weight is not None
        return self._cached_dense_weight

    def _refresh_sparse_weight_cache(self) -> None:
        effective_weight = self._get_dense_weight()
        row_indices, col_indices = torch.nonzero(effective_weight, as_tuple=True)
        if row_indices.numel() == 0:
            crow_indices = torch.zeros(
                self.out_features + 1,
                device=effective_weight.device,
                dtype=torch.int64,
            )
            column_indices = torch.zeros(
                (0,),
                device=effective_weight.device,
                dtype=torch.int64,
            )
            values = torch.zeros(
                (0,),
                device=effective_weight.device,
                dtype=torch.float32,
            )
        else:
            row_counts = torch.bincount(row_indices, minlength=self.out_features)
            crow_indices = torch.zeros(
                self.out_features + 1,
                device=effective_weight.device,
                dtype=torch.int64,
            )
            crow_indices[1:] = torch.cumsum(row_counts, dim=0)
            column_indices = col_indices.to(torch.int64)
            values = effective_weight[row_indices, col_indices]
        self._cached_sparse_weight = torch.sparse_csr_tensor(
            crow_indices,
            column_indices,
            values,
            size=effective_weight.shape,
            device=effective_weight.device,
        )
        self._cached_weight_version = self.weight._version
        self._cached_weight_device = self.weight.device

    def _get_sparse_weight(self) -> torch.Tensor:
        if (
            self._cached_sparse_weight is None
            or self._cached_weight_version != self.weight._version
            or self._cached_weight_device != self.weight.device
        ):
            self._refresh_sparse_weight_cache()
        assert self._cached_sparse_weight is not None
        return self._cached_sparse_weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._should_use_index_inference(inputs):
            outputs = indexed_ternary_linear_cpu(
                inputs,
                self._get_indexed_weight(),
                self.bias,
                output_block_size=self.index_inference_output_block_size,
            )
            return outputs.to(inputs.dtype)

        if self._should_use_sparse_inference(inputs):
            sparse_weight = self._get_sparse_weight()
            outputs = torch.sparse.mm(
                sparse_weight,
                inputs.to(torch.float32).transpose(0, 1),
            ).transpose(0, 1)
            if self.bias is not None:
                outputs = outputs + self.bias.to(outputs.dtype)
            return outputs.to(inputs.dtype)

        if self._should_use_cached_dense_inference(inputs):
            return F.linear(
                inputs,
                self._get_dense_weight().to(device=inputs.device, dtype=inputs.dtype),
                self.bias,
            )

        quantized_weight = self.quantized_weight()
        return F.linear(inputs, quantized_weight * self._scale(), self.bias)


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("Softplus inverse requires a positive value.")
    return math.log(math.expm1(value))


class ShadowFreeTernaryLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        initial_density: float = 0.15,
        update_interval: int = 4,
        activation_std_multiplier: float = 0.5,
        prune_ratio: float = 0.5,
        flip_multiplier: float = 1.5,
        use_cpu_sparse_inference: bool = True,
        sparse_inference_density_threshold: float = 0.15,
        use_cpu_index_inference: bool = False,
        index_inference_min_density: float = 0.15,
        index_inference_density_threshold: float = 0.5,
        index_inference_min_batch_size: int = 128,
        index_inference_output_block_size: int = 64,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_density <= 1.0:
            raise ValueError("initial_density must be in the range (0, 1].")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive.")
        if prune_ratio <= 0.0:
            raise ValueError("prune_ratio must be positive.")
        if flip_multiplier <= 1.0:
            raise ValueError("flip_multiplier must be greater than 1.")

        self.in_features = in_features
        self.out_features = out_features
        self.initial_density = initial_density
        self.update_interval = update_interval
        self.activation_std_multiplier = activation_std_multiplier
        self.prune_ratio = prune_ratio
        self.flip_multiplier = flip_multiplier
        self.use_cpu_sparse_inference = use_cpu_sparse_inference
        self.sparse_inference_density_threshold = sparse_inference_density_threshold
        self.use_cpu_index_inference = use_cpu_index_inference
        self.index_inference_min_density = index_inference_min_density
        self.index_inference_density_threshold = index_inference_density_threshold
        self.index_inference_min_batch_size = index_inference_min_batch_size
        self.index_inference_output_block_size = index_inference_output_block_size
        self.log_scale = nn.Parameter(torch.empty(out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "weight_state",
            torch.empty(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "_accumulated_evidence",
            torch.zeros(out_features, in_features, dtype=torch.float32),
            persistent=False,
        )
        self._batches_since_update = 0
        self._state_version = 0
        self._cached_sparse_weight: torch.Tensor | None = None
        self._cached_sparse_state_version = -1
        self._cached_sparse_scale_version = -1
        self._cached_sparse_device: torch.device | None = None
        self._cached_dense_weight: torch.Tensor | None = None
        self._cached_dense_state_version = -1
        self._cached_dense_scale_version = -1
        self._cached_dense_device: torch.device | None = None
        self._cached_indexed_weight: IndexedTernaryWeight | None = None
        self._cached_indexed_state_version = -1
        self._cached_indexed_scale_version = -1
        self._cached_indexed_device: torch.device | None = None
        self.reset_parameters()

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale).clamp_min(1e-4)

    @torch.no_grad()
    def initialize_from_quantized_state_(
        self,
        weight_state: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        if weight_state.shape != self.weight_state.shape:
            raise ValueError(
                f"Weight-state shape mismatch: expected {tuple(self.weight_state.shape)}, got {tuple(weight_state.shape)}."
            )
        resolved_scale = scale.reshape(-1).to(self.log_scale.dtype)
        if resolved_scale.shape[0] != self.out_features:
            raise ValueError(
                f"Scale shape mismatch: expected {self.out_features}, got {resolved_scale.shape[0]}."
            )

        self.weight_state.copy_(weight_state.to(torch.int8))
        clamped_scale = resolved_scale.clamp_min(1e-4)
        self.log_scale.copy_(torch.log(torch.expm1(clamped_scale)))
        if self.bias is not None:
            if bias is None:
                self.bias.zero_()
            else:
                self.bias.copy_(bias.to(self.bias.dtype))
        self._accumulated_evidence.zero_()
        self._batches_since_update = 0
        self._state_version += 1
        self._invalidate_sparse_cache()

    def reset_parameters(self) -> None:
        scale_init = 1.0 / math.sqrt(max(1, self.in_features))
        activation_mask = (
            torch.rand(self.out_features, self.in_features) < self.initial_density
        )
        sign_mask = torch.where(
            torch.rand(self.out_features, self.in_features) < 0.5,
            torch.ones(
                self.out_features, self.in_features, dtype=torch.int8
            ),
            torch.full(
                (self.out_features, self.in_features),
                -1,
                dtype=torch.int8,
            ),
        )
        with torch.no_grad():
            self.weight_state.zero_()
            self.weight_state[activation_mask] = sign_mask[activation_mask]
            self.log_scale.fill_(_inverse_softplus(scale_init))
            if self.bias is not None:
                self.bias.zero_()
            self._accumulated_evidence.zero_()
        self._batches_since_update = 0
        self._state_version += 1
        self._invalidate_sparse_cache()

    def _invalidate_sparse_cache(self) -> None:
        self._cached_sparse_weight = None
        self._cached_sparse_state_version = -1
        self._cached_sparse_scale_version = -1
        self._cached_sparse_device = None
        self._cached_dense_weight = None
        self._cached_dense_state_version = -1
        self._cached_dense_scale_version = -1
        self._cached_dense_device = None
        self._cached_indexed_weight = None
        self._cached_indexed_state_version = -1
        self._cached_indexed_scale_version = -1
        self._cached_indexed_device = None

    def nonzero_density(self) -> float:
        return float((self.weight_state != 0).float().mean().item())

    def extra_parameter_count(self) -> int:
        return int(self.weight_state.numel())

    def _should_use_index_inference(self, inputs: torch.Tensor) -> bool:
        density = self.nonzero_density()
        return (
            self.use_cpu_sparse_inference
            and self.use_cpu_index_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
            and inputs.shape[0] >= self.index_inference_min_batch_size
            and self.index_inference_min_density <= density <= self.index_inference_density_threshold
        )

    def _should_use_sparse_inference(self, inputs: torch.Tensor) -> bool:
        return (
            self.use_cpu_sparse_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
            and self.nonzero_density() <= self.sparse_inference_density_threshold
        )

    def _should_use_cached_dense_inference(self, inputs: torch.Tensor) -> bool:
        return (
            not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
        )

    def _build_dense_weight(self, device: torch.device) -> torch.Tensor:
        scale = self.scale.detach().to(device=device).unsqueeze(1)
        state = self.weight_state.to(device=device, dtype=torch.float32)
        return (state * scale).contiguous()

    def _get_dense_weight(self, device: torch.device) -> torch.Tensor:
        if (
            self._cached_dense_weight is None
            or self._cached_dense_state_version != self._state_version
            or self._cached_dense_scale_version != self.log_scale._version
            or self._cached_dense_device != device
        ):
            self._cached_dense_weight = self._build_dense_weight(device)
            self._cached_dense_state_version = self._state_version
            self._cached_dense_scale_version = self.log_scale._version
            self._cached_dense_device = device
        return self._cached_dense_weight

    def _build_sparse_weight(self, device: torch.device) -> torch.Tensor:
        dense_weight = self._get_dense_weight(device)
        row_indices, col_indices = torch.nonzero(dense_weight, as_tuple=True)
        if row_indices.numel() == 0:
            crow_indices = torch.zeros(self.out_features + 1, device=device, dtype=torch.int64)
            column_indices = torch.zeros((0,), device=device, dtype=torch.int64)
            values = torch.zeros((0,), device=device, dtype=torch.float32)
            return torch.sparse_csr_tensor(
                crow_indices,
                column_indices,
                values,
                size=(self.out_features, self.in_features),
                device=device,
            )

        values = dense_weight[row_indices, col_indices]
        row_counts = torch.bincount(row_indices, minlength=self.out_features)
        crow_indices = torch.zeros(self.out_features + 1, device=device, dtype=torch.int64)
        crow_indices[1:] = torch.cumsum(row_counts, dim=0)
        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices.to(torch.int64),
            values,
            size=(self.out_features, self.in_features),
            device=device,
        )

    def _get_sparse_weight(self, device: torch.device) -> torch.Tensor:
        if (
            self._cached_sparse_weight is None
            or self._cached_sparse_state_version != self._state_version
            or self._cached_sparse_scale_version != self.log_scale._version
            or self._cached_sparse_device != device
        ):
            self._cached_sparse_weight = self._build_sparse_weight(device)
            self._cached_sparse_state_version = self._state_version
            self._cached_sparse_scale_version = self.log_scale._version
            self._cached_sparse_device = device
        return self._cached_sparse_weight

    def _get_indexed_weight(self, device: torch.device) -> IndexedTernaryWeight:
        if (
            self._cached_indexed_weight is None
            or self._cached_indexed_state_version != self._state_version
            or self._cached_indexed_scale_version != self.log_scale._version
            or self._cached_indexed_device != device
        ):
            self._cached_indexed_weight = pack_ternary_weight(
                self.weight_state.to(torch.int8),
                self.scale.detach(),
            )
            self._cached_indexed_state_version = self._state_version
            self._cached_indexed_scale_version = self.log_scale._version
            self._cached_indexed_device = device
        return self._cached_indexed_weight

    def _accumulate_evidence(
        self, inputs: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        batch_size = max(1, inputs.shape[0])
        evidence = grad_output.to(torch.float32).transpose(0, 1).matmul(
            inputs.to(torch.float32)
        )
        evidence.div_(float(batch_size))
        with torch.no_grad():
            self._accumulated_evidence.add_(evidence)
            self._batches_since_update += 1

    @torch.no_grad()
    def apply_discrete_updates_(self) -> None:
        if self._batches_since_update < self.update_interval:
            return

        averaged_evidence = self._accumulated_evidence / float(self._batches_since_update)
        abs_evidence = averaged_evidence.abs()
        activation_threshold = (
            abs_evidence.mean()
            + self.activation_std_multiplier * abs_evidence.std(unbiased=False)
        ).clamp_min(1e-6)
        prune_threshold = activation_threshold * self.prune_ratio
        flip_threshold = activation_threshold * self.flip_multiplier

        current_state = self.weight_state
        updated_state = current_state.clone()

        zero_mask = current_state == 0
        updated_state[zero_mask & (averaged_evidence <= -activation_threshold)] = 1
        updated_state[zero_mask & (averaged_evidence >= activation_threshold)] = -1

        positive_mask = current_state == 1
        updated_state[positive_mask & (averaged_evidence > flip_threshold)] = -1
        updated_state[
            positive_mask
            & (averaged_evidence >= -prune_threshold)
            & (averaged_evidence <= flip_threshold)
        ] = 0

        negative_mask = current_state == -1
        updated_state[negative_mask & (averaged_evidence < -flip_threshold)] = 1
        updated_state[
            negative_mask
            & (averaged_evidence <= prune_threshold)
            & (averaged_evidence >= -flip_threshold)
        ] = 0

        if not torch.equal(updated_state, current_state):
            self.weight_state.copy_(updated_state)
            self._state_version += 1
            self._invalidate_sparse_cache()

        self._accumulated_evidence.zero_()
        self._batches_since_update = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._should_use_index_inference(inputs):
            outputs = indexed_ternary_linear_cpu(
                inputs,
                self._get_indexed_weight(inputs.device),
                self.bias,
                output_block_size=self.index_inference_output_block_size,
            )
            return outputs.to(inputs.dtype)

        if self._should_use_sparse_inference(inputs):
            sparse_weight = self._get_sparse_weight(inputs.device)
            outputs = torch.sparse.mm(
                sparse_weight,
                inputs.to(torch.float32).transpose(0, 1),
            ).transpose(0, 1)
            if self.bias is not None:
                outputs = outputs + self.bias.to(outputs.dtype)
            return outputs.to(inputs.dtype)

        if self._should_use_cached_dense_inference(inputs):
            outputs = F.linear(
                inputs,
                self._get_dense_weight(inputs.device).to(dtype=inputs.dtype),
                self.bias,
            )
            return outputs

        scale = self.scale.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(1)
        effective_weight = self.weight_state.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        outputs = F.linear(inputs, effective_weight * scale, self.bias)
        if self.training and torch.is_grad_enabled():
            detached_inputs = inputs.detach()

            def _hook(grad_output: torch.Tensor) -> torch.Tensor:
                self._accumulate_evidence(detached_inputs, grad_output.detach())
                return grad_output

            outputs.register_hook(_hook)
        return outputs


def build_binary_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    use_batch_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(BinaryLinear(previous_dim, hidden_dim, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Hardtanh())
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def build_ternary_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    *,
    threshold_scale: float = 0.75,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(
            TernaryLinear(
                previous_dim,
                hidden_dim,
                threshold_scale=threshold_scale,
            )
        )
        layers.append(nn.Hardtanh())
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def build_shadowfree_ternary_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    *,
    initial_density: float = 0.15,
    update_interval: int = 4,
    activation_std_multiplier: float = 0.5,
    prune_ratio: float = 0.5,
    flip_multiplier: float = 1.5,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(
            ShadowFreeTernaryLinear(
                previous_dim,
                hidden_dim,
                initial_density=initial_density,
                update_interval=update_interval,
                activation_std_multiplier=activation_std_multiplier,
                prune_ratio=prune_ratio,
                flip_multiplier=flip_multiplier,
            )
        )
        layers.append(nn.Hardtanh())
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class DenseRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        self.network = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward(self, inputs):
        return self.network(inputs)


class BinaryRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (8,),
        use_batch_norm: bool = False,
        use_input_shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.binary_path = build_binary_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm,
        )
        self.shortcut = nn.Linear(input_dim, 1) if use_input_shortcut else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.binary_path(inputs)
        if self.shortcut is not None:
            outputs = outputs + self.shortcut(inputs)
        return outputs

    def clip_weights_(self) -> None:
        for module in self.modules():
            if isinstance(module, BinaryLinear):
                module.clip_weights_()


class TernaryRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32,),
        *,
        use_input_shortcut: bool = True,
        threshold_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.ternary_path = build_ternary_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            threshold_scale=threshold_scale,
        )
        self.shortcut = nn.Linear(input_dim, 1) if use_input_shortcut else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.ternary_path(inputs)
        if self.shortcut is not None:
            outputs = outputs + self.shortcut(inputs)
        return outputs

    def clip_weights_(self) -> None:
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.clip_weights_()

    def ternary_nonzero_density(self) -> float:
        densities = [
            module.nonzero_density()
            for module in self.modules()
            if isinstance(module, TernaryLinear)
        ]
        if not densities:
            return 0.0
        return float(sum(densities) / len(densities))


class ShadowFreeTernaryRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32,),
        *,
        use_input_shortcut: bool = True,
        initial_density: float = 0.15,
        update_interval: int = 4,
        activation_std_multiplier: float = 0.5,
        prune_ratio: float = 0.5,
        flip_multiplier: float = 1.5,
    ) -> None:
        super().__init__()
        self.ternary_path = build_shadowfree_ternary_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            initial_density=initial_density,
            update_interval=update_interval,
            activation_std_multiplier=activation_std_multiplier,
            prune_ratio=prune_ratio,
            flip_multiplier=flip_multiplier,
        )
        self.shortcut = nn.Linear(input_dim, 1) if use_input_shortcut else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.ternary_path(inputs)
        if self.shortcut is not None:
            outputs = outputs + self.shortcut(inputs)
        return outputs

    def apply_discrete_updates_(self) -> None:
        for module in self.modules():
            if isinstance(module, ShadowFreeTernaryLinear):
                module.apply_discrete_updates_()

    @classmethod
    def from_ste_regressor(
        cls,
        source: TernaryRegressor,
        *,
        target_density: float | None = None,
        projection_structure: str | None = None,
        projection_block_size: int | None = None,
        initial_density: float = 0.15,
        update_interval: int = 4,
        activation_std_multiplier: float = 0.5,
        prune_ratio: float = 0.5,
        flip_multiplier: float = 1.5,
    ) -> "ShadowFreeTernaryRegressor":
        source_layers = [
            module for module in source.modules() if isinstance(module, TernaryLinear)
        ]
        if not source_layers:
            raise ValueError("Source TernaryRegressor contains no TernaryLinear layers.")

        input_dim = source_layers[0].in_features
        hidden_dims = tuple(layer.out_features for layer in source_layers)
        target = cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_input_shortcut=source.shortcut is not None,
            initial_density=initial_density,
            update_interval=update_interval,
            activation_std_multiplier=activation_std_multiplier,
            prune_ratio=prune_ratio,
            flip_multiplier=flip_multiplier,
        )
        source_device = next(source.parameters()).device
        target.to(source_device)
        target.initialize_from_ste_regressor_(
            source,
            target_density=target_density,
            projection_structure=projection_structure,
            projection_block_size=projection_block_size,
        )
        return target

    @torch.no_grad()
    def initialize_from_ste_regressor_(
        self,
        source: TernaryRegressor,
        *,
        target_density: float | None = None,
        projection_structure: str | None = None,
        projection_block_size: int | None = None,
    ) -> None:
        source_layers = [
            module for module in source.modules() if isinstance(module, TernaryLinear)
        ]
        target_layers = [
            module
            for module in self.modules()
            if isinstance(module, ShadowFreeTernaryLinear)
        ]
        if len(source_layers) != len(target_layers):
            raise ValueError(
                f"Ternary-layer count mismatch: source has {len(source_layers)}, target has {len(target_layers)}."
            )

        for source_layer, target_layer in zip(source_layers, target_layers, strict=True):
            weight_state, scale, bias = source_layer.export_shadowfree_state(
                target_density=target_density,
                projection_structure=projection_structure,
                projection_block_size=projection_block_size,
            )
            target_layer.initialize_from_quantized_state_(
                weight_state=weight_state.to(target_layer.weight_state.device),
                scale=scale.to(target_layer.log_scale.device),
                bias=(
                    bias.to(target_layer.bias.device)
                    if bias is not None and target_layer.bias is not None
                    else None
                ),
            )
            if target_density is not None:
                target_layer.sparse_inference_density_threshold = min(
                    target_layer.sparse_inference_density_threshold,
                    PROJECTED_SPARSE_INFERENCE_DENSITY_THRESHOLD,
                )

        source_output_head = cast(nn.Linear, source.ternary_path[-1])
        target_output_head = cast(nn.Linear, self.ternary_path[-1])
        target_output_head.load_state_dict(source_output_head.state_dict())

        if source.shortcut is None:
            self.shortcut = None
        elif self.shortcut is None:
            raise ValueError("Source has a shortcut but target does not.")
        else:
            self.shortcut.load_state_dict(source.shortcut.state_dict())

    def extra_parameter_count(self) -> int:
        return sum(
            module.extra_parameter_count()
            for module in self.modules()
            if isinstance(module, ShadowFreeTernaryLinear)
        )

    def ternary_nonzero_density(self) -> float:
        densities = [
            module.nonzero_density()
            for module in self.modules()
            if isinstance(module, ShadowFreeTernaryLinear)
        ]
        if not densities:
            return 0.0
        return float(sum(densities) / len(densities))
