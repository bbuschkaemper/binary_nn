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
        sparse_inference_density_threshold: float = 0.4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_scale = threshold_scale
        self.use_cpu_sparse_inference = use_cpu_sparse_inference
        self.sparse_inference_density_threshold = sparse_inference_density_threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._cached_sparse_weight: torch.Tensor | None = None
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

    def _scale(self) -> torch.Tensor:
        return self.weight.abs().mean(dim=1, keepdim=True).detach().clamp_min(1e-6)

    def quantized_weight(self) -> torch.Tensor:
        scale = self._scale()
        threshold = scale * self.threshold_scale
        return cast(torch.Tensor, TernaryQuantizeSTE.apply(self.weight, threshold))

    def nonzero_density(self) -> float:
        with torch.no_grad():
            quantized = self.quantized_weight()
            return float((quantized != 0).float().mean().item())

    def _should_use_sparse_inference(self, inputs: torch.Tensor) -> bool:
        return (
            self.use_cpu_sparse_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
            and self.nonzero_density() <= self.sparse_inference_density_threshold
        )

    def _refresh_sparse_weight_cache(self) -> None:
        quantized = self.quantized_weight().detach().to(torch.float32)
        effective_weight = quantized * self._scale()
        row_indices, col_indices = torch.nonzero(effective_weight, as_tuple=True)
        if row_indices.numel() == 0:
            indices = torch.zeros((2, 0), device=effective_weight.device, dtype=torch.int64)
            values = torch.zeros((0,), device=effective_weight.device, dtype=torch.float32)
        else:
            indices = torch.stack((row_indices, col_indices))
            values = effective_weight[row_indices, col_indices]
        self._cached_sparse_weight = torch.sparse_coo_tensor(
            indices,
            values,
            size=effective_weight.shape,
            device=effective_weight.device,
        ).coalesce()
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
        if self._should_use_sparse_inference(inputs):
            sparse_weight = self._get_sparse_weight()
            outputs = torch.sparse.mm(
                sparse_weight,
                inputs.to(torch.float32).transpose(0, 1),
            ).transpose(0, 1)
            if self.bias is not None:
                outputs = outputs + self.bias.to(outputs.dtype)
            return outputs.to(inputs.dtype)

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
        sparse_inference_density_threshold: float = 0.35,
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
        self.reset_parameters()

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale).clamp_min(1e-4)

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

    def nonzero_density(self) -> float:
        return float((self.weight_state != 0).float().mean().item())

    def extra_parameter_count(self) -> int:
        return int(self.weight_state.numel())

    def _should_use_sparse_inference(self, inputs: torch.Tensor) -> bool:
        return (
            self.use_cpu_sparse_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.device.type == "cpu"
            and inputs.ndim == 2
            and self.nonzero_density() <= self.sparse_inference_density_threshold
        )

    def _build_sparse_weight(self, device: torch.device) -> torch.Tensor:
        state = self.weight_state.to(device=device)
        row_indices, col_indices = torch.nonzero(state, as_tuple=True)
        if row_indices.numel() == 0:
            empty_indices = torch.zeros((2, 0), device=device, dtype=torch.int64)
            empty_values = torch.zeros((0,), device=device, dtype=torch.float32)
            sparse = torch.sparse_coo_tensor(
                empty_indices,
                empty_values,
                size=(self.out_features, self.in_features),
                device=device,
            )
            return sparse.coalesce()

        values = (
            state[row_indices, col_indices].to(torch.float32)
            * self.scale.detach().to(device=device)[row_indices]
        )
        sparse = torch.sparse_coo_tensor(
            torch.stack((row_indices, col_indices)),
            values,
            size=(self.out_features, self.in_features),
            device=device,
        ).coalesce()
        return sparse

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
        if self._should_use_sparse_inference(inputs):
            sparse_weight = self._get_sparse_weight(inputs.device)
            outputs = torch.sparse.mm(
                sparse_weight,
                inputs.to(torch.float32).transpose(0, 1),
            ).transpose(0, 1)
            if self.bias is not None:
                outputs = outputs + self.bias.to(outputs.dtype)
            return outputs.to(inputs.dtype)

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
