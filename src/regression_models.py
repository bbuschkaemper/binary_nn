from __future__ import annotations

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

    def _should_use_packed_inference(self, inputs: torch.Tensor) -> bool:
        return (
            self.use_triton_packed_inference
            and not self.training
            and not torch.is_grad_enabled()
            and inputs.is_cuda
            and inputs.ndim == 2
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
