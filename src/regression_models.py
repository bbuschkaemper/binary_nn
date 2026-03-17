from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


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
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        surrogate_mask = (inputs.abs() <= 1).to(grad_output.dtype)
        return grad_output * surrogate_mask


class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def clip_weights_(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(-1.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        binary_weight = BinarySignSTE.apply(self.weight)
        scale = self.weight.abs().mean(dim=1, keepdim=True).detach().clamp_min(1e-6)
        return F.linear(inputs, binary_weight * scale, self.bias)


def build_binary_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(BinaryLinear(previous_dim, hidden_dim, bias=False))
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
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (32, 16)) -> None:
        super().__init__()
        self.network = build_binary_mlp(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def clip_weights_(self) -> None:
        for module in self.modules():
            if isinstance(module, BinaryLinear):
                module.clip_weights_()
