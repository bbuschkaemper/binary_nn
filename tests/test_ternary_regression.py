from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from regression_data import RegressionDataConfig, create_regression_dataloaders
from regression_models import (
    ShadowFreeTernaryLinear,
    ShadowFreeTernaryRegressor,
    TernaryLinear,
    TernaryRegressor,
)


def test_regression_data_supports_nonlinear_residual_targets() -> None:
    nonlinear_bundle = create_regression_dataloaders(
        RegressionDataConfig(
            n_samples=256,
            n_features=8,
            n_informative=8,
            noise=1.0,
            target_kind="nonlinear_residual",
            nonlinear_scale=8.0,
            nonlinear_pair_count=3,
            batch_size=32,
            random_state=7,
        )
    )
    linear_bundle = create_regression_dataloaders(
        RegressionDataConfig(
            n_samples=256,
            n_features=8,
            n_informative=8,
            noise=1.0,
            target_kind="linear",
            batch_size=32,
            random_state=7,
        )
    )

    assert nonlinear_bundle.input_dim == 8
    assert nonlinear_bundle.train_targets_original.shape[0] > 0
    assert np.std(nonlinear_bundle.train_targets_original) > 0.0
    assert not np.allclose(
        np.sort(nonlinear_bundle.train_targets_original),
        np.sort(linear_bundle.train_targets_original),
    )


def test_shadowfree_ternary_linear_sparse_inference_matches_dense_path() -> None:
    layer = ShadowFreeTernaryLinear(
        in_features=4,
        out_features=3,
        initial_density=0.5,
        update_interval=1,
        sparse_inference_density_threshold=1.0,
    )
    with torch.no_grad():
        layer.weight_state.copy_(
            torch.tensor(
                [
                    [1, 0, -1, 0],
                    [0, 1, 0, -1],
                    [1, 1, 0, 0],
                ],
                dtype=torch.int8,
            )
        )
        layer.log_scale.zero_()
        assert layer.bias is not None
        layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    inputs = torch.randn(6, 4)
    layer.eval()
    layer.use_cpu_sparse_inference = False
    dense_outputs = layer(inputs)
    layer.use_cpu_sparse_inference = True
    sparse_outputs = layer(inputs)

    assert torch.allclose(sparse_outputs, dense_outputs, atol=1e-6, rtol=1e-6)


def test_ternary_linear_sparse_inference_matches_dense_path() -> None:
    layer = TernaryLinear(
        in_features=4,
        out_features=3,
        threshold_scale=0.5,
        sparse_inference_density_threshold=1.0,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.9, 0.1, -0.8, 0.0],
                    [0.0, 0.7, 0.0, -0.9],
                    [0.8, 0.9, 0.1, -0.1],
                ],
                dtype=torch.float32,
            )
        )
        assert layer.bias is not None
        layer.bias.copy_(torch.tensor([0.05, -0.1, 0.2]))

    inputs = torch.randn(6, 4)
    layer.eval()
    layer.use_cpu_sparse_inference = False
    dense_outputs = layer(inputs)
    layer.use_cpu_sparse_inference = True
    sparse_outputs = layer(inputs)

    assert torch.allclose(sparse_outputs, dense_outputs, atol=1e-6, rtol=1e-6)


def test_shadowfree_ternary_linear_direct_update_uses_evidence_thresholding() -> None:
    negative_evidence_layer = ShadowFreeTernaryLinear(
        in_features=1,
        out_features=1,
        initial_density=0.5,
        update_interval=1,
    )
    with torch.no_grad():
        negative_evidence_layer.weight_state.zero_()
        negative_evidence_layer._accumulated_evidence.fill_(-2.0)
    negative_evidence_layer._batches_since_update = 1
    negative_evidence_layer.apply_discrete_updates_()

    positive_evidence_layer = ShadowFreeTernaryLinear(
        in_features=1,
        out_features=1,
        initial_density=0.5,
        update_interval=1,
    )
    with torch.no_grad():
        positive_evidence_layer.weight_state.zero_()
        positive_evidence_layer._accumulated_evidence.fill_(2.0)
    positive_evidence_layer._batches_since_update = 1
    positive_evidence_layer.apply_discrete_updates_()

    assert int(negative_evidence_layer.weight_state.item()) == 1
    assert int(positive_evidence_layer.weight_state.item()) == -1


def test_shadowfree_regressor_can_initialize_from_ste_regressor() -> None:
    source = TernaryRegressor(
        input_dim=4,
        hidden_dims=(3,),
        use_input_shortcut=True,
        threshold_scale=0.5,
    )
    with torch.no_grad():
        ternary_layer = next(
            module for module in source.modules() if isinstance(module, TernaryLinear)
        )
        ternary_layer.weight.copy_(
            torch.tensor(
                [
                    [0.9, 0.1, -0.8, 0.0],
                    [0.0, 0.7, 0.0, -0.9],
                    [0.8, 0.9, 0.1, -0.1],
                ],
                dtype=torch.float32,
            )
        )
        assert ternary_layer.bias is not None
        ternary_layer.bias.copy_(torch.tensor([0.05, -0.1, 0.2]))
        output_head = source.ternary_path[-1]
        assert isinstance(output_head, torch.nn.Linear)
        output_head.weight.copy_(torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32))
        output_head.bias.copy_(torch.tensor([0.3], dtype=torch.float32))
        assert source.shortcut is not None
        source.shortcut.weight.copy_(
            torch.tensor([[0.2, -0.1, 0.05, 0.3]], dtype=torch.float32)
        )
        source.shortcut.bias.copy_(torch.tensor([0.15], dtype=torch.float32))

    target = ShadowFreeTernaryRegressor.from_ste_regressor(
        source,
        update_interval=1,
    )
    for module in target.modules():
        if isinstance(module, ShadowFreeTernaryLinear):
            module.use_cpu_sparse_inference = False

    inputs = torch.randn(6, 4)
    source.eval()
    target.eval()

    source_outputs = source(inputs)
    target_outputs = target(inputs)

    assert torch.allclose(target_outputs, source_outputs, atol=1e-6, rtol=1e-6)


def test_ternary_linear_export_can_project_to_target_density() -> None:
    layer = TernaryLinear(
        in_features=4,
        out_features=2,
        threshold_scale=0.5,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.95, 0.8, 0.65, 0.1],
                    [-0.9, 0.75, -0.55, 0.05],
                ],
                dtype=torch.float32,
            )
        )
        assert layer.bias is not None
        layer.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))

    weight_state, scale, bias = layer.export_shadowfree_state(target_density=0.25)

    expected_weight_state = torch.tensor(
        [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
        ],
        dtype=torch.int8,
    )
    expected_scale = torch.tensor([0.625, 0.5625], dtype=torch.float32)

    assert torch.equal(weight_state, expected_weight_state)
    assert torch.allclose(scale, expected_scale, atol=1e-6, rtol=1e-6)
    assert bias is not None
    assert torch.allclose(bias, torch.tensor([0.1, -0.2], dtype=torch.float32))
