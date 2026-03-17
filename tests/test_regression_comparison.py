from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from regression_data import RegressionDataConfig
from regression_experiment import TrainingConfig
from run_regression_comparison import compare_dense_and_binary_regression


def test_regression_comparison_reports_runtime_and_custom_widths() -> None:
    comparison = compare_dense_and_binary_regression(
        data_config=RegressionDataConfig(
            n_samples=512,
            n_features=10,
            n_informative=10,
            noise=8.0,
            batch_size=64,
            random_state=7,
        ),
        dense_training_config=TrainingConfig(
            hidden_dims=(16, 8),
            epochs=5,
            learning_rate=1e-3,
            seed=7,
            accelerator="cpu",
        ),
        binary_training_config=TrainingConfig(
            hidden_dims=(32, 16),
            epochs=5,
            learning_rate=1e-3,
            seed=7,
            accelerator="cpu",
        ),
    )

    assert comparison.dense_result.training_config.hidden_dims == (16, 8)
    assert comparison.binary_result.training_config.hidden_dims == (32, 16)
    assert comparison.dense_result.runtime.parameter_count > 0
    assert comparison.binary_result.runtime.parameter_count > 0
    assert comparison.dense_result.runtime.total_seconds > 0.0
    assert comparison.binary_result.runtime.total_seconds > 0.0
