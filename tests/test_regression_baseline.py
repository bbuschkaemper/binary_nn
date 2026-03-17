from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from regression_data import RegressionDataConfig
from run_regression_baseline import TrainingConfig, train_regression_baseline


def test_dense_regression_baseline_beats_naive_predictor() -> None:
    result = train_regression_baseline(
        data_config=RegressionDataConfig(
            n_samples=1024,
            n_features=10,
            n_informative=10,
            noise=8.0,
            batch_size=128,
            random_state=7,
        ),
        training_config=TrainingConfig(
            hidden_dims=(32, 16),
            epochs=30,
            learning_rate=1e-3,
            seed=7,
        ),
    )

    test_metrics = result.test_metrics
    naive_metrics = result.naive_test_metrics

    assert test_metrics.r2 > 0.9
    assert test_metrics.rmse < naive_metrics.rmse * 0.25
