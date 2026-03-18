from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from regression_data import RegressionDataConfig
from regression_experiment import TrainingConfig
from model_inference_benchmarking import InferenceBenchmarkConfig
from run_regression_comparison import (
    compare_dense_and_binary_regression,
    regression_comparison_result_to_dict,
)


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
            hidden_dims=(8,),
            epochs=5,
            learning_rate=3e-3,
            seed=7,
            accelerator="cpu",
        ),
        inference_benchmark_config=InferenceBenchmarkConfig(
            batch_sizes=(64,),
            iterations=2,
            warmup=1,
            seed=7,
        ),
    )

    assert comparison.dense_result.training_config.hidden_dims == (16, 8)
    assert comparison.binary_result.training_config.hidden_dims == (8,)
    assert comparison.binary_result.training_config.learning_rate == 3e-3
    assert comparison.dense_result.runtime.parameter_count > 0
    assert comparison.binary_result.runtime.parameter_count > 0
    assert comparison.dense_result.runtime.total_seconds > 0.0
    assert comparison.binary_result.runtime.total_seconds > 0.0
    assert comparison.inference_benchmark_records is not None
    assert len(comparison.inference_benchmark_records) >= 2
    assert any(
        record.model_name == "dense"
        for record in comparison.inference_benchmark_records
    )
    assert any(
        record.model_name == "binary"
        for record in comparison.inference_benchmark_records
    )
    assert all(
        record.test_rmse > 0.0 for record in comparison.inference_benchmark_records
    )


def test_regression_comparison_serialization_includes_inference_summary() -> None:
    comparison = compare_dense_and_binary_regression(
        data_config=RegressionDataConfig(
            n_samples=256,
            n_features=10,
            n_informative=10,
            noise=8.0,
            batch_size=64,
            random_state=11,
        ),
        dense_training_config=TrainingConfig(
            hidden_dims=(16, 8),
            epochs=3,
            learning_rate=1e-3,
            seed=11,
            accelerator="cpu",
        ),
        binary_training_config=TrainingConfig(
            hidden_dims=(8,),
            epochs=3,
            learning_rate=3e-3,
            seed=11,
            accelerator="cpu",
        ),
        inference_benchmark_config=InferenceBenchmarkConfig(
            batch_sizes=(32,),
            iterations=1,
            warmup=0,
            seed=11,
        ),
    )

    serialized = regression_comparison_result_to_dict(comparison)

    assert "dense_result" in serialized
    assert "binary_result" in serialized
    assert "deltas" in serialized
    assert "inference_benchmark" in serialized
    assert serialized["inference_benchmark"]["summary"]["candidate_count"] >= 2


def test_regression_comparison_accepts_custom_feature_counts() -> None:
    comparison = compare_dense_and_binary_regression(
        data_config=RegressionDataConfig(
            n_samples=256,
            n_features=32,
            n_informative=32,
            noise=8.0,
            batch_size=64,
            random_state=13,
        ),
        dense_training_config=TrainingConfig(
            hidden_dims=(32, 16),
            epochs=2,
            learning_rate=1e-3,
            seed=13,
            accelerator="cpu",
        ),
        binary_training_config=TrainingConfig(
            hidden_dims=(16,),
            epochs=2,
            learning_rate=3e-3,
            seed=13,
            accelerator="cpu",
        ),
        inference_benchmark_config=InferenceBenchmarkConfig(
            batch_sizes=(16,),
            iterations=1,
            warmup=0,
            seed=13,
        ),
    )

    assert comparison.dense_result.data_config.n_features == 32
    assert comparison.binary_result.data_config.n_features == 32
    assert comparison.inference_benchmark_records is not None
    assert all(record.input_dim == 32 for record in comparison.inference_benchmark_records)
