from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from regression_data import RegressionDataConfig
from regression_experiment import (
    StepBenchmarkConfig,
    TrainingConfig,
    _fit_stage_benchmark_config_for_model,
    _prepare_model_for_fit_stage_benchmark,
    benchmark_regression_run_result_stages,
)
from run_regression_baseline import train_regression_baseline
from run_regression_comparison import regression_run_result_to_dict
from torch import nn


def _train_small_dense_result():
    return train_regression_baseline(
        data_config=RegressionDataConfig(
            n_samples=256,
            n_features=10,
            n_informative=10,
            noise=8.0,
            batch_size=32,
            random_state=17,
        ),
        training_config=TrainingConfig(
            hidden_dims=(16, 8),
            epochs=2,
            learning_rate=1e-3,
            seed=17,
            accelerator="cpu",
        ),
    )


def test_stage_benchmarks_report_fit_test_and_predict_metrics() -> None:
    result = _train_small_dense_result()

    stage_benchmarks = benchmark_regression_run_result_stages(
        result,
        StepBenchmarkConfig(repetitions=2, warmup_steps=1, timed_steps=1),
    )

    assert stage_benchmarks.fit is not None
    assert stage_benchmarks.test is not None
    assert stage_benchmarks.predict is not None
    assert stage_benchmarks.fit.batch_size == 32
    assert stage_benchmarks.test.batch_size == 32
    assert stage_benchmarks.predict.batch_size == 32
    assert stage_benchmarks.fit.mean_step_ms > 0.0
    assert stage_benchmarks.test.mean_step_ms > 0.0
    assert stage_benchmarks.predict.mean_step_ms > 0.0
    assert stage_benchmarks.fit.std_step_ms >= 0.0
    assert stage_benchmarks.fit.samples_per_second > 0.0
    assert stage_benchmarks.fit.peak_memory_mb is None


def test_regression_result_serialization_includes_stage_benchmarks() -> None:
    result = _train_small_dense_result()
    result.runtime.stage_benchmarks = benchmark_regression_run_result_stages(
        result,
        StepBenchmarkConfig(repetitions=1, warmup_steps=0, timed_steps=1),
    )

    serialized = regression_run_result_to_dict(result)

    runtime = serialized["runtime"]
    assert runtime["stage_benchmarks"]["fit"]["mean_step_ms"] > 0.0
    assert runtime["stage_benchmarks"]["test"]["mean_step_ms"] > 0.0
    assert runtime["stage_benchmarks"]["predict"]["mean_step_ms"] > 0.0


class _CycleAwareModule(nn.Module):
    def __init__(self, cycle_length: int) -> None:
        super().__init__()
        self.cycle_length = cycle_length
        self.prepare_calls = 0

    def prepare_for_fit_stage_benchmark_(self) -> None:
        self.prepare_calls += 1

    def fit_stage_benchmark_cycle_length(self) -> int:
        return self.cycle_length


def test_fit_stage_benchmark_config_expands_to_full_cycles() -> None:
    model = nn.Sequential(_CycleAwareModule(8), _CycleAwareModule(4))

    config = _fit_stage_benchmark_config_for_model(
        model,
        StepBenchmarkConfig(repetitions=1, warmup_steps=2, timed_steps=10),
    )

    assert config.timed_steps == 16
    assert config.warmup_steps == 2


def test_prepare_model_for_fit_stage_benchmark_runs_submodule_hooks() -> None:
    first = _CycleAwareModule(8)
    second = _CycleAwareModule(4)
    model = nn.Sequential(first, second)

    _prepare_model_for_fit_stage_benchmark(model)

    assert first.prepare_calls == 1
    assert second.prepare_calls == 1
