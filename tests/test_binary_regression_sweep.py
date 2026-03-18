from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from run_binary_regression_sweep import (
    BinarySweepSummary,
    build_sweep_summary,
    pareto_frontier,
    sweep_summary_to_record,
)


def test_pareto_frontier_keeps_non_dominated_candidates() -> None:
    candidates = [
        BinarySweepSummary(
            (16,),
            3e-3,
            75,
            True,
            rmse=12.7,
            r2=0.9968,
            total_seconds=8.8,
            parameter_count=204,
        ),
        BinarySweepSummary(
            (16,),
            3e-3,
            40,
            True,
            rmse=19.0,
            r2=0.9928,
            total_seconds=3.5,
            parameter_count=204,
        ),
        BinarySweepSummary(
            (16, 16),
            2e-3,
            75,
            True,
            rmse=14.6,
            r2=0.9958,
            total_seconds=14.6,
            parameter_count=476,
        ),
        BinarySweepSummary(
            (32, 16),
            2e-3,
            50,
            True,
            rmse=22.9,
            r2=0.9896,
            total_seconds=5.0,
            parameter_count=908,
        ),
    ]

    frontier = pareto_frontier(candidates)

    assert candidates[0] in frontier
    assert candidates[1] in frontier
    assert candidates[2] not in frontier
    assert candidates[3] not in frontier


def test_sweep_summary_to_record_serializes_shortcut_flag() -> None:
    summary = BinarySweepSummary((8,), 3e-3, 40, False, 15.1, 0.9955, 3.43, 108)

    record = sweep_summary_to_record(summary)

    assert record["hidden_dims"] == [8]
    assert record["use_input_shortcut"] is False


def test_build_sweep_summary_includes_frontier_and_dense_reference() -> None:
    class _Metrics:
        def __init__(self, rmse: float, r2: float) -> None:
            self.rmse = rmse
            self.r2 = r2

    class _Runtime:
        def __init__(self, total_seconds: float, parameter_count: int) -> None:
            self.total_seconds = total_seconds
            self.parameter_count = parameter_count

    class _Training:
        def __init__(self) -> None:
            self.hidden_dims = (64, 32)
            self.learning_rate = 1e-3
            self.epochs = 75

    class _DenseResult:
        def __init__(self) -> None:
            self.test_metrics = _Metrics(rmse=14.9, r2=0.9956)
            self.runtime = _Runtime(total_seconds=8.53, parameter_count=2817)
            self.training_config = _Training()

    summaries = [
        BinarySweepSummary((8,), 3e-3, 75, True, 12.4, 0.9970, 6.3, 108),
        BinarySweepSummary((8,), 3e-3, 40, True, 18.0, 0.9930, 3.8, 108),
    ]

    summary = build_sweep_summary(_DenseResult(), summaries)

    assert summary["candidate_count"] == 2
    assert summary["dense_reference"]["hidden_dims"] == [64, 32]
    assert len(summary["pareto_frontier"]) == 2
    assert summary["best_rmse_candidates"][0]["rmse"] == 12.4
