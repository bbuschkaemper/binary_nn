from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from run_binary_regression_sweep import BinarySweepSummary, pareto_frontier


def test_pareto_frontier_keeps_non_dominated_candidates() -> None:
    candidates = [
        BinarySweepSummary(
            (16,),
            3e-3,
            75,
            rmse=12.7,
            r2=0.9968,
            total_seconds=8.8,
            parameter_count=204,
        ),
        BinarySweepSummary(
            (16,),
            3e-3,
            40,
            rmse=19.0,
            r2=0.9928,
            total_seconds=3.5,
            parameter_count=204,
        ),
        BinarySweepSummary(
            (16, 16),
            2e-3,
            75,
            rmse=14.6,
            r2=0.9958,
            total_seconds=14.6,
            parameter_count=476,
        ),
        BinarySweepSummary(
            (32, 16),
            2e-3,
            50,
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
