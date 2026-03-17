from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from model_inference_benchmarking import (  # noqa: E402
    ModelInferenceBenchmarkRecord,
    build_binary_ablation_matrix,
    build_model_inference_summary,
    model_inference_pareto_frontier,
)


def _record(
    *,
    model_name: str,
    batch_size: int,
    use_input_shortcut: bool,
    use_triton_packed_inference: bool,
    latency_ms: float,
    test_rmse: float,
) -> ModelInferenceBenchmarkRecord:
    return ModelInferenceBenchmarkRecord(
        model_name=model_name,
        batch_size=batch_size,
        input_dim=10,
        hidden_dims=(8,),
        use_input_shortcut=use_input_shortcut,
        use_triton_packed_inference=use_triton_packed_inference,
        latency_ms=latency_ms,
        test_loss=test_rmse**2,
        test_rmse=test_rmse,
        test_mae=test_rmse / 2.0,
        test_r2=0.99,
        parameter_count=108,
        benchmark_device="cpu",
    )


def test_model_inference_pareto_frontier_filters_dominated_records() -> None:
    frontier = model_inference_pareto_frontier(
        [
            _record(
                model_name="dense",
                batch_size=128,
                use_input_shortcut=False,
                use_triton_packed_inference=False,
                latency_ms=1.0,
                test_rmse=10.0,
            ),
            _record(
                model_name="binary",
                batch_size=128,
                use_input_shortcut=True,
                use_triton_packed_inference=False,
                latency_ms=0.8,
                test_rmse=11.0,
            ),
            _record(
                model_name="binary",
                batch_size=128,
                use_input_shortcut=True,
                use_triton_packed_inference=True,
                latency_ms=1.2,
                test_rmse=12.0,
            ),
        ]
    )

    assert len(frontier) == 2
    assert all(record.latency_ms < 1.2 or record.test_rmse < 12.0 for record in frontier)


def test_build_binary_ablation_matrix_computes_variant_deltas() -> None:
    matrix = build_binary_ablation_matrix(
        [
            _record(
                model_name="binary",
                batch_size=128,
                use_input_shortcut=False,
                use_triton_packed_inference=False,
                latency_ms=1.0,
                test_rmse=12.0,
            ),
            _record(
                model_name="binary",
                batch_size=128,
                use_input_shortcut=False,
                use_triton_packed_inference=True,
                latency_ms=0.5,
                test_rmse=12.0,
            ),
            _record(
                model_name="binary",
                batch_size=128,
                use_input_shortcut=True,
                use_triton_packed_inference=False,
                latency_ms=0.8,
                test_rmse=10.0,
            ),
            _record(
                model_name="binary",
                batch_size=128,
                use_input_shortcut=True,
                use_triton_packed_inference=True,
                latency_ms=0.4,
                test_rmse=10.0,
            ),
        ]
    )

    shortcut_triton_row = next(
        row
        for row in matrix
        if row["use_input_shortcut"] is True
        and row["use_triton_packed_inference"] is True
    )

    assert shortcut_triton_row["speedup_vs_same_shortcut_no_triton"] == 2.0
    assert shortcut_triton_row["rmse_delta_vs_no_shortcut_same_triton"] == -2.0


def test_build_model_inference_summary_groups_by_batch_size() -> None:
    summary = build_model_inference_summary(
        [
            _record(
                model_name="dense",
                batch_size=128,
                use_input_shortcut=False,
                use_triton_packed_inference=False,
                latency_ms=1.0,
                test_rmse=10.0,
            ),
            _record(
                model_name="binary",
                batch_size=256,
                use_input_shortcut=True,
                use_triton_packed_inference=False,
                latency_ms=0.9,
                test_rmse=11.0,
            ),
        ]
    )

    assert summary["candidate_count"] == 2
    assert [item["batch_size"] for item in summary["by_batch_size"]] == [128, 256]
    assert len(summary["binary_ablation_matrix"]) == 1