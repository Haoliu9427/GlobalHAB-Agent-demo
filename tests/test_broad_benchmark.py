import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.broad_benchmark import benchmark_catalogue, run_broad_benchmark
from globalhab_demo.data import generate_demo_data


def test_broad_benchmark_uses_same_blocked_rows_and_many_methods():
    frame = generate_demo_data(days=540, seed=42)
    result = run_broad_benchmark(
        frame, lag_days=14, holdout_region="Synthetic_Region_D",
        test_fraction=0.25, include_deep=False,
    )
    summary = result["summary"]
    names = set(summary["model"])
    assert result["card"]["same_outer_rows"] is True
    assert len(summary) >= 15
    assert {"Logistic", "GAM (Spline Logistic)", "Random Forest", "XGBoost", "LightGBM", "STS-Interaction GLM"}.issubset(names)
    assert summary["test_rows"].nunique() == 1
    assert summary["test_events"].nunique() == 1
    assert len(benchmark_catalogue()) >= len(summary)
