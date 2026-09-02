"""Reproduce the default broad model benchmark on the same blocked holdout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.broad_benchmark import run_broad_benchmark  # noqa: E402
from globalhab_demo.data import generate_demo_data  # noqa: E402


def main() -> None:
    frame = generate_demo_data(days=720, seed=42)
    result = run_broad_benchmark(
        frame,
        lag_days=14,
        holdout_region="Synthetic_Region_D",
        test_fraction=0.25,
        include_deep=True,
    )
    summary = result["summary"].sort_values("ap", ascending=False).reset_index(drop=True)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "broad_benchmark_default.csv", index=False)
    (out_dir / "broad_benchmark_default_card.json").write_text(
        json.dumps(result["card"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
