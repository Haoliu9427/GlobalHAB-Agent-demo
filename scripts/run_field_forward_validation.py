"""Run the future cruise/farm forward-validation protocol on uploaded CSV data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from globalhab_demo.florida_sts import (
    DEFAULT_LAGS,
    normalize_current_frame,
    normalize_field_observations,
    project_next_sampling_candidates,
    run_forward_field_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--currents", required=True)
    parser.add_argument("--threshold", type=float, default=100_000.0)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    args = parser.parse_args()

    obs = normalize_field_observations(pd.read_csv(args.observations))
    currents = normalize_current_frame(pd.read_csv(args.currents))
    result = run_forward_field_validation(
        obs, currents, DEFAULT_LAGS, args.threshold, args.test_fraction
    )
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    card = {
        "status": result["status"],
        "quality": result["quality"],
        "selected_lag": result.get("selected_lag"),
        "cut_date": str(result.get("cut_date")) if result.get("cut_date") is not None else None,
        "test_summary": result.get("test_summary"),
        "boundary": result.get("boundary"),
    }
    (out / "field_forward_validation_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if "training_lags" in result:
        result["training_lags"].to_csv(out / "field_forward_training_lags.csv", index=False)
    if result.get("status") == "evaluated":
        result["test_pairs"].to_csv(out / "field_forward_test_pairs.csv", index=False)
        candidates = project_next_sampling_candidates(obs, currents, horizon_days=int(result["selected_lag"]))
        candidates.to_csv(out / "field_next_sampling_candidates.csv", index=False)
    print(json.dumps(card, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
