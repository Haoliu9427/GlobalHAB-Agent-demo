"""Reproduce the bundled default science-structured model comparison.

The Streamlit page computes the same comparison from the current active run.
This standalone script uses the bundled default synthetic frame and discovery
card so it does not repeat the full TE/Durbin/real-evidence workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.sts_gated_tcn import run_dynamic_model_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.json")
    parser.add_argument("--data", default="data/demo_hab.csv")
    parser.add_argument("--card", default="outputs/discovery_card.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    card = json.loads((ROOT / args.card).read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / args.data, parse_dates=["date"])
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    lag_days = int(card["best_candidate"]["lag_days"])
    comparison = run_dynamic_model_comparison(
        frame=frame, lag_days=lag_days,
        holdout_region=config["holdout_region"], test_fraction=config["test_fraction"],
    )
    comparison["summary"].to_csv(output / "model_science_comparison_summary.csv", index=False)
    comparison["seed_results"].to_csv(output / "model_science_comparison_seed_results.csv", index=False)
    comparison["interaction_tuning_trace"].to_csv(output / "model_science_interaction_tuning.csv", index=False)
    comparison["tuning_trace"].to_csv(output / "model_science_tcn_tuning.csv", index=False)
    (output / "model_science_comparison_card.json").write_text(
        json.dumps(comparison["card"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
