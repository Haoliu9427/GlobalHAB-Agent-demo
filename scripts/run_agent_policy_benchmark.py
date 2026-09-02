"""Reproduce the equal-budget experiment-selection policy audit without rerunning heavy mechanism modules."""

from __future__ import annotations

import argparse
from itertools import product
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from globalhab_demo.agent import ExperimentAction  # noqa: E402
from globalhab_demo.bayesian_design import benchmark_agent_policies  # noqa: E402
from globalhab_demo.data import generate_demo_data  # noqa: E402
from globalhab_demo.experiment import evaluate_action, evaluate_baselines  # noqa: E402


def build_catalog(days: int, seed: int, holdout: str, test_fraction: float) -> pd.DataFrame:
    frame = generate_demo_data(days=days, seed=seed)
    baselines = evaluate_baselines(frame, holdout, test_fraction, seed)
    strongest_baseline = float(baselines["pr_auc"].max())
    rows = []
    for route, lag, model in product(
        ("local", "downstream"),
        (3, 7, 14, 21, 30, 45),
        ("logistic", "random_forest"),
    ):
        action = ExperimentAction(route, lag, model)
        feedback, _ = evaluate_action(
            frame,
            action,
            holdout,
            test_fraction,
            seed,
            strongest_baseline,
        )
        rows.append(feedback)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--holdout", default="Synthetic_Region_D")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()

    catalog = build_catalog(
        days=args.days,
        seed=args.seed,
        holdout=args.holdout,
        test_fraction=args.test_fraction,
    )
    audit = benchmark_agent_policies(
        catalog, budget=args.budget, seed=args.seed, repeats=args.repeats
    )
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    audit["summary"].to_csv(out / "agent_policy_benchmark_default.csv", index=False)
    audit["trajectory"].to_csv(out / "agent_policy_trajectories_default.csv", index=False)
    print(audit["summary"].to_string(index=False))
    print("\nSaved to outputs/agent_policy_benchmark_default.csv and agent_policy_trajectories_default.csv")


if __name__ == "__main__":
    main()
