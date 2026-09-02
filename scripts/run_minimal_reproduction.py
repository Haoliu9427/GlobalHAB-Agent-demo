"""Fast reproduction of the core 24-action / 8-step Agent experiment."""
from __future__ import annotations

from itertools import product
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.agent import ExperimentAction, HypothesisAgent  # noqa: E402
from globalhab_demo.data import generate_demo_data  # noqa: E402
from globalhab_demo.experiment import (  # noqa: E402
    evaluate_action,
    evaluate_baselines,
    evaluate_negative_controls,
    random_search_reference,
)


def main() -> None:
    days, seed, budget = 720, 42, 8
    holdout, test_fraction = "Synthetic_Region_D", 0.25
    frame = generate_demo_data(days=days, seed=seed)
    baselines = evaluate_baselines(frame, holdout, test_fraction, seed)
    strongest_baseline = float(baselines["pr_auc"].max())
    actions = [
        ExperimentAction(route, lag, model)
        for route, lag, model in product(
            ("local", "downstream"),
            (3, 7, 14, 21, 30, 45),
            ("logistic", "random_forest"),
        )
    ]
    feedback = {}
    for action in actions:
        row, _ = evaluate_action(
            frame, action, holdout, test_fraction, seed, strongest_baseline
        )
        feedback[action.action_id] = row
    catalog = pd.DataFrame(feedback.values())

    agent = HypothesisAgent(actions, budget)
    for _ in range(budget):
        action = agent.next_action()
        agent.observe(feedback[action.action_id])
    best = agent.best_result()
    best_action = ExperimentAction(str(best["route"]), int(best["lag_days"]), str(best["model"]))
    controls = evaluate_negative_controls(
        frame, best_action, holdout, test_fraction, seed, strongest_baseline
    )
    random_ref = random_search_reference(catalog, budget, seed)

    card = {
        "purpose": "fast reproduction of the core budgeted hypothesis-search loop",
        "candidate_count": len(actions),
        "budget": budget,
        "best_action": best_action.action_id,
        "best_average_precision": float(best["pr_auc"]),
        "hidden_truth_recovered": bool(best_action.route == "downstream" and best_action.lag_days == 14),
        "holdout_region": holdout,
        "time_block": "last 25% of dates",
        "random_search_equal_budget_recovery_rate": float(random_ref["hidden_signal_recovery_rate"]),
        "negative_controls_lower_than_candidate": bool((controls["pr_auc"] < float(best["pr_auc"])).all()),
        "boundary": "synthetic mechanism-recovery audit; not operational HAB forecast performance",
    }
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(agent.log).to_csv(out_dir / "minimal_agent_log.csv", index=False)
    (out_dir / "minimal_reproduction_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
