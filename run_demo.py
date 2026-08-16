"""Run the complete minimal GlobalHAB-Agent competition demo."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo import (  # noqa: E402
    ExperimentAction,
    HypothesisAgent,
    evaluate_action,
    evaluate_seasonal_baseline,
    generate_demo_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / args.output
    data_dir = ROOT / "data"
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    frame = generate_demo_data(config["days"], config["seed"])
    frame.to_csv(data_dir / "demo_hab.csv", index=False)
    baseline = evaluate_seasonal_baseline(
        frame,
        config["holdout_region"],
        config["test_fraction"],
        config["seed"],
    )

    actions = [
        ExperimentAction(route, lag, model)
        for route, lag, model in product(config["routes"], config["lags"], config["models"])
    ]
    agent = HypothesisAgent(actions, config["budget"])
    predictions_by_action: dict[str, pd.DataFrame] = {}
    for _ in range(config["budget"]):
        action = agent.next_action()
        feedback, predictions = evaluate_action(
            frame,
            action,
            config["holdout_region"],
            config["test_fraction"],
            config["seed"],
            baseline["pr_auc"],
        )
        agent.observe(feedback)
        predictions_by_action[action.action_id] = predictions

    log = pd.DataFrame(agent.log).sort_values("step")
    log.to_csv(output / "agent_log.csv", index=False)
    best = agent.best_result()
    best_predictions = predictions_by_action[str(best["action_id"])]
    best_predictions.to_csv(output / "risk_predictions.csv", index=False)
    test_events = int(best_predictions["hab_event"].sum())
    test_prevalence = float(best_predictions["hab_event"].mean())
    recovered_hidden_signal = (
        best["route"] == "downstream" and int(best["lag_days"]) == 14
    )

    discovery = {
        "demo_status": "synthetic_software_verification_only",
        "baseline": baseline,
        "best_candidate": best,
        "synthetic_ground_truth": {
            "route": "downstream",
            "lag_days": 14,
            "recovered_by_agent": recovered_hidden_signal,
        },
        "validation": {
            "time_block": "last 25% of dates",
            "spatial_block": config["holdout_region"],
            "random_split_used": False,
        },
        "interpretation": (
            "The selected action is a candidate signal found in deterministic synthetic data; "
            "it is not evidence of real-world HAB predictive skill or mechanism."
        ),
        "applicability_boundary": [
            "synthetic four-region dataset",
            "discrete upstream relation rather than physical current trajectories",
            "binary event label without species or toxin confirmation",
        ],
        "scientific_variable_rules": {
            "mhw_intensity": "SST minus seasonal climatological mean on synthetic p90 exceedance days",
            "nutrients": ["nitrate", "phosphate", "silicate"],
            "microplastics": "proxy for transport/residence/convergence setting only; not a direct HAB driver",
        },
        "excluded_research_modules": [
            "full multiscale anomaly detector",
            "deep adaptive router",
            "TE/CTE significance network",
            "spatial multiplier impact decomposition",
        ],
    }
    (output / "discovery_card.json").write_text(
        json.dumps(discovery, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    summary = (
        "# GlobalHAB-Agent 最小试跑摘要\n\n"
        f"- 实验预算：{config['budget']}\n"
        f"- 留出海区：{config['holdout_region']}\n"
        f"- 留出样本/事件：{len(best_predictions)}/{test_events}（事件率{test_prevalence:.1%}）\n"
        f"- 季节基线 PR-AUC：{baseline['pr_auc']:.3f}\n"
        f"- 最佳候选：{best['action_id']}\n"
        f"- 预设14天沿流信号恢复：{'是' if recovered_hidden_signal else '否'}\n"
        f"- 最佳候选 PR-AUC：{float(best['pr_auc']):.3f}\n"
        f"- 相对基线增量：{float(best['pr_auc_gain']):.3f}\n"
        f"- Brier：{float(best['brier']):.3f}\n\n"
        "> 以上仅为合成数据上的软件闭环验证，不作为真实 HAB 性能。\n"
    )
    (output / "run_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
