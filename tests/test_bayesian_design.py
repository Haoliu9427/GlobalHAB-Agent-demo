import pandas as pd

from globalhab_demo.bayesian_design import benchmark_agent_policies


def _toy_catalog():
    rows = []
    for route in ["local", "downstream"]:
        for lag in [3, 7, 14, 21, 30, 45]:
            for model in ["logistic", "random_forest"]:
                utility = 0.50 - 0.005 * abs(lag - 14)
                utility += 0.20 if route == "downstream" else 0.0
                utility += 0.01 if model == "random_forest" else 0.0
                if route == "downstream" and lag == 14:
                    utility = 1.0 + (0.01 if model == "random_forest" else 0.0)
                rows.append({
                    "route": route,
                    "lag_days": lag,
                    "model": model,
                    "utility": utility,
                    "pr_auc": utility,
                    "brier_skill": 0.2,
                    "ece": 0.05,
                    "status": "ok",
                })
    return pd.DataFrame(rows)


def test_policy_benchmark_uses_same_budget_and_reports_multiple_strategies():
    result = benchmark_agent_policies(_toy_catalog(), budget=8, seed=42, repeats=8)
    summary = result["summary"]
    trajectory = result["trajectory"]
    assert {
        "current_heuristic", "bayesian_ei", "bayesian_eig", "thompson", "random"
    }.issubset(set(summary["policy"]))
    assert summary["recovery_rate"].between(0.0, 1.0).all()
    assert trajectory.groupby("policy")["step"].max().eq(8).all()
    assert trajectory["action_id"].notna().all()
