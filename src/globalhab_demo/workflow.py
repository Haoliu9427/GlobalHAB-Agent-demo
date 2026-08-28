"""Shared end-to-end exploration workflow for CLI, tests and web UI."""

from __future__ import annotations

from itertools import product

import pandas as pd

from .agent import ExperimentAction, HypothesisAgent
from .data import generate_demo_data
from .experiment import (
    evaluate_action,
    evaluate_baselines,
    evaluate_negative_controls,
    random_search_reference,
)
from .multiscale import detect_multiscale_anomalies
from .router import route_methods
from .spatial_durbin import estimate_spatial_durbin_impacts
from .transfer_entropy import estimate_te_cte_network, summarise_te_cte_by_lag


def run_exploration(
    days: int,
    seed: int,
    budget: int,
    holdout_region: str,
    test_fraction: float,
    routes: tuple[str, ...] = ("local", "downstream"),
    lags: tuple[int, ...] = (3, 7, 14, 21, 30, 45),
    models: tuple[str, ...] = ("logistic", "random_forest"),
) -> dict[str, object]:
    frame = generate_demo_data(days=days, seed=seed)
    anomaly_daily, anomaly_events = detect_multiscale_anomalies(frame)
    router_trace, router_diagnostics = route_methods(frame, anomaly_daily)
    te_cte_network = estimate_te_cte_network(frame, lags=lags, seed=seed)
    te_cte_lag_summary = summarise_te_cte_by_lag(te_cte_network)
    spatial_effects, spatial_diagnostics, spatial_weights = estimate_spatial_durbin_impacts(
        frame, anomaly_daily, seed=seed
    )
    baselines = evaluate_baselines(frame, holdout_region, test_fraction, seed)
    strongest_baseline = float(baselines["pr_auc"].max())
    actions = [
        ExperimentAction(route, lag, model)
        for route, lag, model in product(routes, lags, models)
    ]
    if budget > len(actions):
        raise ValueError("budget cannot exceed the action catalog")

    # Evaluate the fixed catalog once. The agent only receives feedback for the
    # actions it selects; the complete catalog is retained for the equal-budget
    # random-search reference.
    feedback_by_action: dict[str, dict[str, object]] = {}
    predictions_by_action: dict[str, pd.DataFrame] = {}
    for action in actions:
        feedback, predictions = evaluate_action(
            frame,
            action,
            holdout_region,
            test_fraction,
            seed,
            strongest_baseline,
        )
        feedback_by_action[action.action_id] = feedback
        predictions_by_action[action.action_id] = predictions

    catalog = pd.DataFrame(feedback_by_action.values())
    agent = HypothesisAgent(actions, budget)
    for _ in range(budget):
        action = agent.next_action()
        agent.observe(feedback_by_action[action.action_id])

    log = pd.DataFrame(agent.log).sort_values("step", ignore_index=True)
    best = agent.best_result()
    best_action = ExperimentAction(
        str(best["route"]), int(best["lag_days"]), str(best["model"])
    )
    predictions = predictions_by_action[best_action.action_id]
    controls = evaluate_negative_controls(
        frame,
        best_action,
        holdout_region,
        test_fraction,
        seed,
        strongest_baseline,
    )
    random_reference = random_search_reference(catalog, budget, seed)
    recovered = best_action.route == "downstream" and best_action.lag_days == 14
    controls_lower = bool((controls["pr_auc"] < float(best["pr_auc"])).all())
    baseline_gain = float(best["pr_auc"]) - strongest_baseline
    signal_status = (
        "通过合成真值恢复测试"
        if recovered and controls_lower and baseline_gain > 0
        else "未通过，保留为负结果"
    )

    card = {
        "demo_status": "synthetic_software_verification_plus_external_case_card",
        "research_signal_status": signal_status,
        "best_candidate": best,
        "synthetic_ground_truth": {
            "route": "downstream",
            "lag_days": 14,
            "recovered_by_agent": recovered,
        },
        "minimum_references": {
            "strongest_simple_baseline_pr_auc": strongest_baseline,
            "random_search_equal_budget": random_reference,
            "negative_controls_lower_than_candidate": controls_lower,
        },
        "validation": {
            "time_block": f"last {test_fraction:.0%} of dates",
            "spatial_block": holdout_region,
            "random_split_used": False,
            "alert_definition": "top 20% predicted risk; no held-out-label threshold tuning",
        },
        "scientific_variable_rules": {
            "mhw_intensity": "SST - seasonal climatological mean on p90 exceedance days",
            "nutrients": ["nitrate", "phosphate", "silicate"],
            "microplastics": (
                "transport/residence/convergence-state proxy only; "
                "not current velocity/direction or a direct HAB driver"
            ),
        },
        "applicability_boundary": [
            "anonymous synthetic training data",
            "abstract directed pathway rather than a physical current trajectory",
            "binary synthetic label without species or toxin confirmation",
            "aquaculture module is response prioritisation, not loss or closure prediction",
        ],
        "competition_equivalent_modules": {
            "multiscale_anomaly_detection": {
                "scales_days": [7, 14, 30, 60],
                "past_only_reference": True,
                "detected_events": int(len(anomaly_events)),
            },
            "adaptive_router": {
                "selected_branches": router_trace.loc[router_trace["selected"], "branch"].tolist(),
                "diagnostics": router_diagnostics,
            },
            "te_cte_network": {
                "estimator": "discrete conditional mutual information in bits",
                "permutation_control": "circular source shift",
                "fdr": "Benjamini-Hochberg",
                "peak_mean_cte_lag_days": int(
                    te_cte_lag_summary.loc[te_cte_lag_summary["mean_cte_bits"].idxmax(), "lag_days"]
                ),
            },
            "spatial_durbin_decomposition": spatial_diagnostics,
            "implementation_boundary": (
                "fully runnable competition-equivalent methods; not a line-by-line disclosure "
                "of the filed patent production implementation"
            ),
        },
    }
    return {
        "frame": frame,
        "baselines": baselines,
        "catalog": catalog,
        "log": log,
        "best": best,
        "predictions": predictions,
        "controls": controls,
        "random_reference": random_reference,
        "card": card,
        "anomaly_daily": anomaly_daily,
        "anomaly_events": anomaly_events,
        "router_trace": router_trace,
        "router_diagnostics": router_diagnostics,
        "te_cte_network": te_cte_network,
        "te_cte_lag_summary": te_cte_lag_summary,
        "spatial_effects": spatial_effects,
        "spatial_diagnostics": spatial_diagnostics,
        "spatial_weights": spatial_weights,
    }
