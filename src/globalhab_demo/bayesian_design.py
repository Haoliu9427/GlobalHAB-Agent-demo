"""Bayesian sequential experimental-design policies for the fixed Agent catalog.

The scientific environment remains unchanged: 24 route x lag x model actions and a
fixed experiment budget.  This module changes only the *selection policy* used to
choose the next untested action.  A small Gaussian-process surrogate is updated
from observed experiment utilities.  Expected improvement, predictive information
and Thompson sampling are compared with the existing transparent heuristic and an
equal-budget random policy.

No hidden ground-truth label is used by any acquisition function.  Ground truth is
consulted only after a trajectory has finished, for software-verification scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .agent import ExperimentAction, HypothesisAgent


@dataclass(frozen=True)
class PolicyRun:
    policy: str
    trajectory: pd.DataFrame
    recovered_hidden_signal: bool
    first_hidden_step: int | None
    best_utility: float


def _action_features(catalog: pd.DataFrame) -> np.ndarray:
    lag = (catalog["lag_days"].to_numpy(float) - 3.0) / 42.0
    route = catalog["route"].eq("downstream").to_numpy(float)
    model = catalog["model"].eq("random_forest").to_numpy(float)
    # Pre-registered low-dimensional scientific action representation.  The
    # representation contains no knowledge that 14 days is the hidden truth.
    return np.column_stack([lag, route, model, lag * route, lag * lag])


def _gp(seed: int) -> GaussianProcessRegressor:
    # Fixed hyperparameters keep the policy audit deterministic and prevent
    # acquisition-function tuning on the held-out experiment landscape.
    kernel = (
        ConstantKernel(0.30, constant_value_bounds="fixed")
        * RBF([0.18, 0.80, 0.80, 0.25, 0.25], length_scale_bounds="fixed")
        + WhiteKernel(0.01, noise_level_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        optimizer=None,
        random_state=seed,
    )


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / sqrt(2.0 * pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _common_start_index(catalog: pd.DataFrame) -> int:
    preferred = catalog.index[
        catalog["route"].eq("local")
        & catalog["lag_days"].eq(7)
        & catalog["model"].eq("logistic")
    ]
    return int(preferred[0]) if len(preferred) else int(catalog.index[0])


def _score_finished(catalog: pd.DataFrame, chosen: list[int], policy: str) -> PolicyRun:
    trajectory = catalog.loc[chosen].copy().reset_index(drop=True)
    trajectory.insert(0, "step", np.arange(1, len(trajectory) + 1))
    trajectory["action_id"] = (
        trajectory["route"].astype(str) + "__"
        + trajectory["lag_days"].astype(int).astype(str) + "d__"
        + trajectory["model"].astype(str)
    )
    hidden = trajectory["route"].eq("downstream") & trajectory["lag_days"].eq(14)
    first_step = int(trajectory.loc[hidden, "step"].iloc[0]) if hidden.any() else None
    best = trajectory.sort_values("utility", ascending=False).iloc[0]
    recovered = bool(best["route"] == "downstream" and int(best["lag_days"]) == 14)
    return PolicyRun(
        policy=policy,
        trajectory=trajectory,
        recovered_hidden_signal=recovered,
        first_hidden_step=first_step,
        best_utility=float(best["utility"]),
    )


def run_bayesian_policy(
    catalog: pd.DataFrame,
    budget: int,
    policy: str = "bayesian_ei",
    seed: int = 42,
) -> PolicyRun:
    """Run one sequential Bayesian policy on an already evaluated action catalog.

    The catalog acts as the environment.  At each step the policy can observe the
    utility only for actions it has already selected.  Unselected utilities remain
    hidden from the acquisition rule.
    """
    if budget < 1 or budget > len(catalog):
        raise ValueError("budget must be within catalog size")
    if policy not in {"bayesian_ei", "bayesian_eig", "thompson"}:
        raise ValueError(f"unknown Bayesian policy: {policy}")

    work = catalog.reset_index(drop=True).copy()
    x = _action_features(work)
    rng = np.random.default_rng(seed)
    chosen = [_common_start_index(work)]

    while len(chosen) < budget:
        remaining = np.array([i for i in range(len(work)) if i not in chosen], dtype=int)
        observed = np.array(chosen, dtype=int)
        y = work.loc[observed, "utility"].to_numpy(float)
        model = _gp(seed + len(chosen))
        model.fit(x[observed], y)
        mean, std = model.predict(x[remaining], return_std=True)
        std = np.maximum(std, 1e-8)

        if policy == "bayesian_ei":
            improvement = mean - float(np.max(y)) - 0.005
            z = improvement / std
            acquisition = improvement * _normal_cdf(z) + std * _normal_pdf(z)
        elif policy == "bayesian_eig":
            # Entropy reduction proxy for one noisy scalar utility observation.
            # A tiny exploitation term resolves ties without using hidden truth.
            info_gain = 0.5 * np.log1p((std * std) / 0.02)
            acquisition = info_gain + 0.05 * mean
        else:  # Thompson sampling
            acquisition = rng.normal(mean, std)

        next_index = int(remaining[int(np.argmax(acquisition))])
        chosen.append(next_index)

    return _score_finished(work, chosen, policy)


def run_heuristic_policy(catalog: pd.DataFrame, budget: int) -> PolicyRun:
    work = catalog.reset_index(drop=True).copy()
    actions = [
        ExperimentAction(str(row.route), int(row.lag_days), str(row.model))
        for row in work.itertuples(index=False)
    ]
    lookup = {
        f"{row.route}__{int(row.lag_days)}d__{row.model}": row._asdict()
        for row in work.itertuples(index=False)
    }
    agent = HypothesisAgent(actions, budget)
    chosen_ids: list[str] = []
    for _ in range(budget):
        action = agent.next_action()
        feedback = dict(lookup[action.action_id])
        agent.observe(feedback)
        chosen_ids.append(action.action_id)
    id_to_index = {
        f"{row.route}__{int(row.lag_days)}d__{row.model}": int(i)
        for i, row in work.iterrows()
    }
    return _score_finished(work, [id_to_index[x] for x in chosen_ids], "current_heuristic")


def run_random_policy(catalog: pd.DataFrame, budget: int, seed: int) -> PolicyRun:
    work = catalog.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    start = _common_start_index(work)
    remaining = np.array([i for i in range(len(work)) if i != start], dtype=int)
    sample = rng.choice(remaining, size=max(0, budget - 1), replace=False).tolist()
    return _score_finished(work, [start, *map(int, sample)], "random")


def benchmark_agent_policies(
    catalog: pd.DataFrame,
    budget: int,
    seed: int = 42,
    repeats: int = 40,
) -> dict[str, object]:
    """Compare selection policies on the same current experiment landscape."""
    deterministic = [
        run_heuristic_policy(catalog, budget),
        run_bayesian_policy(catalog, budget, "bayesian_ei", seed),
        run_bayesian_policy(catalog, budget, "bayesian_eig", seed),
    ]

    stochastic_rows: list[dict[str, object]] = []
    trajectories: list[pd.DataFrame] = []
    for result in deterministic:
        stochastic_rows.append({
            "policy": result.policy,
            "recovery_rate": float(result.recovered_hidden_signal),
            "median_first_hidden_step": float(result.first_hidden_step) if result.first_hidden_step else np.nan,
            "median_best_utility": result.best_utility,
            "repeats": 1,
        })
        tr = result.trajectory.copy()
        tr["policy"] = result.policy
        trajectories.append(tr)

    for policy in ["thompson", "random"]:
        runs: list[PolicyRun] = []
        for i in range(repeats):
            if policy == "thompson":
                run = run_bayesian_policy(catalog, budget, "thompson", seed + 1000 + i)
            else:
                run = run_random_policy(catalog, budget, seed + 2000 + i)
            runs.append(run)
        first_steps = [r.first_hidden_step for r in runs if r.first_hidden_step is not None]
        stochastic_rows.append({
            "policy": policy,
            "recovery_rate": float(np.mean([r.recovered_hidden_signal for r in runs])),
            "median_first_hidden_step": float(np.median(first_steps)) if first_steps else np.nan,
            "median_best_utility": float(np.median([r.best_utility for r in runs])),
            "repeats": repeats,
        })
        # retain one representative trajectory near median best utility
        median_u = float(np.median([r.best_utility for r in runs]))
        representative = min(runs, key=lambda r: abs(r.best_utility - median_u))
        tr = representative.trajectory.copy()
        tr["policy"] = policy
        trajectories.append(tr)

    summary = pd.DataFrame(stochastic_rows)
    labels = {
        "current_heuristic": "当前受约束策略",
        "bayesian_ei": "Bayesian Expected Improvement",
        "bayesian_eig": "Bayesian Information Gain",
        "thompson": "Thompson Sampling",
        "random": "Random",
    }
    summary["policy_label"] = summary["policy"].map(labels)
    summary = summary.sort_values(
        ["recovery_rate", "median_first_hidden_step"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    trajectory = pd.concat(trajectories, ignore_index=True)
    trajectory["policy_label"] = trajectory["policy"].map(labels)
    return {
        "summary": summary,
        "trajectory": trajectory,
        "boundary": (
            "This benchmark changes only the action-selection policy on the current evaluated "
            "synthetic experiment landscape. Hidden truth is used only after a trajectory ends."
        ),
    }
