"""Candidate experiments and leakage-resistant blocked validation.

The module deliberately separates ranking metrics, calibration metrics and
negative controls. All thresholds are capacity based (top 20% of predicted
risk), so held-out labels are never used to tune an alert cutoff.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import UPSTREAM


REVERSED_UPSTREAM = {value: key for key, value in UPSTREAM.items()}


def _lagged_signal(frame: pd.DataFrame, route: str, lag_days: int) -> pd.Series:
    source = frame[["date", "region", "mhw_intensity_c"]].copy()
    source["date"] = source["date"] + pd.to_timedelta(lag_days, unit="D")
    if route in {"downstream", "reversed"}:
        mapping = UPSTREAM if route == "downstream" else REVERSED_UPSTREAM
        parts = []
        for target, upstream in mapping.items():
            part = source[source["region"].eq(upstream)][
                ["date", "mhw_intensity_c"]
            ].copy()
            part["region"] = target
            parts.append(part)
        source = pd.concat(parts, ignore_index=True)
    elif route != "local":
        raise ValueError(f"unknown route: {route}")

    lookup = frame[["date", "region"]].merge(
        source[["date", "region", "mhw_intensity_c"]],
        on=["date", "region"],
        how="left",
    )
    signal = lookup["mhw_intensity_c"].to_numpy()
    proxy = frame["circulation_residence_proxy"].to_numpy()
    weighted = signal * (0.55 + 0.90 * proxy)
    return pd.Series(weighted, index=frame.index, name="candidate_signal")


def _experiment_frame(
    frame: pd.DataFrame,
    route: str,
    lag_days: int,
    control: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    work = frame.copy()
    work["candidate_signal"] = _lagged_signal(work, route, lag_days)
    if control == "within_region_time_permutation":
        rng = np.random.default_rng(seed + 991)
        work["candidate_signal"] = work.groupby("region")["candidate_signal"].transform(
            lambda values: rng.permutation(values.to_numpy())
        )
    elif control is not None:
        raise ValueError(f"unknown control: {control}")

    day = work["date"].dt.dayofyear.to_numpy()
    work["season_sin"] = np.sin(2 * np.pi * day / 365.25)
    work["season_cos"] = np.cos(2 * np.pi * day / 365.25)
    work["previous_event"] = (
        work.sort_values(["region", "date"])
        .groupby("region")["hab_event"]
        .shift(1)
        .reindex(work.index)
        .fillna(0)
    )
    return work.dropna(subset=["candidate_signal"]).reset_index(drop=True)


def _split(frame: pd.DataFrame, holdout_region: str, test_fraction: float):
    dates = np.sort(frame["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1.0 - test_fraction))])
    train = frame[frame["date"].lt(cut) & frame["region"].ne(holdout_region)].copy()
    test = frame[frame["date"].ge(cut) & frame["region"].eq(holdout_region)].copy()
    if train["hab_event"].nunique() < 2 or test["hab_event"].nunique() < 2:
        raise ValueError("blocked split must contain both classes")
    return train, test, cut


def _ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            value += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(y_true[mask].mean())
            )
    return float(value)


def _metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    reference_prevalence: float,
) -> tuple[dict[str, float], np.ndarray]:
    order = np.argsort(-probability, kind="stable")
    alert_count = max(1, int(np.ceil(len(probability) * 0.20)))
    alerted = np.zeros(len(probability), dtype=bool)
    alerted[order[:alert_count]] = True
    true_positive = int(((y_true == 1) & alerted).sum())
    false_positive = int(((y_true == 0) & alerted).sum())
    positives = max(1, int((y_true == 1).sum()))
    negatives = max(1, int((y_true == 0).sum()))
    brier = float(brier_score_loss(y_true, probability))
    reference = np.full(len(y_true), np.clip(reference_prevalence, 1e-4, 1 - 1e-4))
    reference_brier = float(brier_score_loss(y_true, reference))
    metrics = {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": brier,
        "brier_skill": float(1.0 - brier / reference_brier),
        "ece": _ece(y_true, probability),
        "recall_at_top20": float(true_positive / positives),
        "precision_at_top20": float(true_positive / alert_count),
        "false_alert_share_at_top20": float(false_positive / alert_count),
        "false_positive_rate_at_top20": float(false_positive / negatives),
        "alert_rate": float(alerted.mean()),
        "test_prevalence": float(y_true.mean()),
    }
    return metrics, alerted


def _model(name: str, seed: int, features: list[str] | None = None):
    if features is None:
        features = [
            "candidate_signal",
            "nitrate_mmol_m3",
            "phosphate_mmol_m3",
            "silicate_mmol_m3",
            "season_sin",
            "season_cos",
        ]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), features)
    ])
    if name == "logistic":
        estimator = LogisticRegression(
            max_iter=500, class_weight=None, random_state=seed
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)]), features


def evaluate_baselines(
    frame: pd.DataFrame,
    holdout_region: str,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate seasonal climatology and previous-event persistence."""
    work = _experiment_frame(frame, "local", 1)
    train, test, _ = _split(work, holdout_region, test_fraction)
    baseline_specs = {
        "季节气候态": ["season_sin", "season_cos"],
        "事件持续性": ["previous_event", "season_sin", "season_cos"],
    }
    rows = []
    for name, features in baseline_specs.items():
        model, _ = _model("logistic", seed, features)
        model.fit(train[features], train["hab_event"])
        probability = model.predict_proba(test[features])[:, 1]
        metrics, _ = _metrics(
            test["hab_event"].to_numpy(), probability, float(train["hab_event"].mean())
        )
        rows.append({"baseline": name, **metrics})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)


def evaluate_seasonal_baseline(
    frame: pd.DataFrame,
    holdout_region: str,
    test_fraction: float,
    seed: int,
) -> dict[str, float]:
    """Backward-compatible seasonal baseline helper."""
    rows = evaluate_baselines(frame, holdout_region, test_fraction, seed)
    row = rows[rows["baseline"].eq("季节气候态")].iloc[0]
    return row.drop("baseline").to_dict()


def evaluate_action(
    frame: pd.DataFrame,
    action,
    holdout_region: str,
    test_fraction: float,
    seed: int,
    baseline_pr_auc: float,
    control: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    work = _experiment_frame(frame, action.route, action.lag_days, control, seed)
    train, test, cut = _split(work, holdout_region, test_fraction)
    model, features = _model(action.model, seed)
    model.fit(train[features], train["hab_event"])
    probability = model.predict_proba(test[features])[:, 1]
    metrics, alerted = _metrics(
        test["hab_event"].to_numpy(), probability, float(train["hab_event"].mean())
    )
    cost_units = 1.0 if action.model == "logistic" else 2.5
    feedback = {
        **asdict(action),
        **metrics,
        "pr_auc_gain": metrics["pr_auc"] - baseline_pr_auc,
        "utility": (
            metrics["pr_auc"]
            - baseline_pr_auc
            + 0.10 * metrics["brier_skill"]
            - 0.05 * metrics["ece"]
            - 0.005 * cost_units
        ),
        "hypothesis": (
            f"{action.route}路径的{action.lag_days}天热异常信号与HAB风险相关，"
            f"由{action.model}检验"
        ),
        "control": control or "none",
        "status": "candidate" if metrics["pr_auc"] > baseline_pr_auc else "negative",
        "compute_cost_units": cost_units,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_events": int(test["hab_event"].sum()),
        "cut_date": cut.date().isoformat(),
    }
    predictions = test[["date", "region", "hab_event"]].copy()
    predictions["risk_probability"] = probability
    predictions["top20_alert"] = alerted.astype(int)
    for key, value in asdict(action).items():
        predictions[key] = value
    return feedback, predictions


def evaluate_negative_controls(
    frame: pd.DataFrame,
    best_action,
    holdout_region: str,
    test_fraction: float,
    seed: int,
    baseline_pr_auc: float,
) -> pd.DataFrame:
    """Run direction and temporal-permutation controls for the best candidate."""
    from .agent import ExperimentAction

    specs = [
        ("反向路径", ExperimentAction("reversed", best_action.lag_days, best_action.model), None),
        (
            "区内时间置换",
            ExperimentAction(best_action.route, best_action.lag_days, best_action.model),
            "within_region_time_permutation",
        ),
    ]
    rows = []
    for name, action, control in specs:
        feedback, _ = evaluate_action(
            frame,
            action,
            holdout_region,
            test_fraction,
            seed,
            baseline_pr_auc,
            control=control,
        )
        rows.append({"control_name": name, **feedback})
    return pd.DataFrame(rows)


def random_search_reference(
    catalog: pd.DataFrame,
    budget: int,
    seed: int,
    repeats: int = 200,
) -> dict[str, float]:
    """Compare guided search with random subsets of the same experiment catalog."""
    rng = np.random.default_rng(seed + 73)
    best_utilities = []
    recovered = []
    for _ in range(repeats):
        indices = rng.choice(len(catalog), size=budget, replace=False)
        chosen = catalog.iloc[indices]
        best = chosen.sort_values("utility", ascending=False).iloc[0]
        best_utilities.append(float(best["utility"]))
        recovered.append(best["route"] == "downstream" and int(best["lag_days"]) == 14)
    return {
        "repeats": float(repeats),
        "median_best_utility": float(np.median(best_utilities)),
        "p90_best_utility": float(np.quantile(best_utilities, 0.90)),
        "hidden_signal_recovery_rate": float(np.mean(recovered)),
    }
