"""Candidate experiment construction and leakage-safe blocked validation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import UPSTREAM


def _lagged_signal(frame: pd.DataFrame, route: str, lag_days: int) -> pd.Series:
    source = frame[["date", "region", "mhw_intensity_c"]].copy()
    source["date"] = source["date"] + pd.to_timedelta(lag_days, unit="D")
    if route == "downstream":
        # Multiple targets may share an upstream region. A direct target-wise
        # lookup is clearer and avoids silently dropping duplicate pathways.
        parts = []
        for target, upstream in UPSTREAM.items():
            part = source[source["region"].eq(upstream)][["date", "mhw_intensity_c"]].copy()
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
    # MPs weight only the synthetic transmission setting. They are neither a
    # current-speed/direction measurement nor a direct HAB driver.
    weighted = signal * (0.55 + 0.90 * proxy)
    return pd.Series(weighted, index=frame.index, name="candidate_signal")


def _experiment_frame(frame: pd.DataFrame, route: str, lag_days: int) -> pd.DataFrame:
    work = frame.copy()
    work["candidate_signal"] = _lagged_signal(work, route, lag_days)
    day = work["date"].dt.dayofyear.to_numpy()
    work["season_sin"] = np.sin(2 * np.pi * day / 365.25)
    work["season_cos"] = np.cos(2 * np.pi * day / 365.25)
    return work.dropna(subset=["candidate_signal"]).reset_index(drop=True)


def _split(frame: pd.DataFrame, holdout_region: str, test_fraction: float):
    dates = np.sort(frame["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1.0 - test_fraction))])
    train = frame[frame["date"].lt(cut) & frame["region"].ne(holdout_region)].copy()
    test = frame[frame["date"].ge(cut) & frame["region"].eq(holdout_region)].copy()
    if train["hab_event"].nunique() < 2 or test["hab_event"].nunique() < 2:
        raise ValueError("blocked split must contain both classes")
    return train, test, cut


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    threshold = float(np.quantile(probability, 0.80))
    predicted = probability >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "recall_at_top20": float(recall_score(y_true, predicted, zero_division=0)),
        "alert_rate": float(predicted.mean()),
    }


def _model(name: str, seed: int):
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
        estimator = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
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


def evaluate_seasonal_baseline(
    frame: pd.DataFrame,
    holdout_region: str,
    test_fraction: float,
    seed: int,
) -> dict[str, float]:
    work = frame.copy()
    day = work["date"].dt.dayofyear.to_numpy()
    work["season_sin"] = np.sin(2 * np.pi * day / 365.25)
    work["season_cos"] = np.cos(2 * np.pi * day / 365.25)
    train, test, _ = _split(work, holdout_region, test_fraction)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
    ])
    columns = ["season_sin", "season_cos"]
    model.fit(train[columns], train["hab_event"])
    probability = model.predict_proba(test[columns])[:, 1]
    return _metrics(test["hab_event"].to_numpy(), probability)


def evaluate_action(
    frame: pd.DataFrame,
    action,
    holdout_region: str,
    test_fraction: float,
    seed: int,
    baseline_pr_auc: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    work = _experiment_frame(frame, action.route, action.lag_days)
    train, test, cut = _split(work, holdout_region, test_fraction)
    model, features = _model(action.model, seed)
    model.fit(train[features], train["hab_event"])
    probability = model.predict_proba(test[features])[:, 1]
    metrics = _metrics(test["hab_event"].to_numpy(), probability)
    feedback = {
        **asdict(action),
        **metrics,
        "pr_auc_gain": metrics["pr_auc"] - baseline_pr_auc,
        "utility": metrics["pr_auc"] - baseline_pr_auc - 0.15 * metrics["brier"],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cut_date": cut.date().isoformat(),
    }
    predictions = test[["date", "region", "hab_event"]].copy()
    predictions["risk_probability"] = probability
    for key, value in asdict(action).items():
        predictions[key] = value
    return feedback, predictions
