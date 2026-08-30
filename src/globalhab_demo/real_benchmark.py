"""Leakage-aware retrospective benchmark for Norwegian HAB monitoring.

The task is deliberately narrow: using only the current monitoring visit,
rank whether the next observed sample from the same region (1--14 days later)
meets the study-defined HAB criterion. This is not a continuous forecast.

v3.7 adds nested, training-only model selection. The preceding two years form
an inner validation window; the outer test window never tunes the model. The
reported ranking metric is Average Precision (AP), matching sklearn's
``average_precision_score`` implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "sst_c", "par_e_m2_d", "mixed_layer_depth_m",
    "sea_surface_salinity_psu", "month_sin", "month_cos",
]
ECO_INTERACTION_FEATURES = [
    "sst_x_par", "sst_over_mld", "par_over_mld", "salinity_x_sst",
]
MODEL_FEATURES = NUMERIC_FEATURES + ["region"]
FORWARD_WINDOWS = ((2012, 2013), (2014, 2015), (2016, 2017), (2018, 2019))

# Small, inspectable search space; selection is confined to each training era.
MODEL_CANDIDATES = (
    {"name": "reference_logistic", "c": .30, "positive_weight": 1.,
     "interactions": False, "half_life_years": None},
    {"name": "reference_recent5y", "c": .30, "positive_weight": 1.,
     "interactions": False, "half_life_years": 5.},
    {"name": "eco_interactions", "c": 1., "positive_weight": 1.,
     "interactions": True, "half_life_years": None},
    {"name": "eco_weight3", "c": 1., "positive_weight": 3.,
     "interactions": True, "half_life_years": None},
    {"name": "eco_weight5", "c": 1., "positive_weight": 5.,
     "interactions": True, "half_life_years": None},
    {"name": "eco_weight10", "c": 1., "positive_weight": 10.,
     "interactions": True, "half_life_years": None},
    {"name": "eco_weight5_recent5y", "c": 1., "positive_weight": 5.,
     "interactions": True, "half_life_years": 5.},
)


def prepare_next_sample_task(
    observations: pd.DataFrame,
    min_gap_days: int = 1,
    max_gap_days: int = 14,
) -> pd.DataFrame:
    """Create the next-observed-sample task without inventing missing weeks."""
    required = {
        "sample_date", "region", "target_hab_event", "sst_c", "par_e_m2_d",
        "mixed_layer_depth_m", "sea_surface_salinity_psu",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Norwegian benchmark is missing columns: {missing}")
    frame = observations.copy()
    frame["sample_date"] = pd.to_datetime(frame["sample_date"])
    frame = frame.sort_values(["region", "sample_date"], ignore_index=True)
    grouped = frame.groupby("region", sort=False)
    frame["next_sample_date"] = grouped["sample_date"].shift(-1)
    frame["next_sample_event"] = grouped["target_hab_event"].shift(-1)
    frame["gap_days"] = (frame["next_sample_date"] - frame["sample_date"]).dt.days
    frame = frame[
        frame["gap_days"].between(min_gap_days, max_gap_days)
        & frame["next_sample_event"].notna()
    ].copy()
    month = frame["sample_date"].dt.month.astype(float)
    frame["month_sin"] = np.sin(2. * np.pi * month / 12.)
    frame["month_cos"] = np.cos(2. * np.pi * month / 12.)
    frame["next_sample_event"] = frame["next_sample_event"].astype(bool)
    return frame


def _feature_frame(frame: pd.DataFrame, interactions: bool) -> pd.DataFrame:
    output = frame[MODEL_FEATURES].copy()
    if interactions:
        depth = 1. + output["mixed_layer_depth_m"].clip(lower=0.)
        output["sst_x_par"] = output["sst_c"] * output["par_e_m2_d"]
        output["sst_over_mld"] = output["sst_c"] / depth
        output["par_over_mld"] = output["par_e_m2_d"] / depth
        output["salinity_x_sst"] = output["sea_surface_salinity_psu"] * output["sst_c"]
    return output


def _model(candidate: dict[str, object]) -> object:
    numeric_features = NUMERIC_FEATURES + (
        ECO_INTERACTION_FEATURES if bool(candidate["interactions"]) else []
    )
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    preprocessor = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
    ])
    positive_weight = float(candidate["positive_weight"])
    class_weight = None if positive_weight == 1. else {0: 1., 1: positive_weight}
    return make_pipeline(
        preprocessor,
        LogisticRegression(
            C=float(candidate["c"]), class_weight=class_weight,
            max_iter=2000, random_state=42,
        ),
    )


def _fit_candidate(candidate: dict[str, object], train: pd.DataFrame) -> object:
    estimator = _model(candidate)
    fit_kwargs: dict[str, np.ndarray] = {}
    half_life = candidate["half_life_years"]
    if half_life is not None:
        age_years = (train["sample_date"].max() - train["sample_date"]).dt.days / 365.25
        fit_kwargs["logisticregression__sample_weight"] = np.power(
            .5, age_years.to_numpy(float) / float(half_life)
        )
    estimator.fit(
        _feature_frame(train, bool(candidate["interactions"])),
        train["next_sample_event"].astype(int),
        **fit_kwargs,
    )
    return estimator


def _correct_class_weight_prior(probability: np.ndarray, positive_weight: float) -> np.ndarray:
    """Undo the odds shift introduced by a fixed positive class weight."""
    if positive_weight == 1.:
        return probability
    denominator = positive_weight - (positive_weight - 1.) * probability
    return np.clip(probability / denominator, 0., 1.)


def _predict_candidate(estimator: object, candidate: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    probability = estimator.predict_proba(
        _feature_frame(frame, bool(candidate["interactions"]))
    )[:, 1]
    return _correct_class_weight_prior(probability, float(candidate["positive_weight"]))


def _select_candidate(train: pd.DataFrame, outer_start_year: int) -> tuple[dict[str, object], float]:
    """Select on the last two training years, never on outer test years."""
    inner_train = train[train["sample_date"].dt.year.lt(outer_start_year - 2)].copy()
    inner_validation = train[
        train["sample_date"].dt.year.between(outer_start_year - 2, outer_start_year - 1)
    ].copy()
    if (inner_train["next_sample_event"].nunique() < 2
            or inner_validation["next_sample_event"].nunique() < 2):
        return dict(MODEL_CANDIDATES[0]), float("nan")
    y_validation = inner_validation["next_sample_event"].astype(int).to_numpy()
    scored: list[tuple[float, int, dict[str, object]]] = []
    for index, candidate in enumerate(MODEL_CANDIDATES):
        estimator = _fit_candidate(candidate, inner_train)
        probability = _predict_candidate(estimator, candidate, inner_validation)
        scored.append((float(average_precision_score(y_validation, probability)), -index, candidate))
    best_score, _, best_candidate = max(scored, key=lambda item: (item[0], item[1]))
    return dict(best_candidate), best_score


def _seasonal_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train_month = train["sample_date"].dt.month
    test_month = test["sample_date"].dt.month
    prior = float(train["next_sample_event"].mean())
    grouped = train.assign(month=train_month).groupby("month")["next_sample_event"].agg(["sum", "count"])
    rates = (grouped["sum"] + 10. * prior) / (grouped["count"] + 10.)
    return test_month.map(rates).astype(float).fillna(prior).to_numpy()


def _top_fraction_metrics(y: np.ndarray, probability: np.ndarray, fraction: float) -> dict[str, float | int]:
    count = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(probability)[-count:]
    true_positives = int(y[selected].sum())
    positives = int(y.sum())
    precision = float(true_positives / count)
    prevalence = float(y.mean())
    return {
        "selected": count,
        "true_positives": true_positives,
        "false_positives": int(count - true_positives),
        "precision": precision,
        "recall": float(true_positives / positives) if positives else float("nan"),
        "precision_lift": float(precision / prevalence) if prevalence else float("nan"),
    }


def _metric_row(y: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "top10": _top_fraction_metrics(y, probability, .10),
        "top20": _top_fraction_metrics(y, probability, .20),
    }


def _bootstrap_ap_interval(y: np.ndarray, probability: np.ndarray, repeats: int = 500,
                           seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        sampled_y = y[index]
        if sampled_y.min() == sampled_y.max():
            continue
        values.append(float(average_precision_score(sampled_y, probability[index])))
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(values, [.025, .975]))


def run_forward_monitoring_benchmark(
    observations: pd.DataFrame,
    windows: tuple[tuple[int, int], ...] = FORWARD_WINDOWS,
    permutation_repeats: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Run nested expanding-window evaluation and return auditable evidence."""
    task = prepare_next_sample_task(observations)
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for start_year, end_year in windows:
        train = task[task["sample_date"].dt.year.lt(start_year)].copy()
        test = task[task["sample_date"].dt.year.between(start_year, end_year)].copy()
        if train["next_sample_event"].nunique() < 2 or test["next_sample_event"].nunique() < 2:
            continue
        candidate, inner_ap = _select_candidate(train, start_year)
        estimator = _fit_candidate(candidate, train)
        probability = _predict_candidate(estimator, candidate, test)
        reference_candidate = dict(MODEL_CANDIDATES[0])
        reference_estimator = _fit_candidate(reference_candidate, train)
        reference_probability = _predict_candidate(reference_estimator, reference_candidate, test)
        baseline_probability = _seasonal_baseline(train, test)
        y = test["next_sample_event"].astype(int).to_numpy()
        model_metrics = _metric_row(y, probability)
        reference_metrics = _metric_row(y, reference_probability)
        baseline_metrics = _metric_row(y, baseline_probability)
        fold_rows.append({
            "test_window": f"{start_year}–{end_year}", "train_end_year": start_year - 1,
            "train_samples": int(len(train)), "test_samples": int(len(test)),
            "test_events": int(y.sum()), "event_rate": float(y.mean()),
            "selected_model": str(candidate["name"]),
            "inner_validation_average_precision": inner_ap,
            "model_average_precision": model_metrics["average_precision"],
            "reference_average_precision": reference_metrics["average_precision"],
            "seasonal_average_precision": baseline_metrics["average_precision"],
            "model_brier": model_metrics["brier"], "seasonal_brier": baseline_metrics["brier"],
            "top10_recall": model_metrics["top10"]["recall"],
            "top10_precision": model_metrics["top10"]["precision"],
        })
        output = test[["sample_date", "next_sample_date", "gap_days", "region",
                       "next_sample_event", *NUMERIC_FEATURES]].copy()
        output["probability"] = probability
        output["reference_probability"] = reference_probability
        output["seasonal_probability"] = baseline_probability
        output["selected_model"] = str(candidate["name"])
        output["test_window"] = f"{start_year}–{end_year}"
        prediction_parts.append(output)

    if not prediction_parts:
        raise ValueError("No valid forward folds contained both event classes")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    y = predictions["next_sample_event"].astype(int).to_numpy()
    probability = predictions["probability"].to_numpy(float)
    reference_probability = predictions["reference_probability"].to_numpy(float)
    baseline_probability = predictions["seasonal_probability"].to_numpy(float)
    model_metrics = _metric_row(y, probability)
    reference_metrics = _metric_row(y, reference_probability)
    baseline_metrics = _metric_row(y, baseline_probability)
    ci_low, ci_high = _bootstrap_ap_interval(y, probability, seed=seed)

    rng = np.random.default_rng(seed)
    permuted_ap = np.asarray([
        average_precision_score(rng.permutation(y), probability)
        for _ in range(permutation_repeats)
    ], dtype=float)
    permutation_p = float((1 + np.sum(permuted_ap >= model_metrics["average_precision"]))
                          / (permutation_repeats + 1))
    model_ap = float(model_metrics["average_precision"])
    reference_ap = float(reference_metrics["average_precision"])
    seasonal_ap = float(baseline_metrics["average_precision"])
    top10 = model_metrics["top10"]
    top20 = model_metrics["top20"]
    weakest_index = folds["model_average_precision"].idxmin()
    strongest_index = folds["model_average_precision"].idxmax()
    summary = {
        "task": "next observed sample within 1–14 days",
        "evaluation": "nested model selection plus four expanding-window forward tests",
        "metric_name": "Average Precision (AP; sklearn average_precision_score)",
        "samples": int(len(predictions)), "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "no_information_average_precision": float(y.mean()),
        "model_average_precision": model_ap,
        "model_average_precision_ci95": [ci_low, ci_high],
        "reference_average_precision": reference_ap,
        "relative_improvement_over_reference": float(model_ap / reference_ap - 1.),
        "seasonal_average_precision": seasonal_ap,
        "average_precision_lift_vs_seasonal": float(model_ap / max(seasonal_ap, 1e-12)),
        "model_brier": model_metrics["brier"],
        "reference_brier": reference_metrics["brier"],
        "seasonal_brier": baseline_metrics["brier"],
        "top10_selected": top10["selected"], "top10_true_positives": top10["true_positives"],
        "top10_false_positives": top10["false_positives"], "top10_precision": top10["precision"],
        "top10_recall": top10["recall"], "top10_precision_lift": top10["precision_lift"],
        "top20_selected": top20["selected"], "top20_true_positives": top20["true_positives"],
        "top20_false_positives": top20["false_positives"], "top20_precision": top20["precision"],
        "top20_recall": top20["recall"], "top20_precision_lift": top20["precision_lift"],
        "reference_top10_recall": reference_metrics["top10"]["recall"],
        "folds_beating_seasonal_average_precision": int(
            folds["model_average_precision"].gt(folds["seasonal_average_precision"]).sum()
        ),
        "valid_folds": int(len(folds)),
        "weakest_fold": str(folds.loc[weakest_index, "test_window"]),
        "weakest_fold_average_precision": float(folds.loc[weakest_index, "model_average_precision"]),
        "strongest_fold": str(folds.loc[strongest_index, "test_window"]),
        "strongest_fold_average_precision": float(folds.loc[strongest_index, "model_average_precision"]),
        "permutation_p": permutation_p,
        "model_selection": (
            "Seven prespecified logistic candidates are ranked only on the final two years "
            "of each training era; the outer test window never tunes the model."
        ),
        "leakage_controls": [
            "each outer test window occurs strictly after all fitting and inner selection data",
            "candidate selection uses only the final two years inside the training era",
            "only current-visit environment, season and region are used",
            "current algae counts and future environmental values are excluded",
            "gaps longer than 14 days are excluded rather than labelled negative",
        ],
        "boundary": (
            "Retrospective next-observed-sample ranking benchmark; not a continuous "
            "14-day forecast, farm warning, causal model or regulatory threshold."
        ),
    }
    return {
        "task_frame": task, "predictions": predictions, "folds": folds,
        "summary": summary,
        "permutation_reference": pd.DataFrame({"permuted_average_precision": permuted_ap}),
    }
