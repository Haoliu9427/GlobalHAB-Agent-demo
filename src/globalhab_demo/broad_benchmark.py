"""Broad leakage-safe model benchmark for the current GlobalHAB-Agent split.

The benchmark is intentionally wide so reviewers from statistics/economics and
computer science can recognise familiar reference methods.  All models use the
same downstream lagged experiment rows, the same completely held-out region and
the same forward test window.  Hyper-parameter selection, when used, is done
only inside the outer training era via a final 20% temporal validation block.

This module does not change the 24-candidate Agent search.  It is an independent
benchmark layer for the currently selected lag.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC

from .experiment import _experiment_frame, _metrics, _split
from .model_robustness import evaluate_current_lightweight_tcn
from .sts_gated_tcn import evaluate_current_interaction_glm, evaluate_current_sts_gated_tcn

try:  # optional at import time; requirements install them in the release build.
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - graceful deployment fallback
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


FEATURES = [
    "candidate_signal",
    "nitrate_mmol_m3",
    "phosphate_mmol_m3",
    "silicate_mmol_m3",
    "season_sin",
    "season_cos",
]
CORE_CONTINUOUS = [
    "candidate_signal",
    "nitrate_mmol_m3",
    "phosphate_mmol_m3",
    "silicate_mmol_m3",
]
SEEDS = (17, 42, 73)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    category: str
    idea: str
    candidates: tuple[dict, ...]
    builder: Callable[[dict, int], object]
    stochastic: bool = False
    complexity: str = "—"


def _scaled_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _plain_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def _gam_pipeline(c_value: float):
    transform = ColumnTransformer([
        ("smooth", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ("scale", StandardScaler()),
        ]), CORE_CONTINUOUS),
        ("season", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), ["season_sin", "season_cos"]),
    ])
    return Pipeline([
        ("transform", transform),
        ("model", LogisticRegression(max_iter=900, C=c_value)),
    ])


def _specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "Logistic", "统计基线", "线性概率模型",
            ({"C": 0.3}, {"C": 1.0}, {"C": 3.0}),
            lambda p, s: _scaled_pipeline(LogisticRegression(max_iter=700, C=p["C"], random_state=s)),
            complexity="6个输入特征",
        ),
        ModelSpec(
            "GAM (Spline Logistic)", "非线性统计", "平滑非线性响应 + 可解释概率模型",
            ({"C": 0.3}, {"C": 1.0}, {"C": 3.0}),
            lambda p, s: _gam_pipeline(p["C"]), complexity="4变量样条 + 季节项",
        ),
        ModelSpec(
            "Gaussian Naive Bayes", "经典机器学习", "条件独立概率分类",
            ({"var_smoothing": 1e-9}, {"var_smoothing": 1e-8}, {"var_smoothing": 1e-7}),
            lambda p, s: _scaled_pipeline(GaussianNB(var_smoothing=p["var_smoothing"])),
            complexity="闭式概率模型",
        ),
        ModelSpec(
            "kNN", "经典机器学习", "局部邻域相似性",
            ({"n_neighbors": 7}, {"n_neighbors": 15}, {"n_neighbors": 31}),
            lambda p, s: _scaled_pipeline(KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights="distance")),
            complexity="邻域模型",
        ),
        ModelSpec(
            "RBF-SVM", "经典机器学习", "核方法非线性边界",
            ({"C": 0.5, "gamma": "scale"}, {"C": 1.0, "gamma": "scale"}, {"C": 2.0, "gamma": "scale"}),
            lambda p, s: _scaled_pipeline(SVC(C=p["C"], gamma=p["gamma"], probability=True, class_weight="balanced", random_state=s)),
            stochastic=True, complexity="RBF核支持向量",
        ),
        ModelSpec(
            "Decision Tree", "经典机器学习", "单树规则分裂",
            ({"max_depth": 3}, {"max_depth": 5}, {"max_depth": 8}),
            lambda p, s: _plain_pipeline(DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=4, class_weight="balanced", random_state=s)),
            stochastic=True, complexity="1棵树",
        ),
        ModelSpec(
            "Random Forest", "树集成", "Bagging非线性集成",
            ({"max_depth": None}, {"max_depth": 7}),
            lambda p, s: _plain_pipeline(RandomForestClassifier(n_estimators=180, max_depth=p["max_depth"], min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=s)),
            stochastic=True, complexity="180棵树",
        ),
        ModelSpec(
            "Extra Trees", "树集成", "更强随机化树集成",
            ({"max_depth": None}, {"max_depth": 8}),
            lambda p, s: _plain_pipeline(ExtraTreesClassifier(n_estimators=180, max_depth=p["max_depth"], min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=s)),
            stochastic=True, complexity="180棵极随机树",
        ),
        ModelSpec(
            "AdaBoost", "Boosting强基线", "逐轮关注难样本",
            ({"n_estimators": 80, "learning_rate": 0.05}, {"n_estimators": 140, "learning_rate": 0.05}),
            lambda p, s: _plain_pipeline(AdaBoostClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], random_state=s)),
            stochastic=True, complexity="80–140弱学习器",
        ),
        ModelSpec(
            "Gradient Boosting", "Boosting强基线", "经典梯度提升树",
            ({"n_estimators": 100, "learning_rate": 0.05}, {"n_estimators": 160, "learning_rate": 0.04}),
            lambda p, s: _plain_pipeline(GradientBoostingClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=2, min_samples_leaf=4, random_state=s)),
            stochastic=True, complexity="100–160提升树",
        ),
        ModelSpec(
            "HistGradientBoosting", "Boosting强基线", "直方图梯度提升",
            ({"max_leaf_nodes": 15}, {"max_leaf_nodes": 31}),
            lambda p, s: _plain_pipeline(HistGradientBoostingClassifier(max_iter=140, learning_rate=0.05, max_leaf_nodes=p["max_leaf_nodes"], l2_regularization=0.2, random_state=s)),
            stochastic=True, complexity="140轮直方图提升",
        ),
        ModelSpec(
            "MLP", "神经网络", "浅层全连接非线性模型",
            ({"hidden": (12,), "alpha": 1e-3}, {"hidden": (24,), "alpha": 1e-3}),
            lambda p, s: _scaled_pipeline(MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"], max_iter=250, learning_rate_init=0.003, random_state=s)),
            stochastic=True, complexity="12或24隐藏单元",
        ),
    ]
    if XGBClassifier is not None:
        specs.append(ModelSpec(
            "XGBoost", "Boosting强基线", "成熟梯度提升树实现",
            ({"max_depth": 2, "learning_rate": 0.05}, {"max_depth": 3, "learning_rate": 0.04}),
            lambda p, s: XGBClassifier(
                n_estimators=160, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=3,
                eval_metric="logloss", random_state=s, n_jobs=2,
            ), stochastic=True, complexity="160棵提升树",
        ))
    if LGBMClassifier is not None:
        specs.append(ModelSpec(
            "LightGBM", "Boosting强基线", "高效叶子优先梯度提升",
            ({"num_leaves": 15}, {"num_leaves": 31}),
            lambda p, s: LGBMClassifier(
                n_estimators=160, learning_rate=0.04, num_leaves=p["num_leaves"],
                min_child_samples=20, subsample=0.85, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=s, n_jobs=2, verbosity=-1,
            ), stochastic=True, complexity="160棵提升树",
        ))
    return specs


def benchmark_catalogue() -> pd.DataFrame:
    rows = [
        ["Seasonal Climatology", "规则/统计基线", "仅季节周期"],
        ["Event Persistence", "规则/统计基线", "前一日事件 + 季节周期"],
    ]
    rows.extend([[s.name, s.category, s.idea] for s in _specs()])
    rows.extend([
        ["STS-Interaction GLM", "科学结构模型", "沿流传导 + 营养背景 + 显式交互"],
        ["Lightweight TCN", "时序深度模型", "轻量因果时序卷积"],
        ["STS-Gated TCN", "科学结构深度模型", "局地/上游双分支 + 输运门控"],
    ])
    return pd.DataFrame(rows, columns=["模型", "方法类别", "核心思想"])


def _inner_masks(work: pd.DataFrame, outer_train: np.ndarray):
    dates = np.sort(work.loc[outer_train, "date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * 0.80)])
    inner_train = outer_train & work["date"].lt(cut).to_numpy()
    inner_valid = outer_train & work["date"].ge(cut).to_numpy()
    return inner_train, inner_valid, cut


def _baseline_probability(train: pd.DataFrame, test: pd.DataFrame, name: str) -> np.ndarray:
    features = ["season_sin", "season_cos"] if name == "Seasonal Climatology" else ["previous_event", "season_sin", "season_cos"]
    model = _scaled_pipeline(LogisticRegression(max_iter=700))
    model.fit(train[features], train["hab_event"])
    return model.predict_proba(test[features])[:, 1]


def _select_candidate(spec: ModelSpec, work: pd.DataFrame, outer_train: np.ndarray) -> tuple[dict, pd.DataFrame]:
    inner_train, inner_valid, inner_cut = _inner_masks(work, outer_train)
    rows = []
    for i, params in enumerate(spec.candidates):
        model = spec.builder(params, 2026 + i)
        model.fit(work.loc[inner_train, FEATURES], work.loc[inner_train, "hab_event"])
        probability = model.predict_proba(work.loc[inner_valid, FEATURES])[:, 1]
        rows.append({
            "model": spec.name,
            "candidate": str(params),
            "candidate_index": i,
            "inner_validation_ap": float(average_precision_score(work.loc[inner_valid, "hab_event"], probability)),
            "inner_validation_brier": float(brier_score_loss(work.loc[inner_valid, "hab_event"], probability)),
            "inner_cut_date": inner_cut.date().isoformat(),
        })
    trace = pd.DataFrame(rows).sort_values(["inner_validation_ap", "inner_validation_brier", "candidate_index"], ascending=[False, True, True], ignore_index=True)
    return spec.candidates[int(trace.iloc[0]["candidate_index"])], trace


def _summarise_seed_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    out = []
    for model, g in frame.groupby("model", sort=False):
        first = g.iloc[0]
        out.append({
            "model": model,
            "category": first["category"],
            "ap": float(g["ap"].median()),
            "ap_sd": float(g["ap"].std(ddof=0)),
            "brier_skill": float(g["brier_skill"].mean()),
            "ece": float(g["ece"].mean()),
            "top20_recall": float(g["top20_recall"].mean()),
            "top20_precision": float(g["top20_precision"].mean()),
            "fit_seconds": float(g["fit_seconds"].mean()),
            "repeats": int(len(g)),
            "complexity": first["complexity"],
            "test_rows": int(first["test_rows"]),
            "test_events": int(first["test_events"]),
        })
    return pd.DataFrame(out).sort_values("ap", ascending=False, ignore_index=True)


def run_broad_benchmark(
    frame: pd.DataFrame,
    lag_days: int,
    holdout_region: str,
    test_fraction: float,
    include_deep: bool = True,
) -> dict[str, object]:
    """Evaluate a broad set of mature and science-structured methods."""
    work = _experiment_frame(frame, "downstream", lag_days)
    train, test, cut = _split(work, holdout_region, test_fraction)
    outer_train = work.index.isin(train.index).to_numpy() if hasattr(work.index.isin(train.index), 'to_numpy') else work.index.isin(train.index)
    outer_test = work.index.isin(test.index).to_numpy() if hasattr(work.index.isin(test.index), 'to_numpy') else work.index.isin(test.index)
    # _split preserves original work indices, so masks align exactly.
    outer_train = np.asarray(outer_train, dtype=bool)
    outer_test = np.asarray(outer_test, dtype=bool)
    y_train = work.loc[outer_train, "hab_event"].to_numpy()
    y_test = work.loc[outer_test, "hab_event"].to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("current blocked split must contain both event and non-event labels")
    prevalence = float(y_train.mean())
    rows: list[dict] = []
    tuning_frames = []

    # Exact same rows for simple baselines.
    for baseline in ("Seasonal Climatology", "Event Persistence"):
        start = perf_counter()
        probability = _baseline_probability(work.loc[outer_train], work.loc[outer_test], baseline)
        metrics, _ = _metrics(y_test, probability, prevalence)
        rows.append({
            "model": baseline, "category": "规则/统计基线", "seed": 0,
            "ap": metrics["pr_auc"], "brier_skill": metrics["brier_skill"], "ece": metrics["ece"],
            "top20_recall": metrics["recall_at_top20"], "top20_precision": metrics["precision_at_top20"],
            "fit_seconds": perf_counter() - start, "complexity": "2–3个解释变量",
            "test_rows": int(outer_test.sum()), "test_events": int(y_test.sum()),
        })

    for spec in _specs():
        selected, trace = _select_candidate(spec, work, outer_train)
        tuning_frames.append(trace)
        seeds = (42,)
        for seed in seeds:
            model = spec.builder(selected, seed)
            start = perf_counter()
            model.fit(work.loc[outer_train, FEATURES], y_train)
            probability = model.predict_proba(work.loc[outer_test, FEATURES])[:, 1]
            metrics, _ = _metrics(y_test, probability, prevalence)
            rows.append({
                "model": spec.name, "category": spec.category, "seed": seed,
                "ap": metrics["pr_auc"], "brier_skill": metrics["brier_skill"], "ece": metrics["ece"],
                "top20_recall": metrics["recall_at_top20"], "top20_precision": metrics["precision_at_top20"],
                "fit_seconds": perf_counter() - start, "complexity": spec.complexity,
                "test_rows": int(outer_test.sum()), "test_events": int(y_test.sum()),
            })

    # Always include the lightweight science-structured interaction model in the broad benchmark.
    interaction = evaluate_current_interaction_glm(frame, lag_days, holdout_region, test_fraction)
    rows.append({
        "model": "STS-Interaction GLM", "category": "科学结构模型", "seed": 42,
        "ap": float(interaction["pr_auc"]), "brier_skill": float(interaction["brier_skill"]),
        "ece": float(interaction["ece"]), "top20_recall": float(interaction["recall_at_top20"]),
        "top20_precision": float(interaction["precision_at_top20"]),
        "fit_seconds": float(interaction["fit_seconds"]), "complexity": "6个科学结构特征 + 显式交互",
        "test_rows": int(interaction["test_rows"]), "test_events": int(interaction["test_events"]),
    })

    # Optional live temporal-model extension.  Both architectures are fixed before
    # seeing the current outer-test labels, keeping this broad benchmark responsive.
    deep_details = None
    tcn_details = None
    if include_deep:
        tcn_details = evaluate_current_lightweight_tcn(
            frame, "downstream", lag_days, holdout_region, test_fraction, seed=42,
        )
        rows.append({
            "model": "Lightweight TCN", "category": "时序深度模型", "seed": 42,
            "ap": float(tcn_details["pr_auc"]), "brier_skill": float(tcn_details["brier_skill"]),
            "ece": float(tcn_details["ece"]), "top20_recall": float(tcn_details["recall_at_top20"]),
            "top20_precision": float(tcn_details["precision_at_top20"]), "fit_seconds": float(tcn_details["fit_seconds"]),
            "complexity": str(tcn_details["complexity"]), "test_rows": int(tcn_details["test_rows"]),
            "test_events": int(tcn_details["test_events"]),
        })
        deep_details = evaluate_current_sts_gated_tcn(
            frame, lag_days, holdout_region, test_fraction, seed=42,
        )
        rows.append({
            "model": "STS-Gated TCN", "category": "科学结构深度模型", "seed": 42,
            "ap": float(deep_details["pr_auc"]), "brier_skill": float(deep_details["brier_skill"]),
            "ece": float(deep_details["ece"]), "top20_recall": float(deep_details["recall_at_top20"]),
            "top20_precision": float(deep_details["precision_at_top20"]), "fit_seconds": float(deep_details["fit_seconds"]),
            "complexity": str(deep_details["complexity"]), "test_rows": int(deep_details["test_rows"]),
            "test_events": int(deep_details["test_events"]),
        })

    summary = _summarise_seed_rows(rows)
    tuning = pd.concat(tuning_frames, ignore_index=True) if tuning_frames else pd.DataFrame()
    return {
        "summary": summary,
        "seed_results": pd.DataFrame(rows),
        "tuning_trace": tuning,
        "catalogue": benchmark_catalogue(),
        "card": {
            "lag_days": int(lag_days), "holdout_region": holdout_region, "test_fraction": float(test_fraction),
            "cut_date": cut.date().isoformat(), "test_rows": int(outer_test.sum()), "test_events": int(y_test.sum()),
            "model_count": int(summary["model"].nunique()),
            "same_outer_rows": True,
            "selection_rule": "all hyper-parameter choices use only the final 20% of the outer training era",
        },
        "deep_details": deep_details,
        "tcn_details": tcn_details,
    }
