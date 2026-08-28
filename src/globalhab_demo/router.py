"""Transparent competition-equivalent adaptive method router.

The production deep router is not reproduced. This implementation exposes the
same competition-facing contract: diagnose data state, score analysis branches,
select a route and retain the reason and gate values for inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ROUTER_FEATURES = (
    "completeness", "sample_support", "temporal_dependence",
    "spatial_support", "event_support", "scale_consistency",
)


BRANCH_WEIGHTS = {
    "blocked_prediction": np.array([1.0, 0.9, 0.4, 0.2, 0.8, 0.3]),
    "multiscale_anomaly": np.array([0.8, 0.5, 0.8, 0.1, 0.2, 1.1]),
    "te_cte_network": np.array([0.8, 0.8, 1.1, 0.7, 0.8, 0.5]),
    "spatial_durbin": np.array([0.9, 0.9, 0.5, 1.3, 0.7, 0.4]),
}


def _lag1_autocorrelation(frame: pd.DataFrame, column: str) -> float:
    values = []
    for _, group in frame.sort_values("date").groupby("region"):
        value = group[column].autocorr(lag=1)
        if pd.notna(value):
            values.append(abs(float(value)))
    return float(np.mean(values)) if values else 0.0


def route_methods(
    frame: pd.DataFrame,
    anomaly_daily: pd.DataFrame,
    minimum_score: float = 0.52,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score and select compatible scientific analysis branches."""
    numeric = [
        "mhw_intensity_c", "nitrate_mmol_m3", "phosphate_mmol_m3",
        "silicate_mmol_m3", "circulation_residence_proxy", "hab_event",
    ]
    completeness = 1.0 - float(frame[numeric].isna().mean().mean())
    sample_support = float(np.clip(len(frame) / 2400.0, 0, 1))
    temporal_dependence = _lag1_autocorrelation(frame, "mhw_intensity_c")
    regions = int(frame["region"].nunique())
    spatial_support = float(np.clip((regions - 1) / 3.0, 0, 1))
    event_count = int(frame["hab_event"].sum())
    event_support = float(np.clip(event_count / 120.0, 0, 1))
    scale_consistency = float(
        anomaly_daily.loc[anomaly_daily["anomaly_event"].eq(1), "scale_agreement"].mean()
        / max(1, len([column for column in anomaly_daily if column.startswith("anomaly_score_")]))
    )
    if not np.isfinite(scale_consistency):
        scale_consistency = 0.0

    diagnostics = {
        "completeness": completeness,
        "sample_support": sample_support,
        "temporal_dependence": temporal_dependence,
        "spatial_support": spatial_support,
        "event_support": event_support,
        "scale_consistency": float(np.clip(scale_consistency, 0, 1)),
        "rows": float(len(frame)),
        "regions": float(regions),
        "events": float(event_count),
    }
    vector = np.array([diagnostics[name] for name in ROUTER_FEATURES])

    raw_scores = {
        branch: float(np.dot(weights, vector) / weights.sum())
        for branch, weights in BRANCH_WEIGHTS.items()
    }
    exp_scores = np.exp(np.array(list(raw_scores.values())) * 3.0)
    probabilities = exp_scores / exp_scores.sum()
    reasons = {
        "blocked_prediction": "样本与事件支持足够，保留时空阻断预测路径",
        "multiscale_anomaly": "时间依赖与跨尺度一致性支持事件检测",
        "te_cte_network": "时间依赖、事件数和有向空间图支持方向性检验",
        "spatial_durbin": "多海区覆盖与邻接矩阵支持直接/间接影响分解",
    }
    rows = []
    for probability, (branch, score) in zip(probabilities, raw_scores.items()):
        selected = score >= minimum_score
        rows.append({
            "branch": branch,
            "compatibility_score": score,
            "routing_probability": float(probability),
            "selected": bool(selected),
            "decision": "run" if selected else "defer",
            "reason": reasons[branch],
            **{f"gate_{name}": diagnostics[name] for name in ROUTER_FEATURES},
        })
    trace = pd.DataFrame(rows).sort_values(
        ["selected", "compatibility_score"], ascending=[False, False], ignore_index=True
    )
    return trace, diagnostics

