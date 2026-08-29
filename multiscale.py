"""Auditable multi-scale anomaly detection for the competition environment.

This is a competition-equivalent implementation: causal rolling references,
robust MAD standardisation, cross-scale agreement and event consolidation are
fully executable. It is intentionally independent from any production patent
code or tuned operational thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_SCALES = (7, 14, 30, 60)


def _causal_robust_z(values: pd.Series, window: int) -> pd.Series:
    """Score today against a past-only rolling median and MAD reference."""
    history = values.shift(1)
    min_periods = max(5, window // 2)
    median = history.rolling(window, min_periods=min_periods).median()
    absolute_deviation = (history - median).abs()
    mad = absolute_deviation.rolling(window, min_periods=min_periods).median()
    robust_scale = (1.4826 * mad).clip(lower=0.05)
    return ((values - median) / robust_scale).clip(-8.0, 8.0)


def _attach_event_ids(group: pd.DataFrame, gap_days: int = 2) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    active = group["anomaly_event"].eq(1)
    active_dates = group.loc[active, "date"]
    new_event = active_dates.diff().dt.days.fillna(gap_days + 1).gt(gap_days)
    ids = pd.Series(pd.NA, index=group.index, dtype="Int64")
    if active.any():
        ids.loc[active] = new_event.cumsum().astype("Int64").to_numpy()
    group["event_id"] = ids
    return group


def detect_multiscale_anomalies(
    frame: pd.DataFrame,
    scales: tuple[int, ...] = DEFAULT_SCALES,
    score_threshold: float = 1.35,
    minimum_scale_agreement: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily anomaly evidence and a consolidated event catalogue.

    SST anomalies are defined relative to the supplied climatological mean.
    Nutrient anomalies retain nitrate/phosphate/silicate separately before a
    transparent weighted context score is formed. Only past observations enter
    rolling references, which prevents future leakage.
    """
    required = {
        "date", "region", "sst_c", "climatological_mean_sst_c",
        "nitrate_mmol_m3", "phosphate_mmol_m3", "silicate_mmol_m3",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns for anomaly detection: {missing}")
    if not scales or any(scale < 5 for scale in scales):
        raise ValueError("scales must contain windows of at least 5 days")

    work = frame.sort_values(["region", "date"]).copy()
    work["sst_anomaly_c"] = work["sst_c"] - work["climatological_mean_sst_c"]
    source_columns = {
        "thermal": "sst_anomaly_c",
        "nitrate": "nitrate_mmol_m3",
        "phosphate": "phosphate_mmol_m3",
        "silicate": "silicate_mmol_m3",
    }

    for label, column in source_columns.items():
        for scale in scales:
            destination = f"{label}_z_{scale}d"
            work[destination] = work.groupby("region", group_keys=False)[column].apply(
                lambda values, scale=scale: _causal_robust_z(values, scale)
            )

    scale_scores = []
    for scale in scales:
        thermal = work[f"thermal_z_{scale}d"].clip(lower=0).fillna(0)
        nitrate = work[f"nitrate_z_{scale}d"].clip(lower=0).fillna(0)
        phosphate = work[f"phosphate_z_{scale}d"].clip(lower=0).fillna(0)
        silicate = work[f"silicate_z_{scale}d"].clip(lower=0).fillna(0)
        score = 0.55 * thermal + 0.20 * nitrate + 0.15 * phosphate + 0.10 * silicate
        column = f"anomaly_score_{scale}d"
        work[column] = score.clip(0, 8)
        scale_scores.append(column)

    work["scale_agreement"] = work[scale_scores].ge(score_threshold).sum(axis=1)
    work["multiscale_anomaly_score"] = work[scale_scores].median(axis=1)
    work["anomaly_event"] = (
        work["scale_agreement"].ge(minimum_scale_agreement)
        & work["multiscale_anomaly_score"].ge(score_threshold)
    ).astype(int)
    work = pd.concat(
        [_attach_event_ids(group) for _, group in work.groupby("region", sort=False)],
        ignore_index=True,
    )
    work = work.sort_values(["date", "region"], ignore_index=True)

    event_rows: list[dict[str, object]] = []
    active = work.dropna(subset=["event_id"])
    for (region, event_id), event in active.groupby(["region", "event_id"]):
        peak = event.loc[event["multiscale_anomaly_score"].idxmax()]
        event_rows.append({
            "region": region,
            "event_id": int(event_id),
            "start_date": event["date"].min(),
            "end_date": event["date"].max(),
            "duration_days": int((event["date"].max() - event["date"].min()).days + 1),
            "peak_date": peak["date"],
            "peak_score": float(peak["multiscale_anomaly_score"]),
            "maximum_scale_agreement": int(event["scale_agreement"].max()),
            "thermal_peak_z": float(max(peak[f"thermal_z_{scale}d"] for scale in scales)),
            "event_basis": f"median robust score across {list(scales)}d; past-only reference",
        })
    columns = [
        "region", "event_id", "start_date", "end_date", "duration_days",
        "peak_date", "peak_score", "maximum_scale_agreement", "thermal_peak_z",
        "event_basis",
    ]
    events = pd.DataFrame(event_rows, columns=columns)
    if not events.empty:
        events = events.sort_values("peak_score", ascending=False, ignore_index=True)
    return work, events
