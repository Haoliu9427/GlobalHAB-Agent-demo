"""Runnable replay of the 2025 South Australia Karenia event.

The replay uses the Murray et al. qPCR observations bundled under CC BY 4.0.
It provides descriptive spatiotemporal evidence and explicitly defers analyses
that the single, non-uniformly sampled event cannot support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aquaculture import PRODUCTION_PROFILES


SPECIES = {
    "k_cristata_cells_l": "K. cristata",
    "k_papilionacea_cells_l": "K. papilionacea",
    "k_mikimotoi_cells_l": "K. mikimotoi",
    "k_brevisulcata_cells_l": "K. brevisulcata",
    "k_longicanalis_cells_l": "K. longicanalis",
    "k_brevis_cells_l": "K. brevis",
    "k_hui_cells_l": "K. hui",
}


def load_sa_real_case(data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    derived = data_root / "real_case" / "derived"
    observations = pd.read_csv(
        derived / "sa_qpcr_observations.csv", parse_dates=["sample_date"]
    )
    provenance = json.loads((derived / "provenance.json").read_text(encoding="utf-8"))
    return observations, provenance


def _abundance_band(value: float) -> str:
    if value <= 0:
        return "未检出/报告为0"
    if value < 1e4:
        return "<10⁴ cells L⁻¹"
    if value < 1e5:
        return "10⁴–10⁵ cells L⁻¹"
    if value < 1e6:
        return "10⁵–10⁶ cells L⁻¹"
    return "≥10⁶ cells L⁻¹"


def build_sa_replay(
    observations: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    depth: str = "全部深度",
) -> dict[str, object]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    selected = observations[
        observations["sample_date"].between(start, end, inclusive="both")
    ].copy()
    if depth != "全部深度":
        selected = selected[selected["depth"].eq(depth)].copy()
    if selected.empty:
        raise ValueError("selected replay window has no qPCR observations")

    timeline = selected.groupby("sample_date", as_index=False).agg(
        samples=("source_row", "size"),
        locations=("location", "nunique"),
        k_cristata_peak_cells_l=("k_cristata_cells_l", "max"),
        k_cristata_median_cells_l=("k_cristata_cells_l", "median"),
        karenia_peak_cells_l=("karenia_total_cells_l", "max"),
        k_cristata_detection_share=("k_cristata_detected", "mean"),
    )
    site_summary = selected.groupby(
        ["location", "latitude", "longitude"], as_index=False
    ).agg(
        samples=("source_row", "size"),
        sampling_dates=("sample_date", "nunique"),
        first_sample=("sample_date", "min"),
        last_sample=("sample_date", "max"),
        k_cristata_peak_cells_l=("k_cristata_cells_l", "max"),
        karenia_peak_cells_l=("karenia_total_cells_l", "max"),
        k_cristata_detection_share=("k_cristata_detected", "mean"),
        k_cristata_max_share=("k_cristata_share", "max"),
    )
    site_summary["observed_abundance_band"] = site_summary[
        "k_cristata_peak_cells_l"
    ].map(_abundance_band)
    site_summary["marker_size"] = 8 + 7 * np.log10(
        1 + site_summary["k_cristata_peak_cells_l"]
    )
    site_summary = site_summary.sort_values(
        "k_cristata_peak_cells_l", ascending=False, ignore_index=True
    )

    species_rows = []
    for column, label in SPECIES.items():
        species_rows.append({
            "species": label,
            "summed_cells_l_across_samples": float(selected[column].sum()),
            "peak_cells_l": float(selected[column].max()),
            "detected_samples": int(selected[column].gt(0).sum()),
            "detection_share": float(selected[column].gt(0).mean()),
        })
    species = pd.DataFrame(species_rows).sort_values(
        "summed_cells_l_across_samples", ascending=False, ignore_index=True
    )
    peak = selected.loc[selected["k_cristata_cells_l"].idxmax()]
    card = {
        "mode": "real_event_replay_not_supervised_training",
        "event": "South Australia complex Karenia bloom, 2025",
        "observations": int(len(selected)),
        "sampling_dates": int(selected["sample_date"].nunique()),
        "locations": int(selected["location"].nunique()),
        "date_range": [selected["sample_date"].min().date().isoformat(), selected["sample_date"].max().date().isoformat()],
        "peak_k_cristata": {
            "date": peak["sample_date"].date().isoformat(),
            "location": str(peak["location"]),
            "cells_l": float(peak["k_cristata_cells_l"]),
            "latitude": float(peak["latitude"]),
            "longitude": float(peak["longitude"]),
        },
        "k_cristata_detected_samples": int(selected["k_cristata_detected"].sum()),
        "k_cristata_detection_share": float(selected["k_cristata_detected"].mean()),
        "interpretation": (
            "Descriptive replay of non-uniform event sampling. Detection shares and "
            "abundance bands are not prevalence estimates or regulatory thresholds."
        ),
    }
    return {
        "observations": selected,
        "timeline": timeline,
        "sites": site_summary,
        "species": species,
        "card": card,
    }


def real_data_router(observations: pd.DataFrame, has_daily_environment: bool = False) -> pd.DataFrame:
    dates = int(observations["sample_date"].nunique())
    locations = int(observations["location"].nunique())
    rows = [
        {
            "branch": "spatiotemporal_qpcr_replay", "decision": "run",
            "data_support": 1.0,
            "reason": f"{len(observations)}条qPCR观测、{dates}个日期、{locations}个地点支持描述性回放",
        },
        {
            "branch": "species_composition", "decision": "run", "data_support": 1.0,
            "reason": "同一工作表提供7种Karenia的qPCR细胞丰度",
        },
        {
            "branch": "multiscale_environmental_anomaly",
            "decision": "run" if has_daily_environment else "defer",
            "data_support": 0.85 if has_daily_environment else 0.25,
            "reason": "需要连续日尺度SST/营养盐历史；当前包保留OISST适配器" if not has_daily_environment else "连续环境历史可用",
        },
        {
            "branch": "te_cte_network", "decision": "defer", "data_support": 0.20,
            "reason": "22个非均匀采样日期和单次事件不足以支持稳健方向性网络检验",
        },
        {
            "branch": "spatial_durbin", "decision": "defer", "data_support": 0.20,
            "reason": "缺少规则空间面板及可独立验证的邻接权重矩阵",
        },
        {
            "branch": "supervised_hab_classifier", "decision": "defer", "data_support": 0.10,
            "reason": "单次事件且采样努力不均；Not detected不能解释为完整生态负样本",
        },
    ]
    return pd.DataFrame(rows)


def project_real_aquaculture_priority(
    sites: pd.DataFrame,
    production_type: str,
    exposure_multiplier: float,
) -> pd.DataFrame:
    """Translate observed relative abundance into a verification priority, not loss risk."""
    profile = PRODUCTION_PROFILES[production_type]
    vulnerability = float(profile["toxigenic"])
    global_peak = max(1.0, float(sites["k_cristata_peak_cells_l"].max()))
    output = sites.copy()
    output["observed_hazard_index"] = 100 * np.log1p(
        output["k_cristata_peak_cells_l"]
    ) / np.log1p(global_peak)
    output["farm_exposure_scenario"] = float(np.clip(exposure_multiplier, 0.05, 1.0))
    output["vulnerability_coefficient"] = vulnerability
    output["verification_priority_index"] = (
        output["observed_hazard_index"]
        * output["farm_exposure_scenario"]
        * output["vulnerability_coefficient"]
    )
    output["evidence_grade"] = "B：现场qPCR藻量；毒素机制为事件级外部证据"
    output["recommended_action"] = str(profile["monitoring"])
    return output.sort_values(
        "verification_priority_index", ascending=False, ignore_index=True
    )

