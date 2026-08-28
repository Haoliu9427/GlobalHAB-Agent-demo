"""Additional real observations and global Nature Portfolio evidence cases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


GLOBAL_EVIDENCE_CASES = (
    {
        "case": "南澳大利亚复杂Karenia事件",
        "region": "澳大利亚 · Gulf St Vincent",
        "latitude": -34.7,
        "longitude": 138.0,
        "period": "2025",
        "journal": "Nature Ecology & Evolution",
        "evidence": "qPCR物种丰度与事件影响",
        "product_status": "完整观测回放",
        "records": "115条样本",
        "url": "https://doi.org/10.1038/s41559-026-03115-0",
        "data_url": "https://doi.org/10.5281/zenodo.20227730",
    },
    {
        "case": "挪威沿岸有毒藻监测",
        "region": "挪威 · 58–71°N沿岸",
        "latitude": 64.0,
        "longitude": 10.5,
        "period": "2006–2019",
        "journal": "Communications Earth & Environment",
        "evidence": "周尺度藻细胞计数与海洋环境",
        "product_status": "完整观测回放",
        "records": "5,919条观测",
        "url": "https://doi.org/10.1038/s43247-025-02421-y",
        "data_url": "https://doi.org/10.5281/zenodo.10958487",
    },
    {
        "case": "Orcas Island藻华前置信号",
        "region": "美国 · Salish Sea",
        "latitude": 48.6,
        "longitude": -122.9,
        "period": "高频时间序列",
        "journal": "Nature Communications",
        "evidence": "海洋微生物组与蛋白组前置信号",
        "product_status": "研究证据接口",
        "records": "开放组学数据与代码",
        "url": "https://doi.org/10.1038/s41467-025-59250-y",
        "data_url": "https://doi.org/10.6084/m9.figshare.27166347",
    },
    {
        "case": "全球HAB事件与监测强度",
        "region": "全球海岸带",
        "latitude": 8.0,
        "longitude": -20.0,
        "period": "1985–2018",
        "journal": "Communications Earth & Environment",
        "evidence": "HAEDAT、OBIS与养殖监测强度",
        "product_status": "全球背景证据",
        "records": "多区域事件数据库",
        "url": "https://doi.org/10.1038/s43247-021-00178-8",
        "data_url": "https://github.com/iobis/paper-hab-trends",
    },
)


def global_evidence_frame() -> pd.DataFrame:
    return pd.DataFrame(GLOBAL_EVIDENCE_CASES)


def load_norway_real_case(data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    derived = data_root / "real_case_norway" / "derived"
    observations = pd.read_csv(
        derived / "norway_hab_observations.csv", parse_dates=["sample_date"]
    )
    provenance = json.loads((derived / "provenance.json").read_text(encoding="utf-8"))
    return observations, provenance


def build_norway_replay(
    observations: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    region: str = "全部站点",
) -> dict[str, object]:
    selected = observations[
        observations["sample_date"].between(
            pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both"
        )
    ].copy()
    if region != "全部站点":
        selected = selected[selected["region"].eq(region)].copy()
    if selected.empty:
        raise ValueError("selected Norwegian replay window has no observations")

    timeline = selected.groupby("sample_date", as_index=False).agg(
        samples=("region", "size"),
        regions=("region", "nunique"),
        target_hab_events=("target_hab_event", "sum"),
        target_hab_share=("target_hab_event", "mean"),
        a_tamarense_peak_cells_l=("a_tamarense_cells_l", "max"),
        d_acuta_peak_cells_l=("d_acuta_cells_l", "max"),
        mean_sst_c=("sst_c", "mean"),
        mean_salinity_psu=("sea_surface_salinity_psu", "mean"),
    )
    station_summary = selected.groupby("region", as_index=False).agg(
        observations=("sample_date", "size"),
        sampling_dates=("sample_date", "nunique"),
        first_sample=("sample_date", "min"),
        last_sample=("sample_date", "max"),
        event_observations=("target_hab_event", "sum"),
        event_share=("target_hab_event", "mean"),
        a_tamarense_peak_cells_l=("a_tamarense_cells_l", "max"),
        d_acuta_peak_cells_l=("d_acuta_cells_l", "max"),
        median_sst_c=("sst_c", "median"),
        median_salinity_psu=("sea_surface_salinity_psu", "median"),
    ).sort_values(
        ["event_observations", "event_share"], ascending=False, ignore_index=True
    )

    taxa = pd.DataFrame([
        {
            "taxon": "A. tamarense complex",
            "positive_observations": int(selected["a_tamarense_cells_l"].gt(0).sum()),
            "event_observations": int(selected["a_tamarense_hab_event"].sum()),
            "peak_cells_l": float(selected["a_tamarense_cells_l"].max()),
        },
        {
            "taxon": "D. acuta",
            "positive_observations": int(selected["d_acuta_cells_l"].gt(0).sum()),
            "event_observations": int(selected["d_acuta_hab_event"].sum()),
            "peak_cells_l": float(selected["d_acuta_cells_l"].max()),
        },
    ])
    peak_a = selected.loc[selected["a_tamarense_cells_l"].idxmax()]
    peak_d = selected.loc[selected["d_acuta_cells_l"].idxmax()]
    card = {
        "mode": "real_monitoring_replay_not_new_model_training",
        "event": "Norwegian coast toxic algae monitoring, 2006-2019",
        "observations": int(len(selected)),
        "sampling_dates": int(selected["sample_date"].nunique()),
        "regions": int(selected["region"].nunique()),
        "date_range": [
            selected["sample_date"].min().date().isoformat(),
            selected["sample_date"].max().date().isoformat(),
        ],
        "target_event_observations": int(selected["target_hab_event"].sum()),
        "study_threshold_cells_l": 200.0,
        "peak_a_tamarense": {
            "date": peak_a["sample_date"].date().isoformat(),
            "region": str(peak_a["region"]),
            "cells_l": float(peak_a["a_tamarense_cells_l"]),
        },
        "peak_d_acuta": {
            "date": peak_d["sample_date"].date().isoformat(),
            "region": str(peak_d["region"]),
            "cells_l": float(peak_d["d_acuta_cells_l"]),
        },
        "interpretation": (
            "Replay of public weekly monitoring observations and study-defined event "
            "flags. Percentages describe this monitoring table, not population prevalence."
        ),
    }
    return {
        "observations": selected,
        "timeline": timeline,
        "stations": station_summary,
        "taxa": taxa,
        "card": card,
    }
