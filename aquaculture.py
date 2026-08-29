"""Transparent HAB-to-aquaculture impact prioritisation for the demo.

This is a decision-support layer, not a mortality, toxin or harvest-closure
model. It keeps bloom hazard, farm exposure, stock vulnerability and evidence
confidence separate so users can inspect every assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PRODUCTION_PROFILES = {
    "贝类（牡蛎/贻贝）": {
        "toxigenic": 1.00,
        "ichthyotoxic": 0.72,
        "hypoxia": 0.80,
        "unknown": 0.70,
        "monitoring": "加密藻种、毒素和贝类组织采样；是否停采必须依据当地监管阈值",
    },
    "海水网箱鱼": {
        "toxigenic": 0.72,
        "ichthyotoxic": 1.00,
        "hypoxia": 0.95,
        "unknown": 0.78,
        "monitoring": "加密鳃部症状、溶解氧和鱼群行为监测，准备增氧或转移预案",
    },
    "海水甲壳类": {
        "toxigenic": 0.82,
        "ichthyotoxic": 0.78,
        "hypoxia": 0.88,
        "unknown": 0.72,
        "monitoring": "加密幼体/成体状态、溶解氧和毒素证据监测，核查生命阶段敏感性",
    },
    "海藻养殖": {
        "toxigenic": 0.52,
        "ichthyotoxic": 0.45,
        "hypoxia": 0.58,
        "unknown": 0.50,
        "monitoring": "加密水质、附着污染和产品质量监测；避免从动物毒性直接外推",
    },
}

MECHANISM_LABELS = {
    "unknown": "未知/仅候选水华",
    "toxigenic": "产毒/食品安全相关",
    "ichthyotoxic": "鱼毒性或鳃损伤相关",
    "hypoxia": "高生物量/缺氧相关",
}

EVIDENCE_CONFIDENCE = {
    "A：物种/毒素/危害确认": 0.90,
    "B：现场藻量/闭港等事件证据": 0.70,
    "C：仅卫星或环境候选": 0.45,
}

ZONE_EXPOSURE = {
    "北阿拉斯加湾": 0.62,
    "加州沿岸": 0.78,
    "墨西哥湾": 0.90,
    "北大西洋中部": 0.48,
    "日本黑潮延伸区": 0.76,
    "中国南部近岸": 0.92,
    "南澳大利亚近岸": 0.82,
}


def _priority_level(score: float) -> str:
    if score >= 70:
        return "一级：立即核查"
    if score >= 50:
        return "二级：加密监测"
    if score >= 30:
        return "三级：持续关注"
    return "四级：常规监测"


def _next_action(level: str, profile: dict[str, object]) -> str:
    if level.startswith("一级"):
        prefix = "立即组织现场复核；"
    elif level.startswith("二级"):
        prefix = "在24小时内增加采样频次；"
    elif level.startswith("三级"):
        prefix = "维持每日趋势核查；"
    else:
        prefix = "维持常规监测；"
    return prefix + str(profile["monitoring"])


def project_aquaculture_risk(
    bloom_scenario: pd.DataFrame,
    production_type: str,
    mechanism: str,
    farm_exposure_multiplier: float,
    evidence_grade: str,
) -> pd.DataFrame:
    """Translate bloom hazard into an auditable response-priority index."""
    if production_type not in PRODUCTION_PROFILES:
        raise ValueError(f"unknown production type: {production_type}")
    if mechanism not in MECHANISM_LABELS:
        raise ValueError(f"unknown mechanism: {mechanism}")
    if evidence_grade not in EVIDENCE_CONFIDENCE:
        raise ValueError(f"unknown evidence grade: {evidence_grade}")

    profile = PRODUCTION_PROFILES[production_type]
    vulnerability = float(profile[mechanism])
    confidence = EVIDENCE_CONFIDENCE[evidence_grade]
    rows = []
    for record in bloom_scenario.to_dict("records"):
        exposure = float(
            np.clip(
                ZONE_EXPOSURE.get(str(record["候选海区"]), 0.60)
                * farm_exposure_multiplier,
                0.05,
                1.0,
            )
        )
        hazard = float(record["综合风险指数"]) / 100.0
        priority = 100.0 * hazard * exposure * vulnerability
        uncertainty = 8.0 + 20.0 * (1.0 - confidence)
        lower = max(0.0, priority - uncertainty)
        upper = min(100.0, priority + uncertainty)
        level = _priority_level(priority)
        rows.append({
            "候选海区": record["候选海区"],
            "latitude": record["latitude"],
            "longitude": record["longitude"],
            "预计窗口": record["预计窗口"],
            "养殖对象": production_type,
            "危害机制": MECHANISM_LABELS[mechanism],
            "藻华危害指数": round(float(record["综合风险指数"]), 1),
            "养殖暴露度": round(exposure, 2),
            "脆弱性系数": round(vulnerability, 2),
            "证据置信度": round(confidence, 2),
            "养殖响应优先指数": round(priority, 1),
            "不确定性下限": round(lower, 1),
            "不确定性上限": round(upper, 1),
            "响应级别": level,
            "建议动作": _next_action(level, profile),
        })
    return pd.DataFrame(rows).sort_values(
        "养殖响应优先指数", ascending=False, ignore_index=True
    )
