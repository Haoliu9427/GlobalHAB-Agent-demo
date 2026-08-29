"""Translate real HAB observations into auditable monitoring priorities.

The functions in this module do not estimate mortality, economic loss, toxin
concentration, harvest closure, or an operational forecast.  They keep observed
hazard evidence separate from scenario exposure, parameterised vulnerability,
and missing decision inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aquaculture import PRODUCTION_PROFILES
from .real_replay import project_real_aquaculture_priority


def _priority_level(score: float) -> str:
    if score >= 70:
        return "一级：立即现场复核"
    if score >= 50:
        return "二级：加密采样复核"
    if score >= 30:
        return "三级：持续趋势核查"
    return "四级：维持常规监测"


def _action(level: str, production_type: str) -> str:
    if level.startswith("一级"):
        prefix = "立即组织现场复核；"
    elif level.startswith("二级"):
        prefix = "在24小时内增加采样频次；"
    elif level.startswith("三级"):
        prefix = "维持每日趋势核查；"
    else:
        prefix = "维持常规监测；"
    return prefix + str(PRODUCTION_PROFILES[production_type]["monitoring"])


def _evidence_matrix(
    event_label: str,
    abundance_detail: str,
    location_detail: str,
    production_type: str,
    exposure_multiplier: float,
    mechanism_detail: str,
    physiology_detail: str,
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "研判输入": "藻种与丰度",
            "当前证据": abundance_detail,
            "证据属性": "真实观测",
            "如何影响研判": "形成危害证据，不直接等同于养殖损失",
        },
        {
            "研判输入": "时间与位置",
            "当前证据": location_detail,
            "证据属性": "真实观测",
            "如何影响研判": "确定优先复核的时间和监测区域",
        },
        {
            "研判输入": "危害机制",
            "当前证据": mechanism_detail,
            "证据属性": "事件/文献证据",
            "如何影响研判": "决定优先检测毒素、鳃部损伤或缺氧等指标",
        },
        {
            "研判输入": "养殖暴露",
            "当前证据": f"演示情景系数 {exposure_multiplier:.2f}；未接入真实养殖场空间图层",
            "证据属性": "情景假设",
            "如何影响研判": "仅用于演示同一危害下暴露差异如何改变复核顺序",
        },
        {
            "研判输入": "对象脆弱性",
            "当前证据": f"选择对象：{production_type}",
            "证据属性": "参数设定",
            "如何影响研判": "区分贝类、网箱鱼、甲壳类和海藻的响应重点",
        },
        {
            "研判输入": "现场生物/监管证据",
            "当前证据": physiology_detail,
            "证据属性": "待补数据",
            "如何影响研判": "补齐后才可支持停采、转移或闭区等业务决策",
        },
    ]).assign(事件=event_label)


def build_sa_risk_translation(
    sites: pd.DataFrame,
    production_type: str,
    exposure_multiplier: float,
) -> dict[str, object]:
    """Connect South Australia qPCR evidence to a verification priority."""
    priority = project_real_aquaculture_priority(
        sites, production_type, exposure_multiplier
    ).copy()
    priority["priority_level"] = priority["verification_priority_index"].map(
        _priority_level
    )
    priority["recommended_action"] = priority["priority_level"].map(
        lambda level: _action(level, production_type)
    )
    priority["hazard_evidence_source"] = "真实现场qPCR观测"
    priority["exposure_evidence_source"] = "用户设定情景；非真实养殖场暴露"
    priority["decision_boundary"] = "现场复核优先级；不是损失概率或监管阈值"
    top = priority.iloc[0]
    evidence = _evidence_matrix(
        event_label="南澳大利亚2025复杂Karenia事件",
        abundance_detail=(
            f"K. cristata现场qPCR峰值 {top['k_cristata_peak_cells_l']:,.0f} cells L⁻¹"
        ),
        location_detail=(
            f"最高观测地点 {top['location']}；来自所选回放时间窗"
        ),
        production_type=production_type,
        exposure_multiplier=exposure_multiplier,
        mechanism_detail="复杂Karenia事件的有害/产毒相关证据；等级B",
        physiology_detail="缺少逐场养殖暴露、同步毒素、动物生理反应及当地监管阈值",
    )
    summary = {
        "event": "南澳大利亚2025复杂Karenia事件",
        "observation": (
            f"K. cristata峰值 {top['k_cristata_peak_cells_l']:,.0f} cells L⁻¹"
        ),
        "hazard": "复杂Karenia有害/产毒相关证据 · B级",
        "exposure": f"情景系数 {exposure_multiplier:.2f} · 未接入真实养殖场",
        "vulnerability": (
            f"{production_type} · 系数 {float(top['vulnerability_coefficient']):.2f}"
        ),
        "priority": f"{top['priority_level']} · {top['verification_priority_index']:.1f}/100",
        "action": str(top["recommended_action"]),
        "top_location": str(top["location"]),
        "boundary": "输出为现场复核优先级，不是损失概率、毒素浓度或停采指令",
    }
    return {"priority": priority, "evidence": evidence, "summary": summary}


def build_norway_risk_translation(
    stations: pd.DataFrame,
    production_type: str,
    exposure_multiplier: float,
) -> dict[str, object]:
    """Connect Norwegian monitoring counts to a monitoring priority.

    The hazard index is a within-window relative abundance index.  It compares
    log-transformed station peaks with the largest observed peak in the selected
    replay window and therefore is not a regulatory or toxicological threshold.
    """
    if stations.empty:
        raise ValueError("stations must contain at least one monitoring region")
    if production_type not in PRODUCTION_PROFILES:
        raise ValueError(f"unknown production type: {production_type}")

    output = stations.copy()
    output["target_peak_cells_l"] = output[[
        "a_tamarense_peak_cells_l", "d_acuta_peak_cells_l"
    ]].max(axis=1)
    global_peak = max(1.0, float(output["target_peak_cells_l"].max()))
    output["observed_hazard_index"] = 100 * np.log1p(
        output["target_peak_cells_l"]
    ) / np.log1p(global_peak)
    exposure = float(np.clip(exposure_multiplier, 0.05, 1.0))
    vulnerability = float(PRODUCTION_PROFILES[production_type]["toxigenic"])
    output["farm_exposure_scenario"] = exposure
    output["vulnerability_coefficient"] = vulnerability
    output["verification_priority_index"] = (
        output["observed_hazard_index"] * exposure * vulnerability
    )
    output["priority_level"] = output["verification_priority_index"].map(
        _priority_level
    )
    output["evidence_grade"] = "B：真实藻细胞计数与论文定义事件；非地方监管阈值"
    output["recommended_action"] = output["priority_level"].map(
        lambda level: _action(level, production_type)
    )
    output["hazard_evidence_source"] = "真实沿岸监测计数"
    output["exposure_evidence_source"] = "用户设定情景；非真实养殖场暴露"
    output["decision_boundary"] = "加密监测优先级；不是闭区、停采或损失预测"
    output = output.sort_values(
        "verification_priority_index", ascending=False, ignore_index=True
    )
    top = output.iloc[0]
    evidence = _evidence_matrix(
        event_label="挪威沿岸2006–2019有毒藻监测",
        abundance_detail=(
            f"A. tamarense complex / D. acuta最高计数 {top['target_peak_cells_l']:,.0f} cells L⁻¹"
        ),
        location_detail=f"最高相对危害监测区域 {top['region']}；来自所选回放时间窗",
        production_type=production_type,
        exposure_multiplier=exposure_multiplier,
        mechanism_detail="有毒藻监测与论文定义事件证据；等级B",
        physiology_detail="缺少逐场养殖暴露、同步贝类毒素/鱼类反应及地方管控阈值",
    )
    summary = {
        "event": "挪威沿岸2006–2019有毒藻监测",
        "observation": f"目标藻最高计数 {top['target_peak_cells_l']:,.0f} cells L⁻¹",
        "hazard": "有毒藻监测事件证据 · B级",
        "exposure": f"情景系数 {exposure_multiplier:.2f} · 未接入真实养殖场",
        "vulnerability": (
            f"{production_type} · 系数 {float(top['vulnerability_coefficient']):.2f}"
        ),
        "priority": f"{top['priority_level']} · {top['verification_priority_index']:.1f}/100",
        "action": str(top["recommended_action"]),
        "top_location": str(top["region"]),
        "boundary": "输出为加密监测优先级，不是损失概率、毒素浓度或监管指令",
    }
    return {"priority": output, "evidence": evidence, "summary": summary}

