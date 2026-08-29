"""Inspectable cage-fish biological response sandbox.

This module implements a dimensionless competition-equivalent process model.
It is designed to expose assumptions and intervention trade-offs, not to
predict mortality, biomass loss, toxin concentration, harvest closure, or to
issue automatic operating instructions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


INTERVENTIONS = {
    "维持监测": {
        "feed_multiplier": 1.00,
        "oxygen_gain_mg_l": 0.00,
        "readiness_hours": 18,
        "description": "不改变投喂或供氧，仅持续监测并保留基准轨迹",
    },
    "降低投喂40%": {
        "feed_multiplier": 0.60,
        "oxygen_gain_mg_l": 0.00,
        "readiness_hours": 12,
        "description": "降低代谢与耗氧代理负荷，同时牺牲短期摄食机会",
    },
    "启动增氧": {
        "feed_multiplier": 1.00,
        "oxygen_gain_mg_l": 1.50,
        "readiness_hours": 10,
        "description": "演示性增加有效溶解氧，不代表具体设备的现场增氧能力",
    },
    "转移准备（未执行）": {
        "feed_multiplier": 1.00,
        "oxygen_gain_mg_l": 0.00,
        "readiness_hours": 4,
        "description": "只缩短响应准备时间；未执行转移，因此不降低藻华暴露或生理压力",
    },
    "降低投喂+增氧": {
        "feed_multiplier": 0.60,
        "oxygen_gain_mg_l": 1.50,
        "readiness_hours": 8,
        "description": "联合演示降低耗氧代理负荷与提高有效溶解氧的权衡",
    },
}


BIO_SCENARIO_PRESETS = {
    "复合高压科研情景": {
        "hab_pressure": 80.0,
        "mhw_intensity_c": 2.8,
        "dissolved_oxygen_mg_l": 4.5,
        "stocking_density_kg_m3": 25.0,
        "planned_feeding_pct": 100.0,
        "hab_duration_hours": 48,
        "source_note": "全部为可调整的科研演示输入",
    },
    "藻华主导科研情景": {
        "hab_pressure": 85.0,
        "mhw_intensity_c": 0.8,
        "dissolved_oxygen_mg_l": 6.5,
        "stocking_density_kg_m3": 15.0,
        "planned_feeding_pct": 100.0,
        "hab_duration_hours": 36,
        "source_note": "全部为可调整的科研演示输入",
    },
    "低氧高密度科研情景": {
        "hab_pressure": 45.0,
        "mhw_intensity_c": 1.2,
        "dissolved_oxygen_mg_l": 3.8,
        "stocking_density_kg_m3": 30.0,
        "planned_feeding_pct": 100.0,
        "hab_duration_hours": 48,
        "source_note": "全部为可调整的科研演示输入",
    },
    "低压力对照": {
        "hab_pressure": 20.0,
        "mhw_intensity_c": 0.4,
        "dissolved_oxygen_mg_l": 7.0,
        "stocking_density_kg_m3": 10.0,
        "planned_feeding_pct": 90.0,
        "hab_duration_hours": 24,
        "source_note": "全部为可调整的科研演示输入",
    },
    "南澳真实qPCR峰值锚点": {
        "hab_pressure": 100.0,
        "mhw_intensity_c": 2.0,
        "dissolved_oxygen_mg_l": 5.5,
        "stocking_density_kg_m3": 18.0,
        "planned_feeding_pct": 100.0,
        "hab_duration_hours": 48,
        "source_note": "仅藻华压力锚定真实回放内相对峰值；其余输入为演示假设",
    },
    "挪威真实监测峰值锚点": {
        "hab_pressure": 100.0,
        "mhw_intensity_c": 1.0,
        "dissolved_oxygen_mg_l": 6.0,
        "stocking_density_kg_m3": 18.0,
        "planned_feeding_pct": 100.0,
        "hab_duration_hours": 36,
        "source_note": "仅藻华压力锚定真实回放内相对峰值；其余输入为演示假设",
    },
}


MODEL_PARAMETERS = {
    "HAB主效应权重": (0.34, "无量纲", "藻华危害压力对相对生理压力的贡献"),
    "热异常主效应权重": (0.18, "无量纲", "MHW强度经平滑变换后的贡献"),
    "低氧主效应权重": (0.23, "无量纲", "有效DO不足经平滑变换后的贡献"),
    "养殖密度主效应权重": (0.10, "无量纲", "密度负荷经平滑变换后的贡献"),
    "HAB×热异常权重": (0.10, "无量纲", "复合压力交互项"),
    "低氧×密度权重": (0.08, "无量纲", "复合压力交互项"),
    "摄食×低氧权重": (0.07, "无量纲", "摄食代谢负荷代理交互项"),
    "热异常平滑中心": (1.50, "°C", "科研原型的平滑响应中心；非通用阈值"),
    "低氧平滑中心": (4.80, "mg L⁻¹", "科研原型的平滑响应中心；非监管阈值"),
    "密度平滑中心": (18.0, "kg m⁻³", "科研原型的平滑响应中心；非养殖标准"),
    "增氧演示增益": (1.50, "mg L⁻¹", "情景假设；不代表具体设备能力"),
    "藻华退出半衰期": (12.0, "h", "事件持续期后的压力衰减假设"),
    "参数不确定性缩放": (0.15, "±比例", "对综合挑战强度作±15%敏感性包络"),
}


def _logistic(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _pressure_band(value: float) -> str:
    if value >= 75:
        return "很高：需要优先现场复核"
    if value >= 50:
        return "较高：建议加密监测"
    if value >= 25:
        return "中等：持续跟踪变化"
    return "较低：维持常规监测"


def parameter_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "参数": name,
            "演示值": value,
            "单位": unit,
            "解释边界": note,
            "参数属性": "公开可检查的科研原型设定",
        }
        for name, (value, unit, note) in MODEL_PARAMETERS.items()
    ])


def _validate_inputs(
    hab_pressure: float,
    mhw_intensity_c: float,
    dissolved_oxygen_mg_l: float,
    stocking_density_kg_m3: float,
    planned_feeding_pct: float,
    hab_duration_hours: int,
    horizon_hours: int,
) -> None:
    checks = {
        "hab_pressure": (hab_pressure, 0.0, 100.0),
        "mhw_intensity_c": (mhw_intensity_c, 0.0, 6.0),
        "dissolved_oxygen_mg_l": (dissolved_oxygen_mg_l, 0.0, 14.0),
        "stocking_density_kg_m3": (stocking_density_kg_m3, 1.0, 60.0),
        "planned_feeding_pct": (planned_feeding_pct, 0.0, 120.0),
        "hab_duration_hours": (hab_duration_hours, 1, horizon_hours),
        "horizon_hours": (horizon_hours, 24, 168),
    }
    for name, (value, lower, upper) in checks.items():
        if not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")


def simulate_cage_fish_response(
    hab_pressure: float,
    mhw_intensity_c: float,
    dissolved_oxygen_mg_l: float,
    stocking_density_kg_m3: float,
    planned_feeding_pct: float,
    hab_duration_hours: int,
    intervention: str,
    horizon_hours: int = 72,
    challenge_scale: float = 1.0,
) -> pd.DataFrame:
    """Simulate a relative cage-fish pressure trajectory at hourly resolution."""
    _validate_inputs(
        hab_pressure, mhw_intensity_c, dissolved_oxygen_mg_l,
        stocking_density_kg_m3, planned_feeding_pct,
        hab_duration_hours, horizon_hours,
    )
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {intervention}")
    action = INTERVENTIONS[intervention]
    heat_pressure = _logistic((mhw_intensity_c - 1.50) / 0.55)
    density_pressure = _logistic((stocking_density_kg_m3 - 18.0) / 5.0)
    planned_feed = planned_feeding_pct / 100.0
    stress = 8.0
    rows: list[dict[str, float | int | str]] = []

    for hour in range(horizon_hours + 1):
        if hour <= hab_duration_hours:
            hab_profile = hab_pressure / 100.0
        else:
            elapsed = hour - hab_duration_hours
            hab_profile = (hab_pressure / 100.0) * 0.5 ** (elapsed / 12.0)

        appetite_pre = np.clip(
            1.0 - 0.42 * stress / 100.0 - 0.22 * hab_profile,
            0.15,
            1.0,
        )
        realised_feed = float(
            planned_feed * float(action["feed_multiplier"]) * appetite_pre
        )
        density_drawdown = 0.65 * density_pressure
        feeding_drawdown = 0.30 * realised_feed * density_pressure
        effective_do = float(np.clip(
            dissolved_oxygen_mg_l
            + float(action["oxygen_gain_mg_l"])
            - density_drawdown
            - feeding_drawdown,
            0.0,
            14.0,
        ))
        oxygen_pressure = _logistic((4.80 - effective_do) / 0.65)
        challenge = (
            0.34 * hab_profile
            + 0.18 * heat_pressure
            + 0.23 * oxygen_pressure
            + 0.10 * density_pressure
            + 0.10 * hab_profile * heat_pressure
            + 0.08 * oxygen_pressure * density_pressure
            + 0.07 * realised_feed * oxygen_pressure
        )
        challenge = float(np.clip(challenge * challenge_scale, 0.0, 1.0))
        appetite = float(np.clip(
            1.0
            - 0.42 * stress / 100.0
            - 0.22 * hab_profile
            - 0.22 * oxygen_pressure,
            0.15,
            1.0,
        ))
        realised_feed = float(
            planned_feed * float(action["feed_multiplier"]) * appetite
        )
        feeding_opportunity = float(
            100.0 * realised_feed / max(planned_feed, 1e-9)
        ) if planned_feed > 0 else 100.0
        rows.append({
            "hour": hour,
            "intervention": intervention,
            "relative_physiological_pressure": round(stress, 4),
            "compound_challenge": round(100.0 * challenge, 4),
            "hab_pressure_active": round(100.0 * hab_profile, 4),
            "heat_pressure_component": round(100.0 * heat_pressure, 4),
            "oxygen_pressure_component": round(100.0 * oxygen_pressure, 4),
            "density_pressure_component": round(100.0 * density_pressure, 4),
            "effective_do_mg_l": round(effective_do, 4),
            "appetite_proxy": round(100.0 * appetite, 4),
            "feeding_opportunity_pct": round(feeding_opportunity, 4),
            "response_readiness_hours": int(action["readiness_hours"]),
        })
        accumulation = 1.45 * challenge * (1.0 - stress / 100.0)
        recovery = 0.55 * (1.0 - challenge) * (stress / 100.0)
        stress = float(np.clip(stress + accumulation - recovery, 0.0, 100.0))
    return pd.DataFrame(rows)


def _summarise(trajectory: pd.DataFrame) -> dict[str, object]:
    pressure = trajectory["relative_physiological_pressure"].to_numpy(float)
    feed = trajectory["feeding_opportunity_pct"].to_numpy(float)
    peak = float(pressure.max())
    return {
        "intervention": str(trajectory["intervention"].iloc[0]),
        "peak_pressure_index": peak,
        "mean_pressure_index": float(pressure.mean()),
        "ending_pressure_index": float(pressure[-1]),
        "pressure_load_index_hours": float(np.trapezoid(pressure, dx=1.0)),
        "minimum_effective_do_mg_l": float(trajectory["effective_do_mg_l"].min()),
        "mean_feeding_opportunity_pct": float(feed.mean()),
        "response_readiness_hours": int(trajectory["response_readiness_hours"].iloc[0]),
        "pressure_band": _pressure_band(peak),
        "scenario_interpretation": str(
            INTERVENTIONS[str(trajectory["intervention"].iloc[0])]["description"]
        ),
    }


def compare_interventions(
    hab_pressure: float,
    mhw_intensity_c: float,
    dissolved_oxygen_mg_l: float,
    stocking_density_kg_m3: float,
    planned_feeding_pct: float,
    hab_duration_hours: int,
    horizon_hours: int = 72,
) -> dict[str, object]:
    """Compare all interventions and return trajectories, summaries and audit data."""
    trajectories = []
    summaries = []
    for intervention in INTERVENTIONS:
        central = simulate_cage_fish_response(
            hab_pressure, mhw_intensity_c, dissolved_oxygen_mg_l,
            stocking_density_kg_m3, planned_feeding_pct,
            hab_duration_hours, intervention, horizon_hours, 1.0,
        )
        lower = simulate_cage_fish_response(
            hab_pressure, mhw_intensity_c, dissolved_oxygen_mg_l,
            stocking_density_kg_m3, planned_feeding_pct,
            hab_duration_hours, intervention, horizon_hours, 0.85,
        )
        upper = simulate_cage_fish_response(
            hab_pressure, mhw_intensity_c, dissolved_oxygen_mg_l,
            stocking_density_kg_m3, planned_feeding_pct,
            hab_duration_hours, intervention, horizon_hours, 1.15,
        )
        trajectories.append(central)
        summary = _summarise(central)
        summary["peak_pressure_lower"] = float(
            lower["relative_physiological_pressure"].max()
        )
        summary["peak_pressure_upper"] = float(
            upper["relative_physiological_pressure"].max()
        )
        summaries.append(summary)

    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    baseline_load = float(
        summary_frame.loc[
            summary_frame["intervention"].eq("维持监测"),
            "pressure_load_index_hours",
        ].iloc[0]
    )
    summary_frame["pressure_load_reduction_vs_baseline_pct"] = (
        100.0
        * (baseline_load - summary_frame["pressure_load_index_hours"])
        / max(baseline_load, 1e-9)
    )
    summary_frame = summary_frame.sort_values(
        ["pressure_load_index_hours", "mean_feeding_opportunity_pct"],
        ascending=[True, False],
        ignore_index=True,
    )
    lowest_pressure = str(summary_frame.iloc[0]["intervention"])
    scenario_card = {
        "model_class": "dimensionless_competition_equivalent_biological_response_sandbox",
        "demonstration_species": "generic marine cage fish",
        "horizon_hours": int(horizon_hours),
        "inputs": {
            "hab_pressure": float(hab_pressure),
            "mhw_intensity_c": float(mhw_intensity_c),
            "dissolved_oxygen_mg_l": float(dissolved_oxygen_mg_l),
            "stocking_density_kg_m3": float(stocking_density_kg_m3),
            "planned_feeding_pct": float(planned_feeding_pct),
            "hab_duration_hours": int(hab_duration_hours),
        },
        "lowest_pressure_scenario": lowest_pressure,
        "interpretation": (
            "The lowest-pressure scenario is a sandbox comparison, not an automatic "
            "recommendation. Transfer preparation has no physiological benefit until "
            "a transfer is actually executed."
        ),
        "excluded_claims": [
            "mortality prediction", "biomass loss prediction", "toxin prediction",
            "farm-specific forecast", "automatic feeding/aeration/transfer instruction",
        ],
    }
    return {
        "trajectories": trajectory_frame,
        "summary": summary_frame,
        "parameters": parameter_table(),
        "scenario_card": scenario_card,
    }

