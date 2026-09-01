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


# Real place names identify representative production settings. Environmental
# values remain adjustable scenario defaults, not live observations or local
# operating limits. Capture-only fisheries are mapped for global context but
# are deliberately excluded from the cage-fish response selector.
BIO_PRODUCTION_REGIONS = {
    "挪威海—峡湾沿岸": {
        "latitude": 65.0, "longitude": 10.0,
        "production_type": "海水网箱养殖", "representative_stock": "大西洋鲑",
        "cage_sandbox": True, "hab_pressure": 55.0, "mhw_intensity_c": 1.2,
        "dissolved_oxygen_mg_l": 7.0, "stocking_density_kg_m3": 18.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 36,
    },
    "北阿拉斯加湾": {
        "latitude": 57.0, "longitude": -145.0,
        "production_type": "捕捞渔业背景", "representative_stock": "鲑类、鳕类与北太平洋近岸渔业",
        "cage_sandbox": False,
    },
    "北大西洋中部": {
        "latitude": 36.0, "longitude": -40.0,
        "production_type": "捕捞渔业背景", "representative_stock": "远洋金枪鱼及大西洋远洋渔业",
        "cage_sandbox": False,
    },
    "东地中海—爱琴海": {
        "latitude": 37.2, "longitude": 25.2,
        "production_type": "海水网箱养殖", "representative_stock": "欧洲海鲈、金头鲷",
        "cage_sandbox": True, "hab_pressure": 60.0, "mhw_intensity_c": 2.3,
        "dissolved_oxygen_mg_l": 5.8, "stocking_density_kg_m3": 20.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 42,
    },
    "智利巴塔哥尼亚峡湾": {
        "latitude": -43.5, "longitude": -73.5,
        "production_type": "海水网箱养殖", "representative_stock": "鲑鳟类",
        "cage_sandbox": True, "hab_pressure": 65.0, "mhw_intensity_c": 1.4,
        "dissolved_oxygen_mg_l": 6.4, "stocking_density_kg_m3": 20.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 42,
    },
    "日本黑潮—濑户内海": {
        "latitude": 33.6, "longitude": 133.5,
        "production_type": "海水网箱养殖", "representative_stock": "鰤鱼、真鲷",
        "cage_sandbox": True, "hab_pressure": 62.0, "mhw_intensity_c": 2.2,
        "dissolved_oxygen_mg_l": 5.8, "stocking_density_kg_m3": 18.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 36,
    },
    "中国南部近岸": {
        "latitude": 21.8, "longitude": 113.4,
        "production_type": "海水网箱养殖", "representative_stock": "海鲈、石斑鱼",
        "cage_sandbox": True, "hab_pressure": 68.0, "mhw_intensity_c": 2.6,
        "dissolved_oxygen_mg_l": 5.2, "stocking_density_kg_m3": 16.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 42,
    },
    "东印度洋群岛近岸": {
        "latitude": -5.0, "longitude": 116.0,
        "production_type": "海水网箱养殖", "representative_stock": "石斑鱼、军曹鱼",
        "cage_sandbox": True, "hab_pressure": 64.0, "mhw_intensity_c": 2.5,
        "dissolved_oxygen_mg_l": 5.4, "stocking_density_kg_m3": 15.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 36,
    },
    "南澳大利亚近岸": {
        "latitude": -35.2, "longitude": 137.6,
        "production_type": "海水网箱养殖", "representative_stock": "黄尾鰤、金枪鱼养殖背景",
        "cage_sandbox": True, "hab_pressure": 70.0, "mhw_intensity_c": 2.0,
        "dissolved_oxygen_mg_l": 6.0, "stocking_density_kg_m3": 14.0,
        "planned_feeding_pct": 100.0, "hab_duration_hours": 48,
    },
    "秘鲁—智利洪堡流": {
        "latitude": -22.0, "longitude": -73.5,
        "production_type": "捕捞渔业背景", "representative_stock": "鳀鱼、沙丁鱼等小型中上层鱼",
        "cage_sandbox": False,
    },
    "西印度洋—阿拉伯海": {
        "latitude": 9.0, "longitude": 63.0,
        "production_type": "捕捞渔业背景", "representative_stock": "金枪鱼及类金枪鱼",
        "cage_sandbox": False,
    },
    "加州上升流沿岸": {
        "latitude": 35.5, "longitude": -123.0,
        "production_type": "捕捞与贝类养殖背景", "representative_stock": "中上层鱼、蟹类、贝类",
        "cage_sandbox": False,
    },
    "墨西哥湾": {
        "latitude": 25.5, "longitude": -89.5,
        "production_type": "捕捞与贝类养殖背景", "representative_stock": "虾类、鱼类、牡蛎",
        "cage_sandbox": False,
    },
}


def production_region_frame() -> pd.DataFrame:
    """Return representative global production settings for map display."""
    return pd.DataFrame([
        {
            "region": name,
            "latitude": profile["latitude"],
            "longitude": profile["longitude"],
            "production_type": profile["production_type"],
            "representative_stock": profile["representative_stock"],
            "cage_sandbox": bool(profile["cage_sandbox"]),
            "evidence_boundary": (
                "可进入网箱鱼情景沙盘；默认环境值为可调演示输入"
                if profile["cage_sandbox"]
                else "仅作全球渔业空间背景，不代入网箱鱼生理模型"
            ),
        }
        for name, profile in BIO_PRODUCTION_REGIONS.items()
    ])


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


def evaluate_intervention_robustness(
    hab_pressure: float,
    mhw_intensity_c: float,
    dissolved_oxygen_mg_l: float,
    stocking_density_kg_m3: float,
    planned_feeding_pct: float,
    hab_duration_hours: int,
    horizon_hours: int = 72,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Stress-test intervention trade-offs across 81 nearby input scenarios.

    The grid perturbs four uncertain scenario inputs.  It does not turn the
    prototype into a calibrated fish model; it checks whether conclusions are
    artefacts of a single slider setting.
    """
    perturbations = {
        "hab_multiplier": (0.90, 1.00, 1.10),
        "mhw_delta_c": (-0.40, 0.00, 0.40),
        "do_delta_mg_l": (-0.50, 0.00, 0.50),
        "density_multiplier": (0.90, 1.00, 1.10),
    }
    rows: list[dict[str, object]] = []
    scenario_id = 0
    for hab_multiplier in perturbations["hab_multiplier"]:
        for mhw_delta in perturbations["mhw_delta_c"]:
            for do_delta in perturbations["do_delta_mg_l"]:
                for density_multiplier in perturbations["density_multiplier"]:
                    scenario_id += 1
                    scenario_rows: list[dict[str, object]] = []
                    for intervention in INTERVENTIONS:
                        trajectory = simulate_cage_fish_response(
                            hab_pressure=float(np.clip(hab_pressure * hab_multiplier, 0, 100)),
                            mhw_intensity_c=float(np.clip(mhw_intensity_c + mhw_delta, 0, 6)),
                            dissolved_oxygen_mg_l=float(np.clip(dissolved_oxygen_mg_l + do_delta, 0, 14)),
                            stocking_density_kg_m3=float(np.clip(
                                stocking_density_kg_m3 * density_multiplier, 1, 60
                            )),
                            planned_feeding_pct=planned_feeding_pct,
                            hab_duration_hours=hab_duration_hours,
                            intervention=intervention,
                            horizon_hours=horizon_hours,
                        )
                        summary = _summarise(trajectory)
                        summary.update({
                            "scenario_id": scenario_id,
                            "hab_multiplier": hab_multiplier,
                            "mhw_delta_c": mhw_delta,
                            "do_delta_mg_l": do_delta,
                            "density_multiplier": density_multiplier,
                        })
                        scenario_rows.append(summary)
                    scenario_frame = pd.DataFrame(scenario_rows)
                    baseline_load = float(scenario_frame.loc[
                        scenario_frame["intervention"].eq("维持监测"),
                        "pressure_load_index_hours",
                    ].iloc[0])
                    scenario_frame["pressure_load_reduction_pct"] = 100.0 * (
                        baseline_load - scenario_frame["pressure_load_index_hours"]
                    ) / max(baseline_load, 1e-9)
                    best_load = float(scenario_frame["pressure_load_index_hours"].min())
                    scenario_frame["lowest_pressure"] = np.isclose(
                        scenario_frame["pressure_load_index_hours"], best_load
                    )
                    pareto = []
                    for _, candidate in scenario_frame.iterrows():
                        dominated = (
                            (scenario_frame["pressure_load_index_hours"] <= candidate["pressure_load_index_hours"])
                            & (scenario_frame["mean_feeding_opportunity_pct"] >= candidate["mean_feeding_opportunity_pct"])
                            & (
                                (scenario_frame["pressure_load_index_hours"] < candidate["pressure_load_index_hours"])
                                | (scenario_frame["mean_feeding_opportunity_pct"] > candidate["mean_feeding_opportunity_pct"])
                            )
                        ).any()
                        pareto.append(not bool(dominated))
                    scenario_frame["pareto_efficient"] = pareto
                    rows.extend(scenario_frame.to_dict(orient="records"))

    detail = pd.DataFrame(rows)
    summary = detail.groupby("intervention", as_index=False).agg(
        scenarios=("scenario_id", "nunique"),
        median_pressure_reduction_pct=("pressure_load_reduction_pct", "median"),
        worst_case_pressure_reduction_pct=("pressure_load_reduction_pct", "min"),
        best_case_pressure_reduction_pct=("pressure_load_reduction_pct", "max"),
        median_feeding_opportunity_pct=("mean_feeding_opportunity_pct", "median"),
        lowest_pressure_frequency=("lowest_pressure", "mean"),
        pareto_frequency=("pareto_efficient", "mean"),
    )
    summary["lowest_pressure_frequency"] *= 100.0
    summary["pareto_frequency"] *= 100.0
    summary = summary.sort_values(
        ["pareto_frequency", "median_pressure_reduction_pct"],
        ascending=False,
        ignore_index=True,
    )
    card = {
        "scenario_count": int(detail["scenario_id"].nunique()),
        "perturbed_inputs": perturbations,
        "interpretation": (
            "Pareto frequency reports how often an intervention was not dominated on "
            "both pressure load and feeding opportunity across nearby input scenarios."
        ),
        "boundary": (
            "Input robustness check for a dimensionless prototype; not evidence of "
            "field efficacy, mortality reduction or economic benefit."
        ),
    }
    return {"detail": detail, "summary": summary, "card": card}
