"""Synthetic scenario-to-map projection for the public web demo.

The locations and scores in this module are illustrative. They make the
software output spatially legible without claiming a calibrated operational
forecast for any real ocean region.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


SCENARIO_PRESETS = {
    "复合高风险情景": {
        "mhw_intensity_c": 2.8,
        "nitrate_mmol_m3": 5.5,
        "phosphate_mmol_m3": 0.75,
        "silicate_mmol_m3": 7.0,
        "transport_proxy": 0.85,
    },
    "热异常主导情景": {
        "mhw_intensity_c": 3.2,
        "nitrate_mmol_m3": 2.2,
        "phosphate_mmol_m3": 0.30,
        "silicate_mmol_m3": 3.0,
        "transport_proxy": 0.65,
    },
    "营养输入主导情景": {
        "mhw_intensity_c": 1.4,
        "nitrate_mmol_m3": 7.0,
        "phosphate_mmol_m3": 1.00,
        "silicate_mmol_m3": 9.0,
        "transport_proxy": 0.75,
    },
    "低风险对照情景": {
        "mhw_intensity_c": 0.4,
        "nitrate_mmol_m3": 1.2,
        "phosphate_mmol_m3": 0.15,
        "silicate_mmol_m3": 2.0,
        "transport_proxy": 0.30,
    },
}


# Named ocean areas make the map intuitive, but coordinates are deliberately
# coarse demonstration anchors rather than monitoring stations or forecast
# grid cells.
DEMO_ZONES = (
    {
        "zone": "北阿拉斯加湾",
        "latitude": 57.0,
        "longitude": -145.0,
        "heat_sensitivity": 0.92,
        "nutrient_sensitivity": 0.62,
        "transport_sensitivity": 0.86,
        "production_context": "北太平洋渔业与近岸养殖背景",
    },
    {
        "zone": "加州沿岸",
        "latitude": 35.5,
        "longitude": -123.0,
        "heat_sensitivity": 0.88,
        "nutrient_sensitivity": 0.82,
        "transport_sensitivity": 0.80,
        "production_context": "加州上升流渔业与贝类养殖背景",
    },
    {
        "zone": "墨西哥湾",
        "latitude": 25.5,
        "longitude": -89.5,
        "heat_sensitivity": 0.72,
        "nutrient_sensitivity": 1.00,
        "transport_sensitivity": 0.92,
        "production_context": "墨西哥湾渔业与贝类养殖背景",
    },
    {
        "zone": "北大西洋中部",
        "latitude": 36.0,
        "longitude": -40.0,
        "heat_sensitivity": 0.58,
        "nutrient_sensitivity": 0.48,
        "transport_sensitivity": 0.58,
        "production_context": "远洋渔业背景",
    },
    {
        "zone": "日本黑潮延伸区",
        "latitude": 34.5,
        "longitude": 143.0,
        "heat_sensitivity": 1.00,
        "nutrient_sensitivity": 0.76,
        "transport_sensitivity": 1.00,
        "production_context": "日本海水养殖与近海渔业背景",
    },
    {
        "zone": "中国南部近岸",
        "latitude": 21.8,
        "longitude": 113.4,
        "heat_sensitivity": 0.86,
        "nutrient_sensitivity": 0.94,
        "transport_sensitivity": 0.88,
        "production_context": "中国南部海水养殖背景",
    },
    {
        "zone": "南澳大利亚近岸",
        "latitude": -35.2,
        "longitude": 137.6,
        "heat_sensitivity": 0.80,
        "nutrient_sensitivity": 0.68,
        "transport_sensitivity": 0.96,
        "production_context": "南澳大利亚贝类与海水鱼养殖背景",
    },
    {
        "zone": "挪威海—峡湾沿岸",
        "latitude": 65.0,
        "longitude": 10.0,
        "heat_sensitivity": 0.70,
        "nutrient_sensitivity": 0.58,
        "transport_sensitivity": 0.84,
        "production_context": "北大西洋鲑鱼网箱养殖背景",
    },
    {
        "zone": "东地中海—爱琴海",
        "latitude": 37.2,
        "longitude": 25.2,
        "heat_sensitivity": 0.94,
        "nutrient_sensitivity": 0.64,
        "transport_sensitivity": 0.72,
        "production_context": "海鲈与金头鲷网箱养殖背景",
    },
    {
        "zone": "西印度洋—阿拉伯海",
        "latitude": 9.0,
        "longitude": 63.0,
        "heat_sensitivity": 0.91,
        "nutrient_sensitivity": 0.72,
        "transport_sensitivity": 0.88,
        "production_context": "印度洋金枪鱼与沿岸渔业背景",
    },
    {
        "zone": "秘鲁—智利洪堡流",
        "latitude": -22.0,
        "longitude": -73.5,
        "heat_sensitivity": 0.78,
        "nutrient_sensitivity": 1.00,
        "transport_sensitivity": 0.93,
        "production_context": "高生产力上升流与小型中上层鱼渔业背景",
    },
    {
        "zone": "智利巴塔哥尼亚峡湾",
        "latitude": -43.5,
        "longitude": -73.5,
        "heat_sensitivity": 0.74,
        "nutrient_sensitivity": 0.66,
        "transport_sensitivity": 0.90,
        "production_context": "南美鲑鱼网箱养殖背景",
    },
)


def _risk_level(score: float) -> str:
    if score >= 85:
        return "极高"
    if score >= 70:
        return "高"
    if score >= 50:
        return "中等"
    if score >= 30:
        return "关注"
    return "低"


def project_synthetic_scenario(
    issue_date: date,
    horizon_days: int,
    mhw_intensity_c: float,
    nitrate_mmol_m3: float,
    phosphate_mmol_m3: float,
    silicate_mmol_m3: float,
    transport_proxy: float,
) -> pd.DataFrame:
    """Project one hypothetical compound scenario onto representative ocean zones.

    The output is a dimensionless visual index, not an estimated cell count,
    chlorophyll concentration, toxin concentration, or event probability.
    """
    heat = np.clip(mhw_intensity_c / 4.0, 0.0, 1.25)
    nitrate = np.clip(nitrate_mmol_m3 / 8.0, 0.0, 1.25)
    phosphate = np.clip(phosphate_mmol_m3 / 1.2, 0.0, 1.25)
    silicate = np.clip(silicate_mmol_m3 / 10.0, 0.0, 1.25)
    nutrient = 0.50 * nitrate + 0.30 * phosphate + 0.20 * silicate
    transport = np.clip(transport_proxy, 0.0, 1.0)
    compound = heat * (0.45 + 0.55 * nutrient) * (0.55 + 0.45 * transport)

    center = issue_date + timedelta(days=int(horizon_days))
    window_start = center - timedelta(days=3)
    window_end = center + timedelta(days=3)
    rows: list[dict[str, object]] = []

    for zone in DEMO_ZONES:
        heat_component = heat * float(zone["heat_sensitivity"])
        nutrient_component = nutrient * float(zone["nutrient_sensitivity"])
        transport_component = transport * float(zone["transport_sensitivity"])

        risk_logit = (
            -2.65
            + 2.60 * heat_component
            + 2.00 * nutrient_component
            + 1.05 * transport_component
            + 1.20 * compound
        )
        risk_score = float(100.0 / (1.0 + np.exp(-risk_logit)))
        intensity_index = float(100.0 * np.clip(
            0.50 * heat_component
            + 0.35 * nutrient_component
            + 0.15 * transport_component,
            0.0,
            1.0,
        ))

        rows.append({
            "候选海区": zone["zone"],
            "latitude": zone["latitude"],
            "longitude": zone["longitude"],
            "综合风险指数": round(risk_score, 1),
            "预计藻华强度指数": round(intensity_index, 1),
            "风险等级": _risk_level(risk_score),
            "情景证据等级": "C（环境条件候选）",
            "海区生产背景": zone["production_context"],
            "数据属性": "代表性海区情景锚点（非实时观测）",
            "预计时间": center.isoformat(),
            "预计窗口": f"{window_start.isoformat()} 至 {window_end.isoformat()}",
        })

    return pd.DataFrame(rows).sort_values("综合风险指数", ascending=False, ignore_index=True)
