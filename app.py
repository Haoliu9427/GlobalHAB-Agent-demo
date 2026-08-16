"""Interactive Streamlit front end for the GlobalHAB-Agent minimal demo."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo import (  # noqa: E402
    ExperimentAction,
    HypothesisAgent,
    SCENARIO_PRESETS,
    evaluate_action,
    evaluate_seasonal_baseline,
    generate_demo_data,
    project_synthetic_scenario,
)
from globalhab_demo.data import REGIONS  # noqa: E402


st.set_page_config(
    page_title="GlobalHAB-Agent Demo",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {background:#f6f9fb; border:1px solid #dce6ec;
        border-radius:10px; padding:0.65rem 0.8rem;}
    .demo-note {background:#fff8e8; border-left:4px solid #e39a2f;
        padding:0.75rem 1rem; border-radius:6px; color:#493820;}
    .map-summary {background:linear-gradient(105deg,#eef8f7,#f6fbfd);
        border:1px solid #cfe3e5; border-radius:10px; padding:0.75rem 1rem;
        margin-bottom:0.4rem; color:#173e45;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def run_agent_demo(
    days: int,
    seed: int,
    budget: int,
    holdout_region: str,
    test_fraction: float,
):
    frame = generate_demo_data(days=days, seed=seed)
    baseline = evaluate_seasonal_baseline(frame, holdout_region, test_fraction, seed)
    actions = [
        ExperimentAction(route, lag, model)
        for route, lag, model in product(
            ("local", "downstream"),
            (7, 14, 30),
            ("logistic", "random_forest"),
        )
    ]
    agent = HypothesisAgent(actions, budget)
    predictions_by_action: dict[str, pd.DataFrame] = {}
    for _ in range(budget):
        action = agent.next_action()
        feedback, predictions = evaluate_action(
            frame,
            action,
            holdout_region,
            test_fraction,
            seed,
            baseline["pr_auc"],
        )
        agent.observe(feedback)
        predictions_by_action[action.action_id] = predictions

    log = pd.DataFrame(agent.log).sort_values("step")
    best = agent.best_result()
    best_predictions = predictions_by_action[str(best["action_id"])]
    recovered = best["route"] == "downstream" and int(best["lag_days"]) == 14
    card = {
        "demo_status": "synthetic_software_verification_only",
        "best_candidate": best,
        "synthetic_ground_truth": {
            "route": "downstream",
            "lag_days": 14,
            "recovered_by_agent": recovered,
        },
        "validation": {
            "time_block": f"last {test_fraction:.0%} of dates",
            "spatial_block": holdout_region,
            "random_split_used": False,
        },
        "scientific_variable_rules": {
            "mhw_intensity": "SST − seasonal climatological mean on p90 exceedance days",
            "nutrients": ["nitrate", "phosphate", "silicate"],
            "microplastics": (
                "proxy for transport/residence/convergence setting only; "
                "not current velocity/direction or a direct HAB driver"
            ),
        },
        "applicability_boundary": [
            "synthetic anonymous regions",
            "abstract directed pathway rather than a physical current trajectory",
            "binary synthetic label without species or toxin confirmation",
        ],
    }
    return frame, baseline, log, best, best_predictions, card


st.title("GlobalHAB-Agent")
st.subheader("复杂情景下有害藻华提前预警的最小可运行演示")
st.markdown(
    '<div class="demo-note"><b>重要：</b>本网页使用匿名合成数据验证Agent闭环。'
    "所有结果均不是现实海区的HAB预测性能，也不能用于业务预警。</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("试跑设置")
    days = st.select_slider("合成序列长度（天）", options=[540, 720, 900], value=720)
    budget = st.slider("Agent实验预算", min_value=4, max_value=12, value=8)
    holdout_region = st.selectbox("完全留出区域", REGIONS, index=3)
    test_fraction = st.select_slider(
        "前向留出比例", options=[0.20, 0.25, 0.30], value=0.25,
        format_func=lambda value: f"{value:.0%}",
    )
    seed = st.number_input("固定随机种子", min_value=1, max_value=9999, value=42)
    run_clicked = st.button("运行 Agent 探索", type="primary", use_container_width=True)
    st.caption("首次打开自动准备默认结果；修改试跑设置后请点击按钮更新。")

st.markdown("### 科学变量与固定规则")
rule_a, rule_b, rule_c = st.columns(3)
rule_a.info("**MHW强度**\n\n超过日历日p90阈值时：SST − 季节气候平均值")
rule_b.info("**营养背景**\n\nNitrate、Phosphate和Silicate分别进入模型")
rule_c.info("**微塑料角色**\n\n仅代理输运、停留与汇聚背景，用于路径加权")

if run_clicked or "demo_result" not in st.session_state:
    spinner_text = (
        "正在生成合成观测、执行阻断验证并更新Agent探索日志……"
        if run_clicked
        else "正在准备默认试跑和情景预警地图……"
    )
    with st.spinner(spinner_text):
        st.session_state["demo_result"] = run_agent_demo(
            days, int(seed), budget, holdout_region, test_fraction
        )

frame, baseline, log, best, predictions, card = st.session_state["demo_result"]
recovered = bool(card["synthetic_ground_truth"]["recovered_by_agent"])

st.markdown("### 本次试跑结果")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("最佳候选", str(best["action_id"]))
m2.metric("PR-AUC", f"{float(best['pr_auc']):.3f}")
m3.metric("Brier", f"{float(best['brier']):.3f}")
m4.metric("Top 20%召回", f"{float(best['recall_at_top20']):.1%}")
m5.metric("恢复预设14天信号", "是" if recovered else "否")

st.markdown("### 情景化空间预警展示")
st.caption(
    "选择一种复合环境情景，查看未来7/14/30天五个演示海区的HAB风险位置、时间和相对强度。"
    "海区坐标和指数均用于界面演示，不代表真实业务预报。"
)

scenario_controls, scenario_map = st.columns([1.0, 2.25], gap="large")
with scenario_controls:
    preset_name = st.selectbox("复合环境情景", list(SCENARIO_PRESETS), index=0)
    preset = SCENARIO_PRESETS[preset_name]
    preset_index = list(SCENARIO_PRESETS).index(preset_name)
    issue_date = st.date_input("情景起报日期", value=pd.Timestamp.today().date())
    horizon_days = st.radio(
        "预警提前量",
        options=[7, 14, 30],
        index=1,
        horizontal=True,
        format_func=lambda value: f"{value}天",
    )
    mhw_intensity = st.slider(
        "海洋热浪强度（°C）",
        0.0,
        4.5,
        float(preset["mhw_intensity_c"]),
        0.1,
        key=f"scenario_mhw_{preset_index}",
        help="超过日历日p90阈值时，SST相对季节气候平均值的正异常。",
    )
    nitrate = st.slider(
        "Nitrate（mmol m⁻³）",
        0.0,
        10.0,
        float(preset["nitrate_mmol_m3"]),
        0.1,
        key=f"scenario_no3_{preset_index}",
    )
    phosphate = st.slider(
        "Phosphate（mmol m⁻³）",
        0.0,
        1.5,
        float(preset["phosphate_mmol_m3"]),
        0.05,
        key=f"scenario_po4_{preset_index}",
    )
    silicate = st.slider(
        "Silicate（mmol m⁻³）",
        0.0,
        12.0,
        float(preset["silicate_mmol_m3"]),
        0.1,
        key=f"scenario_sio3_{preset_index}",
    )
    transport_proxy = st.slider(
        "输运／停留／汇聚代理",
        0.0,
        1.0,
        float(preset["transport_proxy"]),
        0.05,
        key=f"scenario_transport_{preset_index}",
        help="由合成微塑料浓度映射的状态代理，不表示真实海流速度或方向。",
    )

scenario_result = project_synthetic_scenario(
    issue_date=issue_date,
    horizon_days=horizon_days,
    mhw_intensity_c=mhw_intensity,
    nitrate_mmol_m3=nitrate,
    phosphate_mmol_m3=phosphate,
    silicate_mmol_m3=silicate,
    transport_proxy=transport_proxy,
)
top_zone = scenario_result.iloc[0]

with scenario_map:
    st.markdown(
        '<div class="map-summary">当前情景下，最高候选风险位于'
        f'<b>{top_zone["候选海区"]}</b>；预计时间约为'
        f'<b>{top_zone["预计时间"]}</b>，综合风险指数'
        f'<b>{top_zone["综合风险指数"]:.1f}/100</b>，预计藻华强度指数'
        f'<b>{top_zone["预计藻华强度指数"]:.1f}/100</b>。</div>',
        unsafe_allow_html=True,
    )
    map_figure = px.scatter_geo(
        scenario_result,
        lat="latitude",
        lon="longitude",
        text="候选海区",
        hover_name="候选海区",
        size="预计藻华强度指数",
        color="综合风险指数",
        range_color=[0, 100],
        size_max=38,
        color_continuous_scale=[
            [0.00, "#2b83ba"],
            [0.30, "#63b995"],
            [0.50, "#f2cf5b"],
            [0.70, "#f28e46"],
            [1.00, "#c9363e"],
        ],
        custom_data=["综合风险指数", "预计藻华强度指数", "风险等级", "预计窗口"],
    )
    map_figure.update_traces(
        textposition="top center",
        textfont=dict(family="Microsoft YaHei, Arial, sans-serif", size=12, color="#18353d"),
        marker=dict(line=dict(width=1.2, color="#ffffff"), opacity=0.90),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "综合风险指数：%{customdata[0]:.1f}/100<br>"
            "预计藻华强度：%{customdata[1]:.1f}/100<br>"
            "风险等级：%{customdata[2]}<br>"
            "预计窗口：%{customdata[3]}<extra></extra>"
        ),
    )
    map_figure.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#e8ecea",
        showocean=True,
        oceancolor="#dceff3",
        showlakes=True,
        lakecolor="#dceff3",
        showcountries=True,
        countrycolor="#ffffff",
        showcoastlines=True,
        coastlinecolor="#78939b",
        coastlinewidth=0.8,
        lataxis_range=[-58, 78],
        bgcolor="#ffffff",
    )
    map_figure.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Microsoft YaHei, Arial, sans-serif", size=12, color="#18353d"),
        coloraxis_colorbar=dict(title="风险指数", len=0.62, thickness=14),
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(
        map_figure,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )
    st.caption("圆点颜色表示综合风险，圆点大小表示预计相对强度；鼠标悬停可查看时间窗口。")

st.markdown(
    f"**当前输入条件：** MHW强度 {mhw_intensity:.1f} °C；Nitrate {nitrate:.1f}、"
    f"Phosphate {phosphate:.2f}、Silicate {silicate:.1f} mmol m⁻³；"
    f"输运／停留／汇聚代理 {transport_proxy:.2f}。"
)
st.dataframe(
    scenario_result[[
        "候选海区", "风险等级", "综合风险指数", "预计藻华强度指数", "预计窗口"
    ]],
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "下载本次情景地图结果 CSV",
    scenario_result.to_csv(index=False).encode("utf-8-sig"),
    file_name="synthetic_scenario_map.csv",
    mime="text/csv",
)
st.info(
    "读图边界：该图回答的是“如果出现这组假设条件，系统将怎样呈现候选风险”。"
    "强度为无量纲演示指数，不是藻细胞密度、叶绿素或毒素浓度；真实预警必须接入实时观测并经现场复核。"
)

tab_data, tab_agent, tab_risk, tab_card = st.tabs(
    ["多源合成观测", "Agent探索日志", "阻断验证风险序列", "发现卡与边界"]
)

with tab_data:
    selected_region = st.selectbox("查看区域", REGIONS, index=3, key="plot_region")
    region_data = frame[frame["region"].eq(selected_region)].copy().set_index("date")
    st.markdown("##### SST、气候态和MHW阈值")
    st.line_chart(region_data[[
        "sst_c", "climatological_mean_sst_c", "climatological_p90_sst_c"
    ]], height=280)
    st.code(
        "is_mhw = SST > climatological_p90\n"
        "mhw_intensity = SST - climatological_mean  # only on MHW days",
        language="python",
    )
    left, right = st.columns(2)
    with left:
        st.markdown("##### 分项营养盐")
        st.line_chart(region_data[[
            "nitrate_mmol_m3", "phosphate_mmol_m3", "silicate_mmol_m3"
        ]], height=260)
    with right:
        st.markdown("##### 微塑料与环流状态代理")
        st.line_chart(region_data[[
            "microplastic_concentration_items_m3", "circulation_residence_proxy"
        ]], height=260)
        st.caption("两列量纲不同，仅用于查看同步变化；微塑料不是流速或流向。")
    st.dataframe(frame.head(30), use_container_width=True, hide_index=True)

with tab_agent:
    st.markdown("##### 每项候选实验的PR-AUC")
    chart = log[["action_id", "pr_auc"]].set_index("action_id")
    st.bar_chart(chart, height=320)
    st.markdown("##### 可回溯探索日志")
    display_columns = [
        "step", "action_id", "pr_auc", "pr_auc_gain", "brier",
        "recall_at_top20", "utility", "budget_remaining",
    ]
    st.dataframe(log[display_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "下载探索日志 CSV",
        log.to_csv(index=False).encode("utf-8-sig"),
        file_name="agent_log.csv",
        mime="text/csv",
    )

with tab_risk:
    risk_plot = predictions.set_index("date")[["risk_probability", "hab_event"]]
    st.line_chart(risk_plot, height=340)
    st.caption(
        "风险概率与合成事件仅用于检查预警输出的数据结构。卫星候选区不能替代物种、毒素或现场确认。"
    )
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "下载风险结果 CSV",
        predictions.to_csv(index=False).encode("utf-8-sig"),
        file_name="risk_predictions.csv",
        mime="text/csv",
    )

with tab_card:
    st.json(card)
    card_bytes = json.dumps(card, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    st.download_button(
        "下载发现卡 JSON",
        card_bytes,
        file_name="discovery_card.json",
        mime="application/json",
    )
    st.warning(
        "失败标准：若最佳结果依赖随机划分、无法恢复预设传导信号，或只在单一区域/阈值成立，"
        "应保留失败日志并收缩假设，而不是将其包装为科学发现。"
    )

st.divider()
st.caption(
    "GlobalHAB-Agent v2.3 Scenario Map Demo · Synthetic software verification only · "
    "No operational or causal claim"
)
