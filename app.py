"""Interactive Streamlit front end for the GlobalHAB-Agent minimal demo."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo import (  # noqa: E402
    ExperimentAction,
    HypothesisAgent,
    evaluate_action,
    evaluate_seasonal_baseline,
    generate_demo_data,
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
    st.caption("参考设置运行约3–8秒，具体取决于云端负载。")

st.markdown("### 科学变量与固定规则")
rule_a, rule_b, rule_c = st.columns(3)
rule_a.info("**MHW强度**\n\n超过日历日p90阈值时：SST − 季节气候平均值")
rule_b.info("**营养背景**\n\nNitrate、Phosphate和Silicate分别进入模型")
rule_c.info("**微塑料角色**\n\n仅代理输运、停留与汇聚背景，用于路径加权")

if not run_clicked and "demo_result" not in st.session_state:
    st.markdown("#### 演示流程")
    st.write(
        "点击左侧“运行 Agent 探索”。Agent将在固定预算内比较局地/沿流、"
        "7/14/30天滞后和两类轻量模型，并使用前向时间＋留一地区验证反馈更新实验顺序。"
    )
    st.stop()

if run_clicked:
    with st.spinner("正在生成合成观测、执行阻断验证并更新Agent探索日志……"):
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

tab_data, tab_agent, tab_risk, tab_card = st.tabs(
    ["多源合成观测", "Agent探索日志", "风险输出", "发现卡与边界"]
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
    "GlobalHAB-Agent v2.2 Web Demo · Synthetic software verification only · "
    "No operational or causal claim"
)

