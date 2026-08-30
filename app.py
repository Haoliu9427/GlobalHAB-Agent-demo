"""GlobalHAB-Agent v3.7.1 product-facing demo."""

from __future__ import annotations

import json
import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.aquaculture import (  # noqa: E402
    EVIDENCE_CONFIDENCE,
    MECHANISM_LABELS,
    PRODUCTION_PROFILES,
    project_aquaculture_risk,
)
from globalhab_demo.bio_response import (  # noqa: E402
    BIO_SCENARIO_PRESETS,
    INTERVENTIONS,
    compare_interventions,
    evaluate_intervention_robustness,
)
from globalhab_demo.data import REGIONS  # noqa: E402
from globalhab_demo.evidence import SOUTH_AUSTRALIA_CASE  # noqa: E402
from globalhab_demo.event_risk import (  # noqa: E402
    build_norway_risk_translation,
    build_sa_risk_translation,
)
from globalhab_demo.global_cases import (  # noqa: E402
    build_norway_replay,
    global_evidence_frame,
    load_norway_real_case,
)
from globalhab_demo.real_replay import (  # noqa: E402
    build_sa_replay,
    load_sa_real_case,
    real_data_router,
)
from globalhab_demo.real_benchmark import (  # noqa: E402
    run_forward_monitoring_benchmark,
)
from globalhab_demo.scenario import (  # noqa: E402
    SCENARIO_PRESETS,
    project_synthetic_scenario,
)
from globalhab_demo.workflow import run_exploration  # noqa: E402


REGION_LABELS = {
    "Synthetic_Region_A": "北太平洋副热带海域（合成数据）",
    "Synthetic_Region_B": "北大西洋副热带海域（合成数据）",
    "Synthetic_Region_C": "南大洋锋面海域（合成数据）",
    "Synthetic_Region_D": "西北太平洋黑潮延伸区（合成数据）",
}


st.set_page_config(
    page_title="GlobalHAB-Agent | GOAI复赛",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px;}
    h1, h2, h3 {letter-spacing:-0.02em;}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#f3f8f7 0%,#edf3f4 100%);}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {line-height:1.55;}
    [data-testid="stMetric"] {
        background:linear-gradient(135deg,#f7fbfc,#eef7f5);
        border:1px solid #cfe1df; border-radius:14px; padding:.75rem .85rem;
        box-shadow:0 5px 18px rgba(22,74,78,.05);
    }
    [data-testid="stMetricValue"] {white-space:normal; overflow:visible;}
    [data-testid="stMetricValue"] > div {white-space:normal; overflow:visible; text-overflow:clip;}
    .eyebrow {font-size:.78rem; font-weight:700; letter-spacing:.12em;
        color:#147a7e; text-transform:uppercase; margin-bottom:.55rem; line-height:1.4;}
    .hero {background:linear-gradient(120deg,#073b4c 0%,#0b6670 58%,#138a83 100%);
        color:white; padding:1.7rem 1.8rem 1.55rem; border-radius:22px; margin:.25rem 0 1.05rem;
        box-shadow:0 16px 38px rgba(4,52,63,.20); overflow:visible;}
    .hero h1 {font-size:2.35rem; margin:0 0 .28rem; color:white; line-height:1.16;}
    .hero .tagline {margin:0; color:#ddf7f3; font-size:1.18rem; font-weight:600;}
    .hero .value {margin:.62rem 0 0; color:#bde8e2; font-size:.92rem; max-width:820px; line-height:1.65;}
    .signal {background:linear-gradient(105deg,#eaf8f4,#f4fbfd);
        border:1px solid #bfe1d8; border-radius:12px; padding:.82rem 1rem;
        color:#123f45; margin-bottom:.5rem;}
    .case-card {background:#f7f9fc; border-left:4px solid #6a4c93;
        padding:.9rem 1.05rem; border-radius:10px; margin:.5rem 0;}
    .formula {background:#082f3a; color:#e8fbf7; padding:.8rem 1rem;
        border-radius:10px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
    .small-muted {color:#61777d; font-size:.84rem;}
    .kpi-grid {display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:.72rem;
        margin:.2rem 0 1.15rem;}
    .kpi-grid.kpi-3 {grid-template-columns:repeat(3,minmax(0,1fr));}
    .kpi {min-height:112px; background:linear-gradient(145deg,#ffffff,#eef8f6);
        border:1px solid #c9e0dc; border-radius:16px; padding:.82rem .9rem;
        box-shadow:0 7px 20px rgba(18,74,75,.06); display:flex; flex-direction:column;
        justify-content:space-between; min-width:0;}
    .kpi-label {font-size:.78rem; color:#557077; font-weight:650; line-height:1.3;}
    .kpi-value {font-size:1.42rem; color:#0c424b; font-weight:760; line-height:1.18;
        overflow-wrap:anywhere; word-break:normal;}
    .kpi-note {font-size:.69rem; color:#789096; line-height:1.25; margin-top:.28rem;}
    .product-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem;
        margin:.65rem 0 1rem;}
    .product-card {background:#f5f9fb; border:1px solid #d8e5e8; border-radius:14px;
        padding:1rem 1.05rem; min-height:132px;}
    .product-card b {color:#075b67; font-size:1rem;}
    .product-card p {color:#506a72; font-size:.86rem; line-height:1.65; margin:.5rem 0 0;}
    .case-badge {display:inline-block; color:#0b6d70; background:#e5f5f2; border-radius:999px;
        padding:.18rem .5rem; font-size:.72rem; font-weight:700; margin-bottom:.35rem;}
    .risk-bridge {background:linear-gradient(120deg,#f7fbfa,#eef7f6); border:1px solid #c7dfdb;
        border-radius:18px; padding:1rem 1.05rem .9rem; margin:.7rem 0 1rem;
        box-shadow:0 8px 24px rgba(18,74,75,.06);}
    .risk-bridge-title {font-size:1.05rem; color:#073f48; font-weight:780; margin-bottom:.18rem;}
    .risk-bridge-subtitle {font-size:.81rem; color:#647b81; line-height:1.55; margin-bottom:.78rem;}
    .risk-chain {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.58rem;}
    .risk-step {background:white; border:1px solid #d5e5e2; border-radius:13px; padding:.72rem .76rem;
        min-height:118px; position:relative;}
    .risk-step:not(:last-child)::after {content:'›'; position:absolute; right:-.48rem; top:42%;
        color:#66a8a4; font-size:1.35rem; font-weight:800; z-index:2;}
    .risk-step-title {font-size:.86rem; color:#0c5058; font-weight:750; margin:.23rem 0 .28rem;}
    .risk-step-value {font-size:.76rem; color:#405d63; line-height:1.46; overflow-wrap:anywhere;}
    .action-strip {background:#073f48; color:#edf9f7; border-radius:12px; padding:.76rem .9rem;
        margin:.75rem 0 .2rem; line-height:1.55; font-size:.82rem;}
    .action-strip b {color:white;}
    .footer-boundary {color:#7b898e; font-size:.76rem; line-height:1.65; text-align:center;
        max-width:1120px; margin:0 auto; padding:.4rem .5rem 0;}
    @media (max-width:1100px) {.kpi-grid{grid-template-columns:repeat(3,minmax(0,1fr));}}
    @media (max-width:700px) {
        .block-container{padding-top:1.3rem;}
        .hero{padding:1.25rem 1.1rem;}
        .hero h1{font-size:1.9rem;}
        .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
        .product-grid{grid-template-columns:1fr;}
        .risk-chain{grid-template-columns:1fr;}
        .risk-step:not(:last-child)::after{content:'⌄'; right:49%; top:auto; bottom:-.72rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_exploration(
    days: int,
    seed: int,
    budget: int,
    holdout_region: str,
    test_fraction: float,
):
    return run_exploration(
        days=days,
        seed=seed,
        budget=budget,
        holdout_region=holdout_region,
        test_fraction=test_fraction,
    )


@st.cache_data(show_spinner=False)
def cached_norway_benchmark(observations: pd.DataFrame):
    return run_forward_monitoring_benchmark(observations)


def kpi_grid(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        '<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        '</div>'
        for label, value, note in items
    )
    grid_class = " kpi-3" if len(items) == 3 else ""
    st.markdown(f'<div class="kpi-grid{grid_class}">{cards}</div>', unsafe_allow_html=True)


def risk_translation_panel(summary: dict[str, object], evidence: pd.DataFrame) -> None:
    """Show how a real observation becomes a bounded monitoring decision."""
    steps = [
        ("现场观测", str(summary["observation"])),
        ("危害证据", str(summary["hazard"])),
        ("养殖暴露", str(summary["exposure"])),
        ("对象脆弱性", str(summary["vulnerability"])),
        ("复核优先级", str(summary["priority"])),
    ]
    cards = "".join(
        '<div class="risk-step">'
        f'<div class="risk-step-title">{html.escape(title)}</div>'
        f'<div class="risk-step-value">{html.escape(value)}</div>'
        '</div>'
        for title, value in steps
    )
    st.markdown(
        '<div class="risk-bridge">'
        '<div class="risk-bridge-title">从观测到监测优先级</div>'
        '<div class="risk-bridge-subtitle">将现场观测、危害证据、养殖暴露和对象脆弱性分开呈现，'
        '最后给出可供现场复核的相对顺序。</div>'
        f'<div class="risk-chain">{cards}</div>'
        '<div class="action-strip"><b>建议行动：</b>'
        f'{html.escape(str(summary["action"]))}<br><b>结论边界：</b>'
        f'{html.escape(str(summary["boundary"]))}</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("查看研判证据清单：哪些已观测、哪些是假设、哪些仍缺失"):
        evidence_display = evidence[["研判输入", "当前证据", "证据属性", "如何影响研判"]]
        st.dataframe(
            evidence_display,
            width="stretch",
            hide_index=True,
            column_config={
                "研判输入": st.column_config.TextColumn("研判输入", width="small"),
                "当前证据": st.column_config.TextColumn("当前证据", width="large"),
                "证据属性": st.column_config.TextColumn("属性", width="small"),
                "如何影响研判": st.column_config.TextColumn("研判作用", width="large"),
            },
        )


def global_case_map(frame: pd.DataFrame) -> go.Figure:
    status_color = {
        "完整观测回放": "#0b7c78",
        "研究证据接口": "#6a4c93",
        "全球背景证据": "#d08a32",
    }
    fig = go.Figure()
    for status, subset in frame.groupby("product_status"):
        fig.add_trace(go.Scattergeo(
            lon=subset["longitude"], lat=subset["latitude"], text=subset["case"],
            customdata=subset[["region", "period", "journal", "evidence", "records"]],
            mode="markers+text", textposition="top center", name=status,
            marker={
                "size": 20 if status == "完整观测回放" else 15,
                "color": status_color[status], "line": {"color": "white", "width": 1.3},
                "opacity": .92,
            },
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]} · %{customdata[1]}"
                "<br>%{customdata[2]}<br>%{customdata[3]}<br>%{customdata[4]}<extra></extra>"
            ),
        ))
    fig.update_geos(
        projection_type="natural earth", showland=True, landcolor="#edf0eb",
        showocean=True, oceancolor="#dceff2", showcountries=True,
        countrycolor="#ffffff", showcoastlines=True, coastlinecolor="#79939a",
    )
    fig.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 5, "b": 0},
        legend={"orientation": "h", "y": 0.02, "x": .5, "xanchor": "center"},
        paper_bgcolor="white", font={"family": "Microsoft YaHei, Arial", "size": 12},
    )
    return fig


def bloom_map(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=frame["longitude"],
        lat=frame["latitude"],
        text=frame["候选海区"],
        customdata=frame[["综合风险指数", "预计藻华强度指数", "风险等级", "预计窗口"]],
        mode="markers+text",
        textposition="top center",
        marker={
            "size": 10 + frame["预计藻华强度指数"] * 0.28,
            "color": frame["综合风险指数"],
            "cmin": 0,
            "cmax": 100,
            "colorscale": [
                [0.00, "#2b83ba"], [0.35, "#55b79a"], [0.55, "#f2cf5b"],
                [0.75, "#f28e46"], [1.00, "#c9363e"],
            ],
            "colorbar": {"title": "HAB风险", "thickness": 13, "len": .60},
            "line": {"width": 1.1, "color": "white"},
            "opacity": .92,
        },
        hovertemplate=(
            "<b>%{text}</b><br>HAB风险：%{customdata[0]:.1f}/100"
            "<br>相对强度：%{customdata[1]:.1f}/100"
            "<br>等级：%{customdata[2]}<br>窗口：%{customdata[3]}<extra></extra>"
        ),
    ))
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#e9eeeb",
        showocean=True, oceancolor="#dff1f4",
        showcountries=True, countrycolor="#ffffff",
        showcoastlines=True, coastlinecolor="#78939b",
        lataxis_range=[-58, 78],
    )
    fig.update_layout(
        height=510, margin={"l": 0, "r": 0, "t": 5, "b": 0},
        paper_bgcolor="white", font={"family": "Microsoft YaHei, Arial", "size": 12},
    )
    return fig


def aquaculture_map(frame: pd.DataFrame) -> go.Figure:
    colors = frame["养殖响应优先指数"]
    fig = go.Figure(go.Scattergeo(
        lon=frame["longitude"],
        lat=frame["latitude"],
        text=frame["候选海区"],
        customdata=frame[[
            "养殖响应优先指数", "响应级别", "不确定性下限", "不确定性上限",
            "养殖暴露度", "脆弱性系数",
        ]],
        mode="markers+text",
        textposition="top center",
        marker={
            "symbol": "diamond",
            "size": 12 + colors * .24,
            "color": colors,
            "cmin": 0,
            "cmax": 100,
            "colorscale": [[0, "#3b82a0"], [.5, "#f4bf4f"], [1, "#b51f3d"]],
            "colorbar": {"title": "响应优先", "thickness": 13, "len": .60},
            "line": {"width": 1.2, "color": "white"},
        },
        hovertemplate=(
            "<b>%{text}</b><br>响应优先：%{customdata[0]:.1f}/100"
            "<br>%{customdata[1]}<br>不确定区间：%{customdata[2]:.1f}–%{customdata[3]:.1f}"
            "<br>暴露：%{customdata[4]:.2f} · 脆弱性：%{customdata[5]:.2f}<extra></extra>"
        ),
    ))
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#ecefea",
        showocean=True, oceancolor="#e0f1f4",
        showcountries=True, countrycolor="#ffffff",
        showcoastlines=True, coastlinecolor="#78939b",
        lataxis_range=[-58, 78],
    )
    fig.update_layout(
        height=470, margin={"l": 0, "r": 0, "t": 5, "b": 0},
        paper_bgcolor="white", font={"family": "Microsoft YaHei, Arial", "size": 12},
    )
    return fig


def real_qpcr_map(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scattergeo(
        lon=frame["longitude"], lat=frame["latitude"], text=frame["location"],
        customdata=frame[[
            "k_cristata_peak_cells_l", "observed_abundance_band", "samples",
            "sampling_dates", "k_cristata_detection_share",
        ]],
        mode="markers+text", textposition="top center",
        marker={
            "size": frame["marker_size"],
            "color": frame["k_cristata_max_share"], "cmin": 0, "cmax": 1,
            "colorscale": [[0, "#2b83ba"], [.5, "#f2cf5b"], [1, "#b51f3d"]],
            "colorbar": {"title": "K. cristata占比", "thickness": 13, "len": .65},
            "line": {"width": 1.1, "color": "white"}, "opacity": .9,
        },
        hovertemplate=(
            "<b>%{text}</b><br>K. cristata峰值：%{customdata[0]:,.0f} cells L⁻¹"
            "<br>展示分档：%{customdata[1]}<br>样本：%{customdata[2]}"
            " · 日期：%{customdata[3]}<br>检出样本占比：%{customdata[4]:.1%}<extra></extra>"
        ),
    ))
    fig.update_geos(
        projection_type="mercator", showland=True, landcolor="#ede9dd",
        showocean=True, oceancolor="#dff1f4", showcoastlines=True,
        coastlinecolor="#617d83", lonaxis_range=[135.3, 139.3],
        lataxis_range=[-36.0, -33.4], fitbounds=False,
    )
    fig.update_layout(
        height=510, margin={"l": 0, "r": 0, "t": 5, "b": 0},
        paper_bgcolor="white", font={"family": "Microsoft YaHei, Arial", "size": 12},
    )
    return fig


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow" style="color:#a7eee2">GOAI · AI FOR RESEARCH</div>
      <h1>GlobalHAB-Agent</h1>
      <p class="tagline">让Agent发现跨区域藻华信号，并把证据转化为可复核的养殖响应</p>
      <p class="value"><b>研究目标：</b>在事件稀有、观测稀疏且证据异质的条件下，
      识别具有时间方向的HAB传播信号，并将结果用于风险研判和网箱鱼干预情景比较。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 运行设置")
    days = st.select_slider("数据序列长度", [540, 720, 900], value=720)
    budget = st.slider("实验预算", 4, 12, 8)
    holdout_region = st.selectbox(
        "留出区域", REGIONS, index=3,
        format_func=lambda value: REGION_LABELS[value],
        help="该海域完全不参与模型拟合，用于检验跨区域适用性；名称对应合成数据中的匿名海域。",
    )
    test_fraction = st.select_slider(
        "前向测试比例", [0.20, 0.25, 0.30], value=0.25,
        format_func=lambda value: f"{value:.0%}",
    )
    seed = st.number_input("随机种子", 1, 9999, 42)
    run_clicked = st.button("运行分析", type="primary", width="stretch")
    st.caption("可调整设置并重新运行；实验结果和验证指标会自动记录。")

if run_clicked or "exploration" not in st.session_state:
    with st.spinner("正在执行阻断验证、随机参照与负对照……"):
        st.session_state["exploration"] = cached_exploration(
            days, int(seed), budget, holdout_region, test_fraction
        )

result = st.session_state["exploration"]
frame = result["frame"]
baselines = result["baselines"]
log = result["log"]
best = result["best"]
predictions = result["predictions"]
controls = result["controls"]
random_ref = result["random_reference"]
card = result["card"]
anomaly_daily = result["anomaly_daily"]
anomaly_events = result["anomaly_events"]
router_trace = result["router_trace"]
te_cte_network = result["te_cte_network"]
te_cte_lag_summary = result["te_cte_lag_summary"]
spatial_effects = result["spatial_effects"]
spatial_diagnostics = result["spatial_diagnostics"]
recovered = bool(card["synthetic_ground_truth"]["recovered_by_agent"])

kpi_grid([
    ("预设传播信号", "已恢复" if recovered else "待确认", "沿流方向与时间滞后"),
    ("识别出的传播模式", f"沿流传播 · {int(best['lag_days'])}天", "留区与前向阻断结果"),
    ("基准 · Average Precision", f"{float(best['pr_auc']):.3f}", "合成数据上的排序能力"),
    ("基准 · Brier Skill", f"{float(best['brier_skill']):.3f}", "相对气候概率基准"),
    ("高风险区事件覆盖", f"{float(best['recall_at_top20']):.1%}", "最高20%容量内的事件比例"),
    ("基准 · 校准误差", f"{float(best['ece']):.3f}", "越接近0越稳定"),
])

tab_alert, tab_real, tab_bio, tab_methods, tab_agent, tab_evidence = st.tabs([
    "风险研判", "真实事件回放", "生物响应沙盘",
    "科学解释", "探索与验证", "数据来源与复核",
])

with tab_alert:
    st.markdown("### 未来7/14/30天藻华风险情景推演")
    control_col, map_col = st.columns([1.0, 2.25], gap="large")
    with control_col:
        preset_name = st.selectbox("复合环境情景", list(SCENARIO_PRESETS), index=0)
        preset = SCENARIO_PRESETS[preset_name]
        issue_date = st.date_input("情景起报日期", value=pd.Timestamp.today().date())
        horizon_days = st.radio(
            "预警窗口", [7, 14, 30], index=1, horizontal=True,
            format_func=lambda value: f"{value}天",
        )
        mhw = st.slider(
            "海洋热浪强度（°C）", 0.0, 4.5, float(preset["mhw_intensity_c"]), .1,
            key=f"mhw_{preset_name}",
        )
        nitrate = st.slider(
            "硝酸盐 Nitrate（mmol m⁻³）", 0.0, 10.0, float(preset["nitrate_mmol_m3"]), .1,
            key=f"nitrate_{preset_name}",
        )
        phosphate = st.slider(
            "磷酸盐 Phosphate（mmol m⁻³）", 0.0, 1.5, float(preset["phosphate_mmol_m3"]), .05,
            key=f"phosphate_{preset_name}",
        )
        silicate = st.slider(
            "硅酸盐 Silicate（mmol m⁻³）", 0.0, 12.0, float(preset["silicate_mmol_m3"]), .1,
            key=f"silicate_{preset_name}",
        )
        transport = st.slider(
            "水团停留与汇聚背景", 0.0, 1.0, float(preset["transport_proxy"]), .05,
            help="由微塑料浓度的有界变换构造，仅代理水团状态，不是流速或流向。",
            key=f"transport_{preset_name}",
        )

    scenario = project_synthetic_scenario(
        issue_date, horizon_days, mhw, nitrate, phosphate, silicate, transport
    )
    top = scenario.iloc[0]
    with map_col:
        st.markdown(
            '<div class="signal">当前情景最高候选区：'
            f'<b>{top["候选海区"]}</b> · {top["预计窗口"]} · '
            f'HAB风险 <b>{top["综合风险指数"]:.1f}/100</b>。'
            '颜色表示风险，圆点大小表示相对强度。</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(bloom_map(scenario), width="stretch", config={"displayModeBar": False})

    st.markdown("### 海水养殖响应优先级")
    st.markdown(
        '<div class="formula">响应优先指数 = HAB危害 × 养殖暴露 × 对象脆弱性；'
        '证据等级单独控制不确定性宽度，不用“低置信度”掩盖潜在高风险。</div>',
        unsafe_allow_html=True,
    )
    a1, a2, a3, a4 = st.columns(4)
    production = a1.selectbox("主要养殖对象", list(PRODUCTION_PROFILES))
    mechanism = a2.selectbox(
        "危害机制证据", list(MECHANISM_LABELS),
        format_func=lambda key: MECHANISM_LABELS[key],
    )
    evidence_grade = a3.selectbox("现场证据等级", list(EVIDENCE_CONFIDENCE), index=2)
    farm_exposure = a4.slider("养殖暴露情景", .25, 1.25, .85, .05)
    aqua = project_aquaculture_risk(
        scenario, production, mechanism, farm_exposure, evidence_grade
    )
    top_aqua = aqua.iloc[0]
    st.markdown(
        '<div class="signal">首要核查区：'
        f'<b>{top_aqua["候选海区"]}</b> · {top_aqua["响应级别"]} · '
        f'响应优先指数 <b>{top_aqua["养殖响应优先指数"]:.1f}/100</b> '
        f'（不确定区间 {top_aqua["不确定性下限"]:.1f}–{top_aqua["不确定性上限"]:.1f}）。</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(aquaculture_map(aqua), width="stretch", config={"displayModeBar": False})
    st.dataframe(
        aqua[[
            "候选海区", "响应级别", "养殖响应优先指数", "不确定性下限",
            "不确定性上限", "养殖暴露度", "脆弱性系数", "建议动作",
        ]],
        width="stretch", hide_index=True,
    )
    st.warning(
        "响应级别用于安排复核和监测顺序。贝类停采、鱼类转移等业务动作必须由真实藻种、"
        "毒素、溶解氧、现场生物反应和当地监管规则共同决定。"
    )
    st.download_button(
        "下载养殖风险结果 CSV",
        aqua.to_csv(index=False).encode("utf-8-sig"),
        "aquaculture_response_priority.csv",
        "text/csv",
    )

with tab_real:
    st.markdown("### 全球真实观测与前沿研究证据")
    st.markdown(
        '<div class="signal"><b>两类真实证据分层使用：</b>南澳qPCR用于事件回放，不参与训练；'
        '挪威长期监测除回放外，另设严格前向的“下一次观测样本”回顾基准。'
        '两者都不与合成基准混合，暴露或监管证据缺失时明确降级为“情景假设/待补数据”。</div>',
        unsafe_allow_html=True,
    )
    evidence_cases = global_evidence_frame()
    st.plotly_chart(global_case_map(evidence_cases), width="stretch", config={"displayModeBar": False})
    st.caption(
        "南澳大利亚和挪威沿岸为随包运行的真实观测回放；美国Salish Sea与全球数据库用于展示可扩展研究接口。"
    )

    real_observations, real_provenance = load_sa_real_case(ROOT / "data")
    norway_observations, norway_provenance = load_norway_real_case(ROOT / "data")
    norway_benchmark = cached_norway_benchmark(norway_observations)
    sa_full_replay = build_sa_replay(
        real_observations, real_observations["sample_date"].min(),
        real_observations["sample_date"].max(),
    )
    norway_full_replay = build_norway_replay(
        norway_observations, norway_observations["sample_date"].min(),
        norway_observations["sample_date"].max(),
    )
    real_card = sa_full_replay["card"]
    norway_card = norway_full_replay["card"]

    case_choice = st.selectbox(
        "选择可运行的真实观测案例",
        ["南澳大利亚 · 2025复杂Karenia事件", "挪威沿岸 · 2006–2019有毒藻监测"],
    )

    if case_choice.startswith("南澳大利亚"):
        st.markdown("#### 南澳大利亚：复杂Karenia藻华现场qPCR回放")
        real_min = real_observations["sample_date"].min().date()
        real_max = real_observations["sample_date"].max().date()
        r1, r2 = st.columns([1.5, 1.0])
        replay_range = r1.date_input(
            "回放时间范围", value=(real_min, real_max), min_value=real_min,
            max_value=real_max, key="sa_replay_range",
        )
        depths = ["全部深度"] + sorted(real_observations["depth"].dropna().unique().tolist())
        replay_depth = r2.selectbox("采样深度", depths, key="sa_depth")
        if isinstance(replay_range, tuple) and len(replay_range) == 2:
            replay_start, replay_end = replay_range
        else:
            replay_start, replay_end = real_min, real_max
        replay = build_sa_replay(real_observations, replay_start, replay_end, replay_depth)
        selected_card = replay["card"]
        peak = selected_card["peak_k_cristata"]
        st.markdown("##### 养殖对象与暴露情景")
        ra1, ra2 = st.columns(2)
        real_production = ra1.selectbox(
            "主要养殖对象", list(PRODUCTION_PROFILES), key="real_production",
            help="对象选择只改变脆弱性与复核内容，不改变真实qPCR观测。",
        )
        real_exposure = ra2.slider(
            "养殖暴露情景（演示）", .25, 1.0, .80, .05, key="real_exposure",
            help="当前尚未接入真实养殖场坐标和养殖密度。该系数仅演示暴露变化如何影响复核顺序。",
        )
        sa_translation = build_sa_risk_translation(
            replay["sites"], real_production, real_exposure
        )
        real_aqua = sa_translation["priority"]
        kpi_grid([
            ("现场qPCR样本", f"{selected_card['observations']:,}", "窗口内真实采样记录"),
            ("采样日期", f"{selected_card['sampling_dates']}", "非均匀现场采样"),
            ("监测地点", f"{selected_card['locations']}", "Gulf St Vincent沿岸"),
            ("K. cristata检出", f"{selected_card['k_cristata_detection_share']:.1%}", "样本检出比例"),
            ("最高观测丰度", f"{peak['cells_l']:.2e}", "cells L⁻¹"),
            ("回放状态", "现场数据", "CC BY 4.0开放数据"),
        ])
        risk_translation_panel(sa_translation["summary"], sa_translation["evidence"])
        st.markdown(
            '<div class="signal">最高现场观测：'
            f'<b>{peak["location"]}</b> · {peak["date"]} · '
            f'<i>K. cristata</i> <b>{peak["cells_l"]:,.0f} cells L⁻¹</b>。'
            '该值是采样点峰值，不代表整个海区的连续最大值。</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            real_qpcr_map(replay["sites"]), width="stretch", config={"displayModeBar": False}
        )
        rt1, rt2 = st.columns([1.35, 1.0], gap="large")
        with rt1:
            timeline_chart = px.line(
                replay["timeline"], x="sample_date", y="k_cristata_peak_cells_l",
                markers=True, log_y=True, title="K. cristata采样日峰值",
                labels={"sample_date": "采样日期", "k_cristata_peak_cells_l": "cells L⁻¹（对数轴）"},
            )
            timeline_chart.update_layout(height=350, margin={"l": 5, "r": 5, "t": 50, "b": 5})
            st.plotly_chart(timeline_chart, width="stretch", config={"displayModeBar": False})
        with rt2:
            composition_chart = px.bar(
                replay["species"], x="species", y="summed_cells_l_across_samples",
                log_y=True, color="species", title="采样集Karenia物种构成",
                labels={"species": "物种", "summed_cells_l_across_samples": "跨样本丰度和"},
            )
            composition_chart.update_layout(
                showlegend=False, height=350, margin={"l": 5, "r": 5, "t": 50, "b": 5}
            )
            st.plotly_chart(composition_chart, width="stretch", config={"displayModeBar": False})

        st.markdown("#### 监测区域排序")
        st.caption(
            "排序使用真实qPCR相对丰度，但养殖暴露仍为演示情景；优先级不是死亡率、经济损失率或停采阈值。"
        )
        st.dataframe(
            real_aqua[[
                "location", "k_cristata_peak_cells_l", "observed_abundance_band",
                "verification_priority_index", "priority_level", "evidence_grade",
                "recommended_action",
            ]].head(15), width="stretch", hide_index=True,
            column_config={
                "location": "监测地点",
                "k_cristata_peak_cells_l": st.column_config.NumberColumn(
                    "K. cristata峰值（cells L⁻¹）", format="%.0f"
                ),
                "observed_abundance_band": "现场丰度分档",
                "verification_priority_index": st.column_config.ProgressColumn(
                    "复核优先指数", min_value=0, max_value=100, format="%.1f"
                ),
                "priority_level": "建议响应",
                "evidence_grade": "证据等级",
                "recommended_action": st.column_config.TextColumn("建议行动", width="large"),
            },
        )
        rd1, rd2, rd3 = st.columns(3)
        rd1.download_button(
            "下载真实qPCR数据", replay["observations"].to_csv(index=False).encode("utf-8-sig"),
            "south_australia_qpcr_replay.csv", "text/csv",
        )
        rd2.download_button(
            "下载事件回放卡", json.dumps(selected_card, ensure_ascii=False, indent=2).encode("utf-8"),
            "south_australia_replay_card.json", "application/json",
        )
        rd3.download_button(
            "下载养殖复核顺序", real_aqua.to_csv(index=False).encode("utf-8-sig"),
            "south_australia_aquaculture_priority.csv", "text/csv",
        )
        st.markdown(
            "来源：[Nature Ecology & Evolution](https://doi.org/10.1038/s41559-026-03115-0) · "
            "[Zenodo数据（CC BY 4.0）](https://doi.org/10.5281/zenodo.20227730)"
        )
    else:
        st.markdown("#### 挪威沿岸：14年有毒藻与环境监测回放")
        norway_min = norway_observations["sample_date"].min().date()
        norway_max = norway_observations["sample_date"].max().date()
        n1, n2 = st.columns([1.5, 1.0])
        norway_range = n1.date_input(
            "监测时间范围", value=(norway_min, norway_max), min_value=norway_min,
            max_value=norway_max, key="norway_replay_range",
        )
        norway_regions = ["全部站点"] + sorted(norway_observations["region"].unique().tolist())
        norway_region = n2.selectbox("沿岸监测区域", norway_regions, key="norway_region")
        if isinstance(norway_range, tuple) and len(norway_range) == 2:
            norway_start, norway_end = norway_range
        else:
            norway_start, norway_end = norway_min, norway_max
        norway_replay = build_norway_replay(
            norway_observations, norway_start, norway_end, norway_region
        )
        selected_card = norway_replay["card"]
        peak_d = selected_card["peak_d_acuta"]
        st.markdown("##### 养殖对象与暴露情景")
        nr1, nr2 = st.columns(2)
        norway_production = nr1.selectbox(
            "主要养殖对象", list(PRODUCTION_PROFILES), key="norway_production",
            help="对象选择只改变脆弱性与复核内容，不改变真实监测计数。",
        )
        norway_exposure = nr2.slider(
            "养殖暴露情景（演示）", .25, 1.0, .75, .05, key="norway_exposure",
            help="当前尚未接入真实养殖场坐标和养殖密度。该系数仅演示暴露变化如何影响监测顺序。",
        )
        norway_translation = build_norway_risk_translation(
            norway_replay["stations"], norway_production, norway_exposure
        )
        norway_aqua = norway_translation["priority"]
        kpi_grid([
            ("真实监测记录", f"{selected_card['observations']:,}", "藻细胞计数与环境条件"),
            ("采样日期", f"{selected_card['sampling_dates']:,}", "2006–2019周尺度监测"),
            ("沿岸区域", f"{selected_card['regions']}", "58–71°N监测网络"),
            ("研究定义事件", f"{selected_card['target_event_observations']}", ">200 cells L⁻¹记录"),
            ("D. acuta最高观测", f"{peak_d['cells_l']:,.0f}", "cells L⁻¹"),
            ("回放状态", "现场数据", "CC BY 4.0开放数据"),
        ])
        risk_translation_panel(
            norway_translation["summary"], norway_translation["evidence"]
        )
        st.markdown(
            '<div class="signal">窗口内 <i>D. acuta</i> 最高观测：'
            f'<b>{peak_d["region"]}</b> · {peak_d["date"]} · '
            f'<b>{peak_d["cells_l"]:,.0f} cells L⁻¹</b>。'
            '事件标识复现论文研究定义，不替代地方贝类毒素管控规则。</div>',
            unsafe_allow_html=True,
        )
        nt1, nt2 = st.columns([1.35, 1.0], gap="large")
        with nt1:
            norway_time = norway_replay["timeline"].copy()
            norway_time["year"] = norway_time["sample_date"].dt.year
            annual = norway_time.groupby("year", as_index=False).agg(
                event_observations=("target_hab_events", "sum"),
                monitored_dates=("sample_date", "nunique"),
            )
            annual_chart = px.bar(
                annual, x="year", y="event_observations",
                title="研究定义事件的年度观测数",
                labels={"year": "年份", "event_observations": "事件观测数"},
                color_discrete_sequence=["#0b7c78"],
            )
            annual_chart.update_layout(height=370, margin={"l": 5, "r": 5, "t": 50, "b": 5})
            st.plotly_chart(annual_chart, width="stretch", config={"displayModeBar": False})
        with nt2:
            top_stations = norway_replay["stations"].head(15).sort_values("event_observations")
            station_chart = px.bar(
                top_stations, x="event_observations", y="region", orientation="h",
                title="需要优先关注的监测区域",
                labels={"event_observations": "事件观测数", "region": "区域"},
                color="event_share", color_continuous_scale=["#d9efeb", "#0b7c78"],
            )
            station_chart.update_layout(
                height=370, margin={"l": 5, "r": 5, "t": 50, "b": 5},
                coloraxis_colorbar={"title": "事件占比"},
            )
            st.plotly_chart(station_chart, width="stretch", config={"displayModeBar": False})
        st.markdown("#### 观测物种与环境条件")
        ne1, ne2 = st.columns([.9, 1.35], gap="large")
        with ne1:
            st.dataframe(norway_replay["taxa"], width="stretch", hide_index=True)
        with ne2:
            environmental = norway_replay["observations"][[
                "sst_c", "sea_surface_salinity_psu", "mixed_layer_depth_m", "par_e_m2_d"
            ]].describe().loc[["mean", "std", "min", "50%", "max"]].T.reset_index()
            environmental.columns = ["环境变量", "平均", "标准差", "最小", "中位", "最大"]
            st.dataframe(environmental, width="stretch", hide_index=True)

        st.markdown("#### 真实数据前向回顾基准：环境状态能否提前排出下一次监测优先级？")
        benchmark_summary = norway_benchmark["summary"]
        st.markdown(
            '<div class="signal"><b>任务定义：</b>仅使用当前采样时已经可见的SST、PAR、混合层深度、'
            '盐度、季节和区域，排序同一区域未来1–14天内的“下一次实际采样”是否达到论文事件定义。'
            '当前藻细胞计数、未来环境值和超过14天的采样空档均不进入模型。</div>',
            unsafe_allow_html=True,
        )
        kpi_grid([
            ("最高风险10% · 事件覆盖", f"{benchmark_summary['top10_recall']:.1%}",
             f"命中 {benchmark_summary['top10_true_positives']}/{benchmark_summary['events']} 个留出事件"),
            ("Top10% · 命中率", f"{benchmark_summary['top10_precision']:.1%}",
             f"事件率 {benchmark_summary['event_rate']:.2%} 的 {benchmark_summary['top10_precision_lift']:.1f} 倍"),
            ("Average Precision", f"{benchmark_summary['model_average_precision']:.3f}",
             "严格前向留出；越高越好"),
            ("v3.6参考模型", f"{benchmark_summary['reference_average_precision']:.3f}",
             f"v3.7 AP 提升 {benchmark_summary['relative_improvement_over_reference']:.1%}"),
            ("季节基线 AP", f"{benchmark_summary['seasonal_average_precision']:.3f}",
             "仅使用训练期月份概率"),
            ("留出事件率", f"{benchmark_summary['event_rate']:.2%}",
             f"{benchmark_summary['events']} / {benchmark_summary['samples']:,} 个样本"),
        ])
        st.markdown(
            f'<div class="signal"><b>把指标翻译成监测容量：</b>若只复核模型排序最高的10%样本，'
            f'需要检查 <b>{benchmark_summary["top10_selected"]}</b> 个样本，命中 '
            f'<b>{benchmark_summary["top10_true_positives"]}</b> 个事件，同时包含 '
            f'<b>{benchmark_summary["top10_false_positives"]}</b> 个非事件。'
            '这可用于比较加密监测顺序，但误报代价仍高，不能直接作为养殖场报警。</div>',
            unsafe_allow_html=True,
        )
        rb1, rb2 = st.columns([1.15, 1.0], gap="large")
        with rb1:
            fold_chart_data = norway_benchmark["folds"].melt(
                id_vars=["test_window", "test_events"],
                value_vars=["model_average_precision", "reference_average_precision",
                            "seasonal_average_precision"],
                var_name="method", value_name="average_precision",
            )
            fold_chart_data["method"] = fold_chart_data["method"].map({
                "model_average_precision": "v3.7内层选择模型",
                "reference_average_precision": "v3.6参考模型",
                "seasonal_average_precision": "季节基线",
            })
            fold_chart = px.bar(
                fold_chart_data, x="test_window", y="average_precision", color="method",
                barmode="group", title="四个前向时间窗：改进是否稳定？",
                labels={"test_window": "测试年份", "average_precision": "Average Precision（AP）", "method": "方法"},
                color_discrete_map={"v3.7内层选择模型": "#0b7c78",
                                    "v3.6参考模型": "#d9a441", "季节基线": "#b8c8cc"},
                hover_data={"test_events": True},
            )
            fold_chart.update_layout(height=360, margin={"l": 5, "r": 5, "t": 55, "b": 5})
            st.plotly_chart(fold_chart, width="stretch", config={"displayModeBar": False})
        with rb2:
            st.markdown("##### 结果如何解读")
            st.markdown(
                f"""
                - v3.7使用生态交互、稀有事件权重和时间衰减候选，但只在训练期内部选择；
                  外层测试窗从不参与调参。
                - 合并前向AP为 **{benchmark_summary['model_average_precision']:.3f}**，95%自助区间为
                  **{benchmark_summary['model_average_precision_ci95'][0]:.3f}–{benchmark_summary['model_average_precision_ci95'][1]:.3f}**；
                  相比v3.6参考模型提升 **{benchmark_summary['relative_improvement_over_reference']:.1%}**。
                - 模型在 **{benchmark_summary['folds_beating_seasonal_average_precision']}/{benchmark_summary['valid_folds']}**
                  个时间窗高于季节基线；标签置换参照 *p* = **{benchmark_summary['permutation_p']:.3f}**。
                - Brier误差为 **{benchmark_summary['model_brier']:.4f}**，季节基线为
                  **{benchmark_summary['seasonal_brier']:.4f}**。
                - 最弱时间窗是 **{benchmark_summary['weakest_fold']}**（AP
                  **{benchmark_summary['weakest_fold_average_precision']:.3f}**），说明跨时期稳定性仍不足。
                - AP绝对值仍属初步研究信号；这是回顾性“下一次已观测样本”排序，不是连续14天业务预报。
                """
            )
            with st.expander("检查四项防泄漏规则"):
                for rule in benchmark_summary["leakage_controls"]:
                    st.markdown(f"- {rule}")
        rbd1, rbd2, rbd3 = st.columns(3)
        rbd1.download_button(
            "下载前向预测明细",
            norway_benchmark["predictions"].to_csv(index=False).encode("utf-8-sig"),
            "norway_forward_benchmark_predictions.csv", "text/csv",
        )
        rbd2.download_button(
            "下载时间窗指标",
            norway_benchmark["folds"].to_csv(index=False).encode("utf-8-sig"),
            "norway_forward_benchmark_folds.csv", "text/csv",
        )
        rbd3.download_button(
            "下载基准审计卡",
            json.dumps(benchmark_summary, ensure_ascii=False, indent=2).encode("utf-8"),
            "norway_forward_benchmark_card.json", "application/json",
        )
        st.markdown("#### 监测区域排序")
        st.caption(
            "相对危害指数仅比较当前回放窗口内各区域的对数丰度，不是地方毒素阈值；养殖暴露仍为演示情景。"
        )
        st.dataframe(
            norway_aqua[[
                "region", "target_peak_cells_l", "event_observations",
                "verification_priority_index", "priority_level", "evidence_grade",
                "recommended_action",
            ]].head(15),
            width="stretch",
            hide_index=True,
            column_config={
                "region": "监测区域",
                "target_peak_cells_l": st.column_config.NumberColumn(
                    "目标藻最高计数（cells L⁻¹）", format="%.0f"
                ),
                "event_observations": "论文定义事件观测数",
                "verification_priority_index": st.column_config.ProgressColumn(
                    "加密监测优先指数", min_value=0, max_value=100, format="%.1f"
                ),
                "priority_level": "建议响应",
                "evidence_grade": "证据等级",
                "recommended_action": st.column_config.TextColumn("建议行动", width="large"),
            },
        )
        nd1, nd2, nd3 = st.columns(3)
        nd1.download_button(
            "下载挪威观测回放", norway_replay["observations"].to_csv(index=False).encode("utf-8-sig"),
            "norway_hab_monitoring_replay.csv", "text/csv",
        )
        nd2.download_button(
            "下载挪威回放卡", json.dumps(selected_card, ensure_ascii=False, indent=2).encode("utf-8"),
            "norway_replay_card.json", "application/json",
        )
        nd3.download_button(
            "下载养殖监测顺序", norway_aqua.to_csv(index=False).encode("utf-8-sig"),
            "norway_aquaculture_monitoring_priority.csv", "text/csv",
        )
        st.markdown(
            "来源：[Communications Earth & Environment](https://doi.org/10.1038/s43247-025-02421-y) · "
            "[Zenodo数据与模型（CC BY 4.0）](https://doi.org/10.5281/zenodo.10958487)"
        )

with tab_bio:
    st.markdown("### 网箱鱼生物响应沙盘")
    st.markdown(
        '<div class="signal"><b>网箱鱼响应情景：</b>本沙盘将藻华、高温、溶解氧、'
        '养殖密度和计划投喂转化为网箱鱼的相对生理压力轨迹，并并列比较监测、降低投喂、增氧和'
        '转移准备。参数用于情景比较，尚未按具体鱼种或场站进行标定。</div>',
        unsafe_allow_html=True,
    )

    bio_control, bio_result = st.columns([1.0, 2.15], gap="large")
    with bio_control:
        bio_preset_name = st.selectbox(
            "选择生物响应情景", list(BIO_SCENARIO_PRESETS), index=0
        )
        bio_preset = BIO_SCENARIO_PRESETS[bio_preset_name]
        hab_pressure = st.slider(
            "藻华危害压力（0–100）", 0.0, 100.0,
            float(bio_preset["hab_pressure"]), 1.0,
            key=f"bio_hab_{bio_preset_name}",
            help="无量纲外部压力。真实事件锚点只表示所选回放内的相对峰值，不是毒素或死亡阈值。",
        )
        bio_mhw = st.slider(
            "海洋热浪强度（°C）", 0.0, 5.0,
            float(bio_preset["mhw_intensity_c"]), .1,
            key=f"bio_mhw_{bio_preset_name}",
        )
        bio_do = st.slider(
            "场景溶解氧（mg L⁻¹）", 1.0, 10.0,
            float(bio_preset["dissolved_oxygen_mg_l"]), .1,
            key=f"bio_do_{bio_preset_name}",
            help="情景输入，不代表当前真实养殖场观测。",
        )
        bio_density = st.slider(
            "养殖密度（kg m⁻³）", 2.0, 45.0,
            float(bio_preset["stocking_density_kg_m3"]), 1.0,
            key=f"bio_density_{bio_preset_name}",
            help="用于模拟密度相关氧负荷；不是推荐养殖密度。",
        )
        bio_feed = st.slider(
            "计划投喂水平（%）", 0.0, 120.0,
            float(bio_preset["planned_feeding_pct"]), 5.0,
            key=f"bio_feed_{bio_preset_name}",
        )
        bio_duration = st.slider(
            "藻华压力持续时间（小时）", 12, 72,
            int(bio_preset["hab_duration_hours"]), 6,
            key=f"bio_duration_{bio_preset_name}",
        )
        bio_horizon = st.radio(
            "模拟时间", [48, 72, 96], index=1, horizontal=True,
            format_func=lambda value: f"{value}小时",
        )
        st.caption(str(bio_preset["source_note"]))

    bio_simulation = compare_interventions(
        hab_pressure=hab_pressure,
        mhw_intensity_c=bio_mhw,
        dissolved_oxygen_mg_l=bio_do,
        stocking_density_kg_m3=bio_density,
        planned_feeding_pct=bio_feed,
        hab_duration_hours=min(bio_duration, bio_horizon),
        horizon_hours=bio_horizon,
    )
    bio_robustness = evaluate_intervention_robustness(
        hab_pressure=hab_pressure,
        mhw_intensity_c=bio_mhw,
        dissolved_oxygen_mg_l=bio_do,
        stocking_density_kg_m3=bio_density,
        planned_feeding_pct=bio_feed,
        hab_duration_hours=min(bio_duration, bio_horizon),
        horizon_hours=bio_horizon,
    )
    bio_summary = bio_simulation["summary"]
    bio_trajectories = bio_simulation["trajectories"]
    lowest = bio_summary.iloc[0]
    baseline = bio_summary[bio_summary["intervention"].eq("维持监测")].iloc[0]
    transfer = bio_summary[
        bio_summary["intervention"].eq("转移准备（未执行）")
    ].iloc[0]

    with bio_result:
        if bio_preset_name.startswith("南澳"):
            anchor_text = (
                f"真实藻华锚点：K. cristata现场qPCR峰值 "
                f"{real_card['peak_k_cristata']['cells_l']:,.0f} cells L⁻¹ · "
                f"{real_card['peak_k_cristata']['location']} · "
                f"{real_card['peak_k_cristata']['date']}。仅在回放内部归一化为100。"
            )
        elif bio_preset_name.startswith("挪威"):
            anchor_text = (
                f"真实藻华锚点：D. acuta最高监测值 "
                f"{norway_card['peak_d_acuta']['cells_l']:,.0f} cells L⁻¹ · "
                f"{norway_card['peak_d_acuta']['region']} · "
                f"{norway_card['peak_d_acuta']['date']}。仅在回放内部归一化为100。"
            )
        else:
            anchor_text = "本情景全部输入均为可调整的科研演示值，不代表具体养殖场。"
        st.markdown(
            f'<div class="case-card"><span class="case-badge">数据来源</span><br>'
            f'{html.escape(anchor_text)}</div>', unsafe_allow_html=True,
        )
        kpi_grid([
            ("基准峰值压力", f"{baseline['peak_pressure_index']:.1f}/100", "维持监测情景"),
            ("沙盘最低压力方案", str(lowest["intervention"]), "仅为情景比较，不是自动建议"),
            ("累计压力变化", f"−{lowest['pressure_load_reduction_vs_baseline_pct']:.1f}%", "相对维持监测基准"),
            ("摄食机会保留", f"{lowest['mean_feeding_opportunity_pct']:.1f}%", "模型中的相对摄食代理"),
            ("最低有效DO", f"{lowest['minimum_effective_do_mg_l']:.2f}", "mg L⁻¹ · 场景代理"),
            ("准备响应时间", f"{transfer['response_readiness_hours']}小时", "转移准备但尚未执行"),
        ])

    bio_steps = [
        ("环境条件", f"HAB {hab_pressure:.0f} · MHW {bio_mhw:.1f}°C"),
        ("养殖条件", f"DO {bio_do:.1f} · 密度 {bio_density:.0f} kg m⁻³"),
        ("生理响应", "压力累积、恢复与摄食机会"),
        ("干预方案", "降投喂、增氧、转移准备"),
        ("结果解读", "比较轨迹，不外推死亡率"),
    ]
    step_cards = "".join(
        '<div class="risk-step">'
        f'<div class="risk-step-title">{title}</div>'
        f'<div class="risk-step-value">{value}</div>'
        '</div>'
        for title, value in bio_steps
    )
    st.markdown(
        '<div class="risk-bridge"><div class="risk-bridge-title">网箱鱼响应概览</div>'
        '<div class="risk-bridge-subtitle">模型沿时间推进复合压力，并在相同外部条件下比较干预方案。</div>'
        f'<div class="risk-chain">{step_cards}</div></div>', unsafe_allow_html=True,
    )

    selected_interventions = st.multiselect(
        "选择需要比较的干预轨迹",
        list(INTERVENTIONS),
        default=list(INTERVENTIONS),
    )
    if not selected_interventions:
        selected_interventions = ["维持监测"]
    bio_plot_data = bio_trajectories[
        bio_trajectories["intervention"].isin(selected_interventions)
    ]
    trajectory_chart = px.line(
        bio_plot_data,
        x="hour",
        y="relative_physiological_pressure",
        color="intervention",
        title="不同干预情景下的相对生理压力轨迹",
        labels={
            "hour": "模拟时间（小时）",
            "relative_physiological_pressure": "相对生理压力指数（0–100）",
            "intervention": "干预情景",
        },
    )
    trajectory_chart.update_layout(
        height=440, margin={"l": 5, "r": 5, "t": 55, "b": 5},
        legend={"orientation": "h", "y": -0.20},
        hovermode="x unified",
    )
    trajectory_chart.update_yaxes(range=[0, 100])
    st.plotly_chart(trajectory_chart, width="stretch", config={"displayModeBar": False})

    bc1, bc2 = st.columns([1.25, 1.0], gap="large")
    with bc1:
        error_plus = (
            bio_summary["peak_pressure_upper"] - bio_summary["peak_pressure_index"]
        )
        error_minus = (
            bio_summary["peak_pressure_index"] - bio_summary["peak_pressure_lower"]
        )
        pressure_bar = go.Figure(go.Bar(
            x=bio_summary["intervention"],
            y=bio_summary["peak_pressure_index"],
            marker_color=["#0b7c78", "#267c9c", "#d08a32", "#6a4c93", "#78939b"],
            error_y={
                "type": "data", "array": error_plus, "arrayminus": error_minus,
                "visible": True, "color": "#3f5860",
            },
            customdata=bio_summary[[
                "mean_feeding_opportunity_pct", "minimum_effective_do_mg_l",
                "pressure_load_reduction_vs_baseline_pct",
            ]],
            hovertemplate=(
                "<b>%{x}</b><br>峰值压力：%{y:.1f}/100"
                "<br>摄食机会：%{customdata[0]:.1f}%"
                "<br>最低有效DO：%{customdata[1]:.2f} mg L⁻¹"
                "<br>累计压力变化：%{customdata[2]:.1f}%<extra></extra>"
            ),
        ))
        pressure_bar.update_layout(
            title="峰值压力与±15%参数敏感性包络", height=390,
            margin={"l": 5, "r": 5, "t": 55, "b": 95},
            xaxis={"tickangle": -18}, yaxis={"title": "相对压力指数", "range": [0, 100]},
        )
        st.plotly_chart(pressure_bar, width="stretch", config={"displayModeBar": False})
    with bc2:
        tradeoff_chart = px.scatter(
            bio_summary,
            x="mean_feeding_opportunity_pct",
            y="pressure_load_reduction_vs_baseline_pct",
            color="intervention",
            size=(20 - bio_summary["response_readiness_hours"]).clip(lower=2),
            title="压力缓解—摄食机会权衡",
            labels={
                "mean_feeding_opportunity_pct": "摄食机会保留（%）",
                "pressure_load_reduction_vs_baseline_pct": "累计压力降低（%）",
                "intervention": "干预情景",
            },
            hover_data={"response_readiness_hours": True},
        )
        tradeoff_chart.update_layout(
            height=390, margin={"l": 5, "r": 5, "t": 55, "b": 5},
            showlegend=False,
        )
        st.plotly_chart(tradeoff_chart, width="stretch", config={"displayModeBar": False})

    st.markdown("#### 参数扰动下的稳定性")
    robustness_summary = bio_robustness["summary"]
    robustness_card = bio_robustness["card"]
    st.markdown(
        '<div class="signal"><b>邻域压力测试：</b>系统自动组合HAB压力±10%、MHW±0.4°C、'
        'DO±0.5 mg L⁻¹和密度±10%，形成81个邻近情景。报告帕累托出现率，'
        '而不是强行给出一个“唯一最佳操作”。</div>',
        unsafe_allow_html=True,
    )
    robustness_chart = px.scatter(
        robustness_summary,
        x="median_feeding_opportunity_pct",
        y="median_pressure_reduction_pct",
        size="pareto_frequency",
        color="intervention",
        title="81个邻近输入情景下的干预稳健性",
        labels={
            "median_feeding_opportunity_pct": "中位摄食机会保留（%）",
            "median_pressure_reduction_pct": "中位累计压力降低（%）",
            "pareto_frequency": "帕累托出现率（%）",
            "intervention": "干预情景",
        },
        hover_data={
            "worst_case_pressure_reduction_pct": ":.1f",
            "best_case_pressure_reduction_pct": ":.1f",
            "lowest_pressure_frequency": ":.1f",
        },
    )
    robustness_chart.update_layout(
        height=410, margin={"l": 5, "r": 5, "t": 55, "b": 5}, showlegend=False,
    )
    st.plotly_chart(robustness_chart, width="stretch", config={"displayModeBar": False})
    st.dataframe(
        robustness_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "intervention": "干预情景",
            "scenarios": "扰动情景数",
            "median_pressure_reduction_pct": st.column_config.NumberColumn(
                "中位压力降低（%）", format="%.1f"
            ),
            "worst_case_pressure_reduction_pct": st.column_config.NumberColumn(
                "最弱情景变化（%）", format="%.1f"
            ),
            "best_case_pressure_reduction_pct": st.column_config.NumberColumn(
                "最强情景变化（%）", format="%.1f"
            ),
            "median_feeding_opportunity_pct": st.column_config.NumberColumn(
                "中位摄食机会（%）", format="%.1f"
            ),
            "lowest_pressure_frequency": st.column_config.NumberColumn(
                "最低压力出现率（%）", format="%.1f"
            ),
            "pareto_frequency": st.column_config.ProgressColumn(
                "帕累托出现率（%）", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )
    st.caption(
        "帕累托出现率高表示该方案在“降低相对压力”和“保留摄食机会”两个目标上较少被其他方案同时压过；"
        "它不是现场有效率或推荐概率。"
    )

    st.markdown("#### 干预情景对照表")
    bio_display = bio_summary[[
        "intervention", "peak_pressure_index", "peak_pressure_lower",
        "peak_pressure_upper", "pressure_load_reduction_vs_baseline_pct",
        "mean_feeding_opportunity_pct", "minimum_effective_do_mg_l",
        "response_readiness_hours", "pressure_band", "scenario_interpretation",
    ]].copy()
    st.dataframe(
        bio_display, width="stretch", hide_index=True,
        column_config={
            "intervention": "干预情景",
            "peak_pressure_index": st.column_config.ProgressColumn(
                "峰值压力", min_value=0, max_value=100, format="%.1f"
            ),
            "peak_pressure_lower": st.column_config.NumberColumn("敏感性下界", format="%.1f"),
            "peak_pressure_upper": st.column_config.NumberColumn("敏感性上界", format="%.1f"),
            "pressure_load_reduction_vs_baseline_pct": st.column_config.NumberColumn(
                "累计压力变化（%）", format="%.1f"
            ),
            "mean_feeding_opportunity_pct": st.column_config.NumberColumn(
                "摄食机会保留（%）", format="%.1f"
            ),
            "minimum_effective_do_mg_l": st.column_config.NumberColumn(
                "最低有效DO", format="%.2f"
            ),
            "response_readiness_hours": "准备响应时间（h）",
            "pressure_band": "展示分档",
            "scenario_interpretation": st.column_config.TextColumn("情景解释", width="large"),
        },
    )
    st.info(
        "“转移准备（未执行）”与“维持监测”的生理轨迹相同是预期结果：准备本身只缩短响应时间，"
        "在实际转移发生前不会减少藻华暴露。"
    )

    with st.expander("模型设定、参数与输入说明"):
        st.markdown(
            "综合挑战由HAB、热异常、低氧、密度及三个交互项加权组成。压力状态按小时更新："
        )
        st.latex(
            r"P_{t+1}=\operatorname{clip}\left[P_t+1.45C_t(1-P_t/100)"
            r"-0.55(1-C_t)P_t/100,\ 0,\ 100\right]"
        )
        st.caption(
            "Cₜ为0–1综合挑战，Pₜ为0–100相对生理压力。平滑中心和权重均在下表列出，"
            "但尚未通过具体鱼种的死亡、生长或代谢数据进行现场标定。"
        )
        evidence_rows = [
            ["藻华危害压力", "真实回放相对峰值" if "真实" in bio_preset_name else "科研情景", "危害外部输入"],
            ["MHW强度", "情景假设", "热压力输入"],
            ["溶解氧", "情景假设", "低氧压力输入"],
            ["养殖密度", "情景假设", "密度与耗氧代理"],
            ["计划投喂", "情景假设", "摄食与代谢负荷代理"],
            ["生物响应参数", "未标定参数", "需用物种/场站数据再标定"],
        ]
        evidence_frame = pd.DataFrame(
            evidence_rows, columns=["模型输入", "证据属性", "在沙盘中的作用"]
        )
        st.dataframe(evidence_frame, width="stretch", hide_index=True)
        st.dataframe(bio_simulation["parameters"], width="stretch", hide_index=True)

    bd1, bd2, bd3, bd4 = st.columns(4)
    bd1.download_button(
        "下载响应轨迹", bio_trajectories.to_csv(index=False).encode("utf-8-sig"),
        "cage_fish_response_trajectories.csv", "text/csv",
    )
    bd2.download_button(
        "下载干预对照", bio_summary.to_csv(index=False).encode("utf-8-sig"),
        "cage_fish_intervention_comparison.csv", "text/csv",
    )
    bd3.download_button(
        "下载模型参数", bio_simulation["parameters"].to_csv(index=False).encode("utf-8-sig"),
        "cage_fish_sandbox_parameters.csv", "text/csv",
    )
    bd4.download_button(
        "下载沙盘卡", json.dumps(
            bio_simulation["scenario_card"], ensure_ascii=False, indent=2
        ).encode("utf-8"),
        "cage_fish_sandbox_card.json", "application/json",
    )
    st.download_button(
        "下载81情景稳健性结果",
        bio_robustness["detail"].to_csv(index=False).encode("utf-8-sig"),
        "cage_fish_intervention_robustness.csv", "text/csv",
    )
    st.warning(
        "能力边界：本模块不输出死亡率、真实生物量损失、毒素浓度或养殖场级预测；"
        "降低投喂、增氧和转移等措施必须由养殖人员结合真实DO、鱼群行为、鳃部状态、设备能力和监管要求决定。"
    )
    st.caption(
        "架构参考：[Føre等，Computers and Electronics in Agriculture（2024）]"
        "(https://doi.org/10.1016/j.compag.2024.108676)与"
        "[Lima等，Open Research Europe（2023）]"
        "(https://open-research-europe.ec.europa.eu/articles/2-16)。"
        "文献用于支持“环境—生物状态—运营对照”的结构设计，不构成当前参数的鱼种标定。"
    )

with tab_methods:
    st.markdown("### 异常识别与跨区域影响")
    st.caption("以下分析共享同一套时空数据、时间窗口和验证设置。")

    st.markdown("#### 持续异常识别")
    kpi_grid([
        ("观测时间尺度", "7 / 14 / 30 / 60天", "同时识别短期冲击与持续变化"),
        ("合并异常事件", f"{len(anomaly_events)}", "相邻异常自动合并为事件"),
        ("未来信息泄漏", "无", "所有参照仅使用当日以前数据"),
        ("稳健性判定", "≥2个尺度一致", "降低单一窗口误报"),
        ("异常强度", "MAD标准化", "抵抗极端离群值干扰"),
        ("输出形式", "事件目录", "可下载、可追踪、可复核"),
    ])
    anomaly_region = st.selectbox(
        "选择异常轨迹海域", REGIONS, index=3, key="anomaly_region",
        format_func=lambda value: REGION_LABELS[value],
    )
    anomaly_plot = anomaly_daily[anomaly_daily["region"].eq(anomaly_region)].set_index("date")[[
        "anomaly_score_7d", "anomaly_score_14d", "anomaly_score_30d",
        "anomaly_score_60d", "multiscale_anomaly_score",
    ]]
    st.line_chart(anomaly_plot, height=300)
    anomaly_events_display = anomaly_events.head(12).copy()
    anomaly_events_display["region"] = anomaly_events_display["region"].map(REGION_LABELS)
    st.dataframe(
        anomaly_events_display, width="stretch", hide_index=True,
        column_config={
            "region": "海域", "peak_score": st.column_config.NumberColumn("峰值强度", format="%.3f")
        },
    )
    st.download_button(
        "下载多尺度事件目录", anomaly_events.to_csv(index=False).encode("utf-8-sig"),
        "multiscale_event_catalog.csv", "text/csv",
    )

    st.markdown("#### 分析路径选择")
    route_display = router_trace[[
        "branch", "compatibility_score", "routing_probability", "decision", "reason"
    ]].copy()
    route_display["branch"] = route_display["branch"].map({
        "blocked_prediction": "跨时间与跨区域风险验证",
        "multiscale_anomaly": "多时间尺度异常识别",
        "te_cte_network": "跨区域传播路径识别",
        "spatial_durbin": "邻近海区溢出影响",
    })
    route_display["decision"] = route_display["decision"].map({"run": "运行", "defer": "暂缓"})
    st.dataframe(
        route_display, width="stretch", hide_index=True,
        column_config={
            "compatibility_score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
            "routing_probability": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "两阶段门控先计算完整度、样本支持、时间依赖、空间支持、事件支持和跨尺度一致性，"
        "再对四条分支给出兼容度与软路由概率；每次决策均进入日志。"
    )

    st.markdown("#### 跨区域传播时滞")
    st.caption("比较预设输运方向和反向路径，识别最可能的传播时滞，并对偶然相关进行多重检验控制。")
    te_long = te_cte_lag_summary.melt(
        id_vars="lag_days",
        value_vars=["mean_cte_bits", "mean_reverse_cte_bits"],
        var_name="direction", value_name="information_bits",
    )
    te_long["direction"] = te_long["direction"].map({
        "mean_cte_bits": "预设沿流方向 CTE",
        "mean_reverse_cte_bits": "反向路径 CTE",
    })
    te_chart = px.line(
        te_long, x="lag_days", y="information_bits", color="direction", markers=True,
        labels={"lag_days": "时滞（天）", "information_bits": "条件传递熵（bit）", "direction": "路径"},
        color_discrete_sequence=["#0d7f79", "#b8865b"],
    )
    te_chart.add_vline(x=14, line_dash="dot", line_color="#6a4c93")
    te_chart.update_layout(height=350, margin={"l": 5, "r": 5, "t": 20, "b": 5})
    st.plotly_chart(te_chart, width="stretch", config={"displayModeBar": False})
    peak_lag = int(te_cte_lag_summary.loc[te_cte_lag_summary["mean_cte_bits"].idxmax(), "lag_days"])
    st.success(f"结果指向约 {peak_lag} 天的跨区域传播窗口；反向路径更弱，且显著性经过多重比较校正。")
    te_display = te_cte_network[[
            "source_region", "target_region", "lag_days", "te_bits", "cte_bits",
            "reverse_cte_bits", "net_directionality_bits", "permutation_p", "fdr_q",
            "significant_fdr_0_10",
        ]].head(16).copy()
    te_display["source_region"] = te_display["source_region"].map(REGION_LABELS)
    te_display["target_region"] = te_display["target_region"].map(REGION_LABELS)
    st.dataframe(
        te_display, width="stretch", hide_index=True,
        column_config={
            "source_region": "信号来源区", "target_region": "风险响应区",
            "lag_days": "传播时滞（天）", "cte_bits": "方向信息量",
            "reverse_cte_bits": "反向路径信息量", "fdr_q": "校正后可靠性",
            "significant_fdr_0_10": "通过可靠性检验",
        },
    )
    st.download_button(
        "下载TE/CTE边级网络", te_cte_network.to_csv(index=False).encode("utf-8-sig"),
        "te_cte_network.csv", "text/csv",
    )

    st.markdown("#### 邻近海区影响")
    st.caption("将本地影响、邻近海区溢出和总体关联分开呈现，便于确定需要同步监测的范围。")
    labels = {
        "multiscale_anomaly_score_lag14": "14天滞后多尺度异常",
        "nutrient_context": "营养盐背景",
        "circulation_residence_proxy": "输运/停留/汇聚代理",
    }
    effect_plot = spatial_effects.copy()
    effect_plot["变量"] = effect_plot["variable"].map(labels)
    effect_plot["影响"] = effect_plot["effect_type"].map({
        "direct": "直接", "indirect": "间接", "total": "总影响"
    })
    effect_chart = px.bar(
        effect_plot, x="变量", y="effect_per_1sd", color="影响", barmode="group",
        labels={"effect_per_1sd": "合成HAB概率变化 / 1 SD"},
        color_discrete_sequence=["#357e8a", "#e39b47", "#6a4c93"],
    )
    effect_chart.update_layout(height=390, margin={"l": 5, "r": 5, "t": 20, "b": 5})
    st.plotly_chart(effect_chart, width="stretch", config={"displayModeBar": False})
    kpi_grid([
        ("区域联动强度", f"ρ = {float(spatial_diagnostics['rho']):.2f}", "相邻区域结果的同步程度"),
        ("探索性解释度", f"{float(spatial_diagnostics['pseudo_r2']):.3f}", "用于合成基准的软件验证"),
        ("不确定性复核", f"{int(spatial_diagnostics['bootstrap_repeats'])}次", "按区域块重复抽样"),
        ("本地影响", "Direct", "异常发生区自身变化"),
        ("邻区溢出", "Indirect", "沿网络传播到周边的关联"),
        ("总体关联", "Total", "本地与邻区影响之和"),
    ])
    st.caption(
        "区域传播关系根据预设输运方向标准化；14天窗口来自前述风险模式和传播路径结果。"
        "这里量化的是合成环境中的关联强度，用于验证影响分解流程。"
    )
    visible_effects = effect_plot[[
        "变量", "影响", "effect_per_1sd", "ci90_lower", "ci90_upper"
    ]].copy()
    st.dataframe(
        visible_effects, width="stretch", hide_index=True,
        column_config={
            "effect_per_1sd": "每1 SD关联强度", "ci90_lower": "不确定区间下限",
            "ci90_upper": "不确定区间上限",
        },
    )
    st.download_button(
        "下载Durbin影响分解", spatial_effects.to_csv(index=False).encode("utf-8-sig"),
        "spatial_durbin_effects.csv", "text/csv",
    )

with tab_agent:
    st.markdown("### 实验对照与探索记录")
    c1, c2 = st.columns([1.15, 1.0], gap="large")
    with c1:
        comparison = pd.concat([
            baselines[["baseline", "pr_auc", "brier_skill", "ece"]].rename(columns={"baseline": "方法"}),
            pd.DataFrame([{
                "方法": "GlobalHAB-Agent最佳候选",
                "pr_auc": best["pr_auc"],
                "brier_skill": best["brier_skill"],
                "ece": best["ece"],
            }]),
        ], ignore_index=True)
        fig = px.bar(
            comparison, x="方法", y="pr_auc", color="方法",
            title="阻断验证Average Precision（AP）：Agent候选 vs 平凡解",
            text_auto=".3f",
            color_discrete_sequence=["#8aa6a3", "#c7a76c", "#0d7f79"],
        )
        fig.update_layout(showlegend=False, height=350, margin={"l": 5, "r": 5, "t": 55, "b": 10})
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("#### 随机探索参照")
        kpi_grid([
            ("随机探索恢复14天信号", f"{float(random_ref['hidden_signal_recovery_rate']):.1%}", "相同实验次数下的参照"),
            ("随机探索中位效用", f"{float(random_ref['median_best_utility']):.3f}", "用于判断Agent选择增益"),
            ("重复对照次数", f"{int(random_ref['repeats'])}", "每次使用相同候选空间"),
        ])
        st.caption(
            f"基于{int(random_ref['repeats'])}次随机选择；每次使用相同候选空间、"
            f"相同预算（{budget}步）和相同阻断验证结果。"
        )
        st.markdown("#### 负对照")
        st.dataframe(
            controls[["control_name", "pr_auc", "pr_auc_gain", "brier_skill", "ece"]],
            width="stretch", hide_index=True,
        )

    st.markdown("#### 完整探索轨迹：正结果、负结果和成本均保留")
    display = log[[
        "step", "hypothesis", "action_id", "status", "pr_auc", "pr_auc_gain",
        "brier_skill", "ece", "false_positive_rate_at_top20",
        "compute_cost_units", "budget_remaining",
    ]]
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "下载完整探索日志 CSV",
        log.to_csv(index=False).encode("utf-8-sig"),
        "agent_exploration_log.csv",
        "text/csv",
    )

    st.markdown("#### 留出海区风险序列")
    plot = predictions.set_index("date")[["risk_probability", "hab_event", "top20_alert"]]
    st.line_chart(plot, height=320)
    st.caption("Top20%报警是固定容量排名，不使用留出标签选择阈值。")

with tab_evidence:
    st.markdown("### 数据来源与结果复核")
    st.caption("现场观测用于事件回放，全球数据用于背景校准，公开研究用于补充预警信号。")
    case_columns = st.columns(4)
    for column, case_row in zip(case_columns, evidence_cases.to_dict("records")):
        with column:
            st.markdown(
                '<div class="case-card">'
                f'<span class="case-badge">{case_row["product_status"]}</span><br>'
                f'<b>{case_row["case"]}</b><br>'
                f'<span class="small-muted">{case_row["journal"]}<br>'
                f'{case_row["period"]} · {case_row["records"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'[论文]({case_row["url"]}) · [开放数据]({case_row["data_url"]})')

    st.markdown("### 结论与证据对应关系")
    claim_ledger = pd.DataFrame([
        ["Agent能恢复预设14天沿流信号", "匿名合成真值", "完整留区+前向阻断；随机搜索与时间置换", "软件正确性通过；不代表真实海洋性能"],
        ["真实环境状态包含有限但可复核的下一次监测排序信号", "挪威2006–2019开放监测", "训练期内层选择；四个前向窗；AP区间；容量/误报分解", "AP绝对值与弱窗限制明显；不是业务报警"],
        ["南澳事件危害证据可定位到时间、地点和藻种", "115条现场qPCR", "原始工作簿、派生表、哈希与事件卡", "采样峰值；不代表全海域连续最大值"],
        ["真实危害可以转为现场复核顺序", "真实丰度+显式暴露/脆弱性假设", "证据矩阵逐项标注观测、假设和缺口", "不是损失概率、毒素阈值或监管指令"],
        ["网箱鱼干预存在压力—摄食权衡", "公开方程与参数设定", "±15%参数包络+81个邻近输入情景", "尚无物种/场站标定，不预测死亡率"],
        ["传播方向与邻区影响可被分解", "合成基准", "TE/CTE置换、BH-FDR与Durbin效应分解", "关联与传播线索，不宣称因果"],
    ], columns=["可公开结论", "证据来源", "检查方式", "不能据此宣称"])
    st.dataframe(
        claim_ledger, width="stretch", hide_index=True,
        column_config={
            "可公开结论": st.column_config.TextColumn("可公开结论", width="large"),
            "证据来源": st.column_config.TextColumn("证据来源", width="medium"),
            "检查方式": st.column_config.TextColumn("检查方式", width="large"),
            "不能据此宣称": st.column_config.TextColumn("不能据此宣称", width="large"),
        },
    )

    st.markdown("### 环境信息如何转化为可读风险线索")
    st.markdown(
        """
        <div class="product-grid">
          <div class="product-card"><b>海温异常识别</b><p>基于逐日季节气候态和高温阈值，
          量化海表温度偏离正常背景的幅度、持续时间与累积影响。</p></div>
          <div class="product-card"><b>营养环境画像</b><p>分别保留硝酸盐、磷酸盐和硅酸盐信息，
          描述不同营养条件对藻华形成与物种竞争的支持背景。</p></div>
          <div class="product-card"><b>输运与汇聚线索</b><p>利用微塑料浓度的有界代理表达水团停留、
          汇聚与输运背景，用于判断风险信号可能向哪里传播。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 下载结果与复核材料")
    card_bytes = json.dumps(card, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button("下载结果摘要 JSON", card_bytes, "discovery_card.json", "application/json")
    evidence_bundle = {
        "discovery_card": card,
        "south_australia_external_evidence": SOUTH_AUSTRALIA_CASE,
        "baselines": baselines.to_dict("records"),
        "negative_controls": controls.to_dict("records"),
        "adaptive_router": router_trace.to_dict("records"),
        "te_cte_lag_summary": te_cte_lag_summary.to_dict("records"),
        "spatial_durbin_effects": spatial_effects.to_dict("records"),
        "south_australia_real_replay": real_card,
        "south_australia_real_data_provenance": real_provenance,
        "norway_real_replay": norway_card,
        "norway_real_data_provenance": norway_provenance,
        "norway_forward_benchmark": norway_benchmark["summary"],
        "norway_forward_benchmark_folds": norway_benchmark["folds"].to_dict("records"),
        "cage_fish_robustness": robustness_card,
        "claim_ledger": claim_ledger.to_dict("records"),
        "global_nature_portfolio_evidence": evidence_cases.to_dict("records"),
    }
    d2.download_button(
        "下载证据包 JSON",
        json.dumps(evidence_bundle, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        "semifinal_evidence_bundle.json",
        "application/json",
    )
    with st.expander("查看结构化结果"):
        st.json(card)

st.divider()
st.markdown(
    """
    <div class="footer-boundary">
      <b>能力边界：</b>合成基准用于验证信号恢复、跨区域评估和机制模块的软件正确性；
      南澳大利亚使用真实qPCR作事件回放；挪威长期监测另设严格前向的回顾性下一样本基准，
      均不与合成数据混合。
      生物响应沙盘采用公开文献支持的参数结构，尚未经物种/场站标定；风险地图、复核顺序和干预对照
      不构成死亡率或损失预测、业务预报、因果结论、统一毒素阈值或自动运营指令。
      <br>GlobalHAB-Agent v3.7.1 GOAI Semifinal · synthetic recovery + nested real forward benchmark + event replay + biological-response sandbox ·
      no mortality, operational, causal or automatic action claim
    </div>
    """,
    unsafe_allow_html=True,
)
