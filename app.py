"""GlobalHAB-Agent GOAI semifinal interactive demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo import (  # noqa: E402
    EVIDENCE_CONFIDENCE,
    MECHANISM_LABELS,
    PRODUCTION_PROFILES,
    SCENARIO_PRESETS,
    SOUTH_AUSTRALIA_CASE,
    build_sa_replay,
    load_sa_real_case,
    project_aquaculture_risk,
    project_real_aquaculture_priority,
    project_synthetic_scenario,
    real_data_router,
    run_exploration,
)
from globalhab_demo.data import REGIONS  # noqa: E402


st.set_page_config(
    page_title="GlobalHAB-Agent | GOAI复赛",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1480px;}
    h1, h2, h3 {letter-spacing:-0.02em;}
    [data-testid="stMetric"] {
        background:linear-gradient(135deg,#f7fbfc,#eef7f5);
        border:1px solid #cfe1df; border-radius:14px; padding:.75rem .85rem;
        box-shadow:0 5px 18px rgba(22,74,78,.05);
    }
    .eyebrow {font-size:.78rem; font-weight:700; letter-spacing:.12em;
        color:#147a7e; text-transform:uppercase; margin-bottom:.2rem;}
    .hero {background:linear-gradient(120deg,#073b4c 0%,#0b6670 58%,#138a83 100%);
        color:white; padding:1.15rem 1.35rem; border-radius:18px; margin-bottom:1rem;
        box-shadow:0 12px 32px rgba(4,52,63,.18);}
    .hero h1 {font-size:2.05rem; margin:.05rem 0 .25rem; color:white;}
    .hero p {margin:0; color:#d9f2ef; font-size:1rem;}
    .status-row {display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.75rem;}
    .pill {display:inline-block; padding:.25rem .62rem; border-radius:999px;
        background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.24);
        font-size:.78rem; color:white;}
    .boundary {background:#fff8e7; border:1px solid #f0d6a0; color:#5c4217;
        padding:.72rem .9rem; border-radius:10px; margin:.45rem 0 1rem;}
    .signal {background:linear-gradient(105deg,#eaf8f4,#f4fbfd);
        border:1px solid #bfe1d8; border-radius:12px; padding:.82rem 1rem;
        color:#123f45; margin-bottom:.5rem;}
    .case-card {background:#f7f9fc; border-left:4px solid #6a4c93;
        padding:.9rem 1.05rem; border-radius:10px; margin:.5rem 0;}
    .formula {background:#082f3a; color:#e8fbf7; padding:.8rem 1rem;
        border-radius:10px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
    .small-muted {color:#61777d; font-size:.84rem;}
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
      <div class="eyebrow" style="color:#a7eee2">GOAI · AI for Research · Open Exploration</div>
      <h1>GlobalHAB-Agent</h1>
      <p>合成真值恢复、机制分解与南澳真实qPCR事件回放的可检查探索闭环</p>
      <div class="status-row">
        <span class="pill">前向时间 + 留一海区</span>
        <span class="pill">随机探索参照</span>
        <span class="pill">反向路径/时间置换负对照</span>
        <span class="pill">7/14/30/60天多尺度异常</span>
        <span class="pill">TE/CTE + Durbin影响分解</span>
        <span class="pill">115条南澳真实qPCR样本</span>
        <span class="pill">A/B/C证据分层</span>
      </div>
    </div>
    <div class="boundary"><b>能力边界：</b>PR-AUC等模型性能来自匿名合成数据的软件验证；
    南澳大利亚页使用真实qPCR观测进行事件回放，但不参与监督训练。地图和复核优先级
    不构成真实预报、自动停采指令或统一毒素阈值。</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 探索环境")
    days = st.select_slider("合成序列长度", [540, 720, 900], value=720)
    budget = st.slider("Agent实验预算", 4, 12, 8)
    holdout_region = st.selectbox("完全留出区域", REGIONS, index=3)
    test_fraction = st.select_slider(
        "前向留出比例", [0.20, 0.25, 0.30], value=0.25,
        format_func=lambda value: f"{value:.0%}",
    )
    seed = st.number_input("固定随机种子", 1, 9999, 42)
    run_clicked = st.button("重新运行探索", type="primary", width="stretch")
    st.caption("候选空间：局地/沿流 × 3/7/14/21/30/45天 × 两类模型；另运行四个竞赛等价机制模块。")

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

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("研究信号", "恢复" if recovered else "未恢复")
m2.metric("最佳候选", f"{best['route']} · {int(best['lag_days'])} d")
m3.metric("PR-AUC", f"{float(best['pr_auc']):.3f}")
m4.metric("Brier Skill", f"{float(best['brier_skill']):.3f}")
m5.metric("Top20%召回", f"{float(best['recall_at_top20']):.1%}")
m6.metric("校准误差 ECE", f"{float(best['ece']):.3f}")

tab_alert, tab_real, tab_methods, tab_agent, tab_evidence = st.tabs([
    "01 藻华与养殖风险", "02 南澳真实事件回放", "03 机制模块与影响分解",
    "04 Agent探索与研究信号", "05 证据链与复现",
])

with tab_alert:
    st.markdown("### 情景化HAB空间预警")
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
            "MHW强度（°C）", 0.0, 4.5, float(preset["mhw_intensity_c"]), .1,
            key=f"mhw_{preset_name}",
        )
        nitrate = st.slider(
            "Nitrate（mmol m⁻³）", 0.0, 10.0, float(preset["nitrate_mmol_m3"]), .1,
            key=f"nitrate_{preset_name}",
        )
        phosphate = st.slider(
            "Phosphate（mmol m⁻³）", 0.0, 1.5, float(preset["phosphate_mmol_m3"]), .05,
            key=f"phosphate_{preset_name}",
        )
        silicate = st.slider(
            "Silicate（mmol m⁻³）", 0.0, 12.0, float(preset["silicate_mmol_m3"]), .1,
            key=f"silicate_{preset_name}",
        )
        transport = st.slider(
            "输运/停留/汇聚代理", 0.0, 1.0, float(preset["transport_proxy"]), .05,
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

    st.markdown("### 由藻华风险转向海水养殖响应优先级")
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
    st.markdown("### 南澳大利亚2025复杂Karenia藻华：真实qPCR事件回放")
    st.success(
        "本页读取Murray等人论文配套Zenodo数据中的115条真实qPCR样本。"
        "这些数据不参与本项目保留的v3.1合成基准训练，也不被扩充成虚构负样本。"
    )
    real_observations, real_provenance = load_sa_real_case(ROOT / "data")
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
    real_card = replay["card"]

    rm1, rm2, rm3, rm4, rm5 = st.columns(5)
    rm1.metric("真实样本", f"{real_card['observations']}")
    rm2.metric("采样日期", f"{real_card['sampling_dates']}")
    rm3.metric("地点", f"{real_card['locations']}")
    rm4.metric("K. cristata检出", f"{real_card['k_cristata_detection_share']:.1%}")
    rm5.metric("峰值丰度", f"{real_card['peak_k_cristata']['cells_l'] / 1e6:.2f}M L⁻¹")
    peak = real_card["peak_k_cristata"]
    st.markdown(
        '<div class="signal">回放窗口内最高观测：'
        f'<b>{peak["location"]}</b> · {peak["date"]} · '
        f'<i>K. cristata</i> <b>{peak["cells_l"]:,.0f} cells L⁻¹</b>。'
        '这是采样峰值，不是整个海区的连续最大值。</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(real_qpcr_map(replay["sites"]), width="stretch", config={"displayModeBar": False})
    st.caption(
        "圆点大小表示该地点观测到的K. cristata峰值（对数缩放）；颜色表示其在该样本Karenia总丰度中的最大占比。"
        "丰度分档仅用于界面展示，不是公共卫生、停采或养殖监管阈值。"
    )

    st.markdown("#### 时间演变与物种组成")
    rt1, rt2 = st.columns([1.35, 1.0], gap="large")
    with rt1:
        timeline_chart = px.line(
            replay["timeline"], x="sample_date", y="k_cristata_peak_cells_l",
            markers=True, log_y=True,
            labels={"sample_date": "采样日期", "k_cristata_peak_cells_l": "当日观测峰值（cells L⁻¹，对数轴）"},
            title="K. cristata采样日峰值",
        )
        timeline_chart.update_layout(height=360, margin={"l": 5, "r": 5, "t": 50, "b": 5})
        st.plotly_chart(timeline_chart, width="stretch", config={"displayModeBar": False})
    with rt2:
        composition_chart = px.bar(
            replay["species"], x="species", y="summed_cells_l_across_samples",
            log_y=True, color="species",
            labels={"species": "Karenia物种", "summed_cells_l_across_samples": "跨样本丰度和（仅描述采样集）"},
            title="采样集物种构成",
        )
        composition_chart.update_layout(
            showlegend=False, height=360, margin={"l": 5, "r": 5, "t": 50, "b": 5}
        )
        st.plotly_chart(composition_chart, width="stretch", config={"displayModeBar": False})

    st.markdown("#### 真实数据自适应路由：能做什么、不能做什么")
    real_router = real_data_router(replay["observations"], has_daily_environment=False)
    st.dataframe(
        real_router, width="stretch", hide_index=True,
        column_config={"data_support": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0)},
    )
    st.warning(
        "当前真实包足以运行时空qPCR回放和物种组成分析，但单次事件、22个非均匀采样日期不足以稳健训练监督分类器，"
        "也不足以直接估计TE/CTE或Durbin网络。scripts/prepare_sa_real_replay.py保留NOAA OISST适配器；"
        "接入连续环境历史后，路由器才会开放多尺度环境异常分支。"
    )

    st.markdown("#### 基于真实观测的海水养殖复核优先级")
    ra1, ra2 = st.columns(2)
    real_production = ra1.selectbox(
        "养殖对象（真实回放）", list(PRODUCTION_PROFILES), key="real_production"
    )
    real_exposure = ra2.slider(
        "养殖暴露情景（真实回放）", .25, 1.0, .80, .05, key="real_exposure"
    )
    real_aqua = project_real_aquaculture_priority(
        replay["sites"], real_production, real_exposure
    )
    st.dataframe(
        real_aqua[[
            "location", "k_cristata_peak_cells_l", "observed_abundance_band",
            "observed_hazard_index", "farm_exposure_scenario",
            "vulnerability_coefficient", "verification_priority_index",
            "evidence_grade", "recommended_action",
        ]].head(15),
        width="stretch", hide_index=True,
    )
    st.caption(
        "该指数只安排现场复核顺序：观测丰度经对数缩放后与用户设定暴露、对象脆弱性相乘。"
        "它不是死亡概率、经济损失或停采阈值。"
    )
    rd1, rd2, rd3 = st.columns(3)
    rd1.download_button(
        "下载窗口内真实qPCR CSV",
        replay["observations"].to_csv(index=False).encode("utf-8-sig"),
        "sa_real_qpcr_replay.csv", "text/csv",
    )
    rd2.download_button(
        "下载真实回放卡 JSON",
        json.dumps(real_card, ensure_ascii=False, indent=2).encode("utf-8"),
        "sa_real_replay_card.json", "application/json",
    )
    rd3.download_button(
        "下载真实养殖复核优先级",
        real_aqua.to_csv(index=False).encode("utf-8-sig"),
        "sa_real_aquaculture_verification_priority.csv", "text/csv",
    )
    st.markdown(
        "数据来源：[Nature Ecology & Evolution论文](https://doi.org/10.1038/s41559-026-03115-0) · "
        "[Zenodo配套数据（CC BY 4.0）](https://doi.org/10.5281/zenodo.20227730)"
    )

with tab_methods:
    st.markdown("### 四个竞赛等价模块：同一条可检查证据链")
    st.info(
        "这里公开的是完整可运行、可复核的竞赛实现，不是已提交专利生产工程的逐行复刻。"
        "所有结果基于匿名合成数据，只验证方法链和预设信号恢复能力。"
    )

    st.markdown("#### 1. 多尺度异常检测：过去窗口、稳健尺度与事件合并")
    a1, a2, a3 = st.columns(3)
    a1.metric("检测尺度", "7 / 14 / 30 / 60 天")
    a2.metric("合并异常事件", f"{len(anomaly_events)}")
    a3.metric("未来信息进入滚动基准", "否")
    anomaly_region = st.selectbox("异常轨迹海区", REGIONS, index=3, key="anomaly_region")
    anomaly_plot = anomaly_daily[anomaly_daily["region"].eq(anomaly_region)].set_index("date")[[
        "anomaly_score_7d", "anomaly_score_14d", "anomaly_score_30d",
        "anomaly_score_60d", "multiscale_anomaly_score",
    ]]
    st.line_chart(anomaly_plot, height=300)
    st.dataframe(
        anomaly_events.head(12), width="stretch", hide_index=True,
        column_config={"peak_score": st.column_config.NumberColumn(format="%.3f")},
    )
    st.download_button(
        "下载多尺度事件目录", anomaly_events.to_csv(index=False).encode("utf-8-sig"),
        "multiscale_event_catalog.csv", "text/csv",
    )

    st.markdown("#### 2. 自适应路由：数据诊断 → 方法分支 → 决策原因")
    route_display = router_trace[[
        "branch", "compatibility_score", "routing_probability", "decision", "reason"
    ]].copy()
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

    st.markdown("#### 3. TE/CTE网络：方向、时滞、条件依赖与FDR")
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
    st.success(
        f"沿流CTE的跨边平均峰值位于 {peak_lag} 天；圆周移位置换保留源序列自相关，"
        "边级p值经Benjamini–Hochberg校正。"
    )
    st.dataframe(
        te_cte_network[[
            "source_region", "target_region", "lag_days", "te_bits", "cte_bits",
            "reverse_cte_bits", "net_directionality_bits", "permutation_p", "fdr_q",
            "significant_fdr_0_10",
        ]].head(16),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "下载TE/CTE边级网络", te_cte_network.to_csv(index=False).encode("utf-8-sig"),
        "te_cte_network.csv", "text/csv",
    )

    st.markdown("#### 4. 空间Durbin乘数：直接、间接与总影响")
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
    d1, d2, d3 = st.columns(3)
    d1.metric("空间系数 ρ", f"{float(spatial_diagnostics['rho']):.2f}")
    d2.metric("解释度（探索性）", f"{float(spatial_diagnostics['pseudo_r2']):.3f}")
    d3.metric("块Bootstrap", f"{int(spatial_diagnostics['bootstrap_repeats'])} 次")
    st.caption(
        "W为行标准化的匿名有向上游图；14天空间暴露尺度来自Agent与TE/CTE恢复结果。"
        "线性概率SDM的影响是合成环境中的关联尺度，不解释为真实海洋因果效应。"
    )
    st.dataframe(spatial_effects, width="stretch", hide_index=True)
    st.download_button(
        "下载Durbin影响分解", spatial_effects.to_csv(index=False).encode("utf-8-sig"),
        "spatial_durbin_effects.csv", "text/csv",
    )

with tab_agent:
    st.markdown("### 探索不是单点高分，而是同预算证据比较")
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
            title="阻断验证PR-AUC：Agent候选 vs 平凡解",
            text_auto=".3f",
            color_discrete_sequence=["#8aa6a3", "#c7a76c", "#0d7f79"],
        )
        fig.update_layout(showlegend=False, height=350, margin={"l": 5, "r": 5, "t": 55, "b": 10})
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("#### 随机探索参照")
        st.metric("相同预算随机恢复14天信号", f"{float(random_ref['hidden_signal_recovery_rate']):.1%}")
        st.metric("随机搜索中位最佳效用", f"{float(random_ref['median_best_utility']):.3f}")
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
    st.markdown("### 真实事件证据卡：只做外部复核，不冒充训练数据")
    case = SOUTH_AUSTRALIA_CASE
    st.markdown(
        '<div class="case-card">'
        f'<b>{case["title"]}</b><br>{case["period"]} · {case["spatial_extent"]} · '
        f'证据等级 {case["evidence_grade"]}<br><span class="small-muted">'
        f'{case["model_use"]}</span></div>',
        unsafe_allow_html=True,
    )
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown("#### 确认证据")
        for item in case["confirmed_signals"]:
            st.markdown(f"- {item}")
    with e2:
        st.markdown("#### 已报道影响")
        for item in case["reported_impacts"]:
            st.markdown(f"- {item}")
    st.info(case["aquaculture_interpretation"])
    for source in case["sources"]:
        st.markdown(f"- [{source['label']}]({source['url']})")

    st.markdown("### 数据与解释边界")
    v1, v2, v3 = st.columns(3)
    v1.info("**MHW**\n\nSST超过日历日p90时，强度=SST−季节气候平均值")
    v2.info("**营养盐**\n\nNitrate、Phosphate、Silicate分项进入模型，不合并成模糊指数")
    v3.info("**微塑料**\n\n仅代理输运/停留/汇聚状态，不作为HAB直接生物驱动")

    st.markdown("### 机器可检查的发现卡")
    st.json(card)
    card_bytes = json.dumps(card, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button("下载发现卡 JSON", card_bytes, "discovery_card.json", "application/json")
    evidence_bundle = {
        "discovery_card": card,
        "external_case": case,
        "baselines": baselines.to_dict("records"),
        "negative_controls": controls.to_dict("records"),
        "adaptive_router": router_trace.to_dict("records"),
        "te_cte_lag_summary": te_cte_lag_summary.to_dict("records"),
        "spatial_durbin_effects": spatial_effects.to_dict("records"),
        "south_australia_real_replay": real_card,
        "south_australia_real_data_provenance": real_provenance,
    }
    d2.download_button(
        "下载证据包 JSON",
        json.dumps(evidence_bundle, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        "semifinal_evidence_bundle.json",
        "application/json",
    )

st.divider()
st.caption(
    "GlobalHAB-Agent v3.2 GOAI Semifinal · synthetic benchmark + real-event replay · "
    "no operational, causal or automatic closure claim"
)
