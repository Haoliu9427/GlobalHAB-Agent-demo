"""Visual control-board homepage for GlobalHAB-Agent.

The homepage reads existing outputs only. It does not recompute models or create
new scientific results. Every quantitative panel is grounded in files already
produced by the project.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _workspace_jump(label: str, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        st.session_state["_workspace_jump"] = label
        st.rerun()


def _case_state(root: Path) -> tuple[int, int, int, dict[str, int]]:
    try:
        from globalhab_demo.case_manager import case_status_counts, list_cases

        cases = list_cases(root)
        counts = case_status_counts(root)
        pending = sum(int(counts.get(k, 0)) for k in ("pending_review", "in_progress", "visual_defer", "visual_screened", "lab_pending"))
        evidence = sum(len(c.get("evidence") or []) for c in cases)
        return len(cases), pending, evidence, {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return 0, 0, 0, {}


def _visual_library_count(root: Path) -> int:
    path = root / "data" / "field_visual" / "user_library" / "records.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _registered_user_runs(root: Path) -> int:
    base = root / "outputs" / "real_training"
    if not base.exists():
        return 0
    return sum(1 for p in base.iterdir() if p.is_dir() and (p / "metrics.csv").exists())


def _layout(fig: go.Figure, *, height: int, margin_t: int = 50) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=26, r=24, t=margin_t, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, PingFang SC, Arial", color="#173f52", size=12),
        hoverlabel=dict(font_size=12, font_family="Microsoft YaHei, PingFang SC, Arial"),
        title=dict(font=dict(size=17, color="#173f52"), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _icon(kind: str) -> str:
    icons = {
        "data": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"/></svg>',
        "route": '<svg viewBox="0 0 24 24"><circle cx="7" cy="7" r="3"/><circle cx="17" cy="17" r="3"/><path d="M9.5 8.5l5 6"/><path d="M13 5h6v6"/></svg>',
        "science": '<svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="17" r="2.5"/><circle cx="19" cy="17" r="2.5"/><path d="M10.7 7.2L6.5 14.8M13.3 7.2l4.2 7.6M7.5 17h9"/></svg>',
        "evidence": '<svg viewBox="0 0 24 24"><path d="M7 3h8l4 4v14H7z"/><path d="M15 3v5h5M10 12h6M10 16h6"/></svg>',
        "transfer": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 4.2 6 4.2 9S15 18 12 21M12 3C9 6 7.8 9 7.8 12S9 18 12 21"/></svg>',
    }
    return icons[kind]


def _overview_header(best: dict[str, Any], norway: dict[str, Any], case_pending: int, library_count: int) -> str:
    route = html.escape(str(best.get("route", "—")))
    lag = best.get("lag_days")
    candidate = f"{route} · {lag}d" if lag is not None else route
    cut_date = html.escape(str(best.get("cut_date", "—")))
    ap = norway.get("model_average_precision")
    ap_txt = f"{float(ap):.3f}" if ap is not None else "—"
    return f"""
    <div class="home-shell">
      <div class="home-topband">
        <div class="home-topband-left">
          <div class="home-title">项目总览</div>
          <div class="home-tagline">多源观测 · 科学推理 · 证据落地 · 迁移验证</div>
        </div>
        <div class="home-topband-right">让数据看见海洋，<br>用科学守护蔚蓝</div>
      </div>
      <div class="home-status-grid">
        <div class="home-status-card"><div class="home-status-icon">◎</div><div class="home-status-copy"><span>当前候选</span><b>{candidate}</b><small>前向切分 · {cut_date}</small></div></div>
        <div class="home-status-card"><div class="home-status-icon">▥</div><div class="home-status-copy"><span>挪威 AP</span><b>{ap_txt}</b><small>2006–2019 长期前向验证</small></div></div>
        <div class="home-status-card"><div class="home-status-icon">▣</div><div class="home-status-copy"><span>待处理 Case</span><b>{case_pending}</b><small>现场复核与证据更新队列</small></div></div>
        <div class="home-status-card"><div class="home-status-icon">▤</div><div class="home-status-copy"><span>影像记录</span><b>{library_count}</b><small>本地现场影像库</small></div></div>
      </div>
    </div>
    """


def _research_flow_html() -> str:
    steps = [
        ("data", "1. 多源观测", "卫星 / 现场 / 模型", "多源数据汇聚"),
        ("route", "2. 路由与异常识别", "多尺度异常", "时空覆盖诊断"),
        ("science", "3. ST / STS 科学推理", "时滞 / 方向 / 阻断预测", "空间溢出效应"),
        ("evidence", "4. 证据落地与验证", "Case 现场复核", "影像 / 实验室证据"),
        ("transfer", "5. 迁移性评估", "时序 / 空间迁移", "跨区域前向验证"),
    ]
    pieces = []
    for i, (kind, title, line1, line2) in enumerate(steps):
        pieces.append(
            f'<div class="overview-step-card"><div class="overview-step-icon">{_icon(kind)}</div>'
            f'<div class="overview-step-title">{title}</div><div class="overview-step-meta">{line1}<br>{line2}</div></div>'
        )
        if i < len(steps) - 1:
            pieces.append('<div class="overview-arrow">→</div>')
    return (
        '<div class="panel-title-row"><div><div class="panel-title">研究思路概览</div>'
        '<div class="panel-subtitle">把完整方法框架压缩成首页可读的五步闭环</div></div></div>'
        '<div class="overview-step-flow">' + "".join(pieces) + '</div>'
    )


def _workspace_linkage_html(own_runs: int, case_pending: int, library_count: int) -> str:
    items = [
        ("项目总览", "状态联动"),
        ("研究与验证", "ST / STS"),
        ("自有数据分析", f"已登记 {own_runs} 组结果"),
        ("现场影像甄别", f"影像 {library_count} · 队列 {case_pending}"),
        ("大模型结果解读", "结构化解释"),
    ]
    pieces = []
    for i, (title, sub) in enumerate(items):
        pieces.append(f'<div class="workspace-box"><div class="workspace-box-title">{title}</div><div class="workspace-box-sub">{sub}</div></div>')
        if i < len(items) - 1:
            pieces.append('<div class="workspace-arrow">→</div>')
    return (
        '<div class="panel-title-row"><div><div class="panel-title">工作区联动</div>'
        '<div class="panel-subtitle">从研究候选到现场证据与结果解释</div></div></div>'
        '<div class="workspace-flow">' + "".join(pieces) + '</div>'
    )


def _case_donut(counts: dict[str, int], total: int) -> go.Figure:
    if total <= 0:
        labels, values, colors = ["暂无 Case"], [1], ["#1c78c0"]
    else:
        groups = [
            ("待复核", counts.get("pending_review", 0) + counts.get("visual_defer", 0), "#1c78c0"),
            ("处理中", counts.get("in_progress", 0) + counts.get("visual_screened", 0) + counts.get("lab_pending", 0), "#2cb7b1"),
            ("已确认", counts.get("confirmed", 0), "#7cb9e8"),
            ("已归档", counts.get("archived", 0), "#8aa0b8"),
        ]
        groups = [g for g in groups if int(g[1]) > 0]
        labels, values, colors = [g[0] for g in groups], [int(g[1]) for g in groups], [g[2] for g in groups]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.70, sort=False, textinfo="none", marker=dict(colors=colors, line=dict(color="#fff", width=2)), hovertemplate="%{label}<br>%{value} 个<extra></extra>"))
    fig.add_annotation(x=.5, y=.55, text=f"<b>{total}</b>", showarrow=False, font=dict(size=28, color="#173f52"))
    fig.add_annotation(x=.5, y=.39, text="Case 总数", showarrow=False, font=dict(size=12, color="#74899a"))
    fig.update_layout(title="Case 状态", legend=dict(orientation="v", x=1.0, y=.86))
    return _layout(fig, height=292)


def _agent_trace_figure(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "agent_log.csv")
    fig = go.Figure()
    if not df.empty and {"step", "pr_auc"}.issubset(df.columns):
        frame = df.sort_values("step").copy()
        fig.add_trace(go.Scatter(x=frame["step"], y=frame["pr_auc"], mode="lines+markers", line=dict(width=2.6, color="#1f739c"), marker=dict(size=8, color="#6ec5cf", line=dict(color="#fff", width=1.3)), hovertemplate="试验步 %{x}<br>AP %{y:.3f}<extra></extra>", name="探索轨迹"))
        idx = frame["pr_auc"].astype(float).idxmax()
        best_step, best_ap = int(frame.loc[idx, "step"]), float(frame.loc[idx, "pr_auc"])
        fig.add_trace(go.Scatter(x=[best_step], y=[best_ap], mode="markers", marker=dict(size=16, color="#27c3a6", line=dict(color="#0c8b73", width=2)), name="当前候选", hovertemplate="当前候选<br>试验步 %{x}<br>AP %{y:.3f}<extra></extra>"))
        fig.add_annotation(x=best_step, y=best_ap, text=f"当前候选<br><b>AP = {best_ap:.3f}</b>", showarrow=True, arrowhead=2, ax=18, ay=-48, bgcolor="rgba(255,255,255,.96)", bordercolor="#d2e0ea")
    fig.update_xaxes(title="试验步", dtick=1)
    fig.update_yaxes(title="Average Precision (AP)", rangemode="tozero")
    fig.update_layout(title="Agent 探索轨迹")
    return _layout(fig, height=330)


def _normalize_row(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lo, hi = np.nanmin(values[finite]), np.nanmax(values[finite])
    if hi <= lo:
        out = np.zeros_like(values)
        out[finite] = .5
        return out
    return (values - lo) / (hi - lo)


def _sa_replay_heatmap(root: Path) -> go.Figure:
    """Real timeline heatmap; no synthetic station expansion is used."""
    df = _read_csv(root / "outputs" / "sa_real_replay_timeline.csv")
    fig = go.Figure()
    required = {"sample_date", "k_cristata_peak_cells_l", "k_cristata_median_cells_l", "k_cristata_detection_share", "samples"}
    if df.empty or not required.issubset(df.columns):
        fig.add_annotation(text="暂无事件回放数据", x=.5, y=.5, showarrow=False)
        fig.update_layout(title="南澳真实事件回放")
        return _layout(fig, height=330)

    frame = df.copy()
    frame["sample_date"] = pd.to_datetime(frame["sample_date"], errors="coerce")
    frame = frame.dropna(subset=["sample_date"]).sort_values("sample_date")
    x = frame["sample_date"].dt.strftime("%m-%d").tolist()
    peak = np.log10(np.clip(pd.to_numeric(frame["k_cristata_peak_cells_l"], errors="coerce").to_numpy(float), 1, None))
    median = np.log10(np.clip(pd.to_numeric(frame["k_cristata_median_cells_l"], errors="coerce").to_numpy(float), 1, None))
    share = pd.to_numeric(frame["k_cristata_detection_share"], errors="coerce").to_numpy(float)
    samples = pd.to_numeric(frame["samples"], errors="coerce").to_numpy(float)
    z = np.vstack([_normalize_row(peak), _normalize_row(median), _normalize_row(share), _normalize_row(samples)])
    text = np.vstack([
        np.array([f"{10**v:.2e}" for v in peak], dtype=object),
        np.array([f"{10**v:.2e}" for v in median], dtype=object),
        np.array([f"{v:.0%}" if np.isfinite(v) else "—" for v in share], dtype=object),
        np.array([f"{int(v)}" if np.isfinite(v) else "—" for v in samples], dtype=object),
    ])
    rows = ["K. cristata 峰值", "K. cristata 中位数", "检出样本占比", "采样数"]
    fig.add_trace(go.Heatmap(z=z, x=x, y=rows, text=text, colorscale=[[0,"#e9f7f4"],[.35,"#b9e1d4"],[.62,"#f3d78e"],[.82,"#ef9b5e"],[1,"#df5c43"]], zmin=0, zmax=1, colorbar=dict(title="各指标<br>相对强度", thickness=11), customdata=text, hovertemplate="%{y}<br>日期 %{x}<br>原始值 %{customdata}<extra></extra>"))
    fig.update_xaxes(title="日期（2025）", tickangle=-35)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="南澳真实事件回放")
    return _layout(fig, height=330)


def _evidence_matrix(root: Path) -> go.Figure:
    """Sparse evidence-coverage matrix using only reported project outputs."""
    lag = _read_csv(root / "outputs" / "te_cte_lag_summary.csv")
    spatial = _read_csv(root / "outputs" / "spatial_durbin_effects.csv")
    controls = _read_csv(root / "outputs" / "negative_controls.csv")
    sa_species = _read_csv(root / "outputs" / "sa_real_species_summary.csv")
    sa_sites = _read_csv(root / "outputs" / "sa_real_site_summary.csv")
    norway = _read_csv(root / "outputs" / "norway_forward_benchmark_folds.csv")
    diag = _read_json(root / "outputs" / "method_diagnostics.json")

    rows = ["温度 / MHW", "营养盐", "洋流输运", "空间溢出", "南澳事件", "挪威前向"]
    cols = ["时滞证据", "空间证据", "真实事件", "跨区域前向"]
    text = np.full((len(rows), len(cols)), "", dtype=object)
    z = np.full((len(rows), len(cols)), np.nan, dtype=float)

    if not lag.empty and "mean_cte_bits" in lag:
        r = lag.loc[pd.to_numeric(lag["mean_cte_bits"], errors="coerce").idxmax()]
        text[0,0] = f"CTE {float(r['mean_cte_bits']):.3f} bit<br>{int(r['lag_days'])} d"
        text[2,0] = f"净方向 {float(r.get('mean_net_directionality_bits',0)):.3f} bit"
        z[0,0] = z[2,0] = 1

    if not spatial.empty and {"variable","effect_type","effect_per_1sd"}.issubset(spatial.columns):
        nutrient = spatial[(spatial["variable"]=="nutrient_context") & (spatial["effect_type"]=="indirect")]
        anomaly = spatial[(spatial["variable"]=="multiscale_anomaly_score_lag14") & (spatial["effect_type"]=="indirect")]
        if not nutrient.empty:
            text[1,1] = f"间接 {float(nutrient.iloc[0]['effect_per_1sd']):+.3f}"
            z[1,1] = 1
        if not anomaly.empty:
            text[3,1] = f"间接 {float(anomaly.iloc[0]['effect_per_1sd']):+.3f}"
            z[3,1] = 1

    spatial_support = _safe_float((diag.get("router") or {}).get("spatial_support"), np.nan)
    if np.isfinite(spatial_support):
        text[2,1] = f"Router {spatial_support:.2f}"
        z[2,1] = 1

    if not sa_species.empty and "species" in sa_species:
        kc = sa_species[sa_species["species"].astype(str).str.contains("cristata", case=False, na=False)]
        if not kc.empty:
            share = _safe_float(kc.iloc[0].get("detection_share"), 0)
            peak = _safe_float(kc.iloc[0].get("peak_cells_l"), 0)
            text[4,2] = f"峰值 {peak:.2e}<br>检出 {share:.0%}"
            z[4,2] = 1
        elif not sa_sites.empty:
            peak = _safe_float(pd.to_numeric(sa_sites.get("k_cristata_peak_cells_l"), errors="coerce").max(), 0)
            text[4,2] = f"峰值 {peak:.2e}"
            z[4,2] = 1

    if not norway.empty and "model_average_precision" in norway:
        mean_ap = float(pd.to_numeric(norway["model_average_precision"], errors="coerce").mean())
        mean_recall = float(pd.to_numeric(norway.get("top10_recall"), errors="coerce").mean()) if "top10_recall" in norway else np.nan
        text[5,3] = f"AP {mean_ap:.3f}" + (f"<br>Top10 {mean_recall:.0%}" if np.isfinite(mean_recall) else "")
        z[5,3] = 1

    # A negative-control cell is attached to the MHW temporal evidence in hover, not scored as a new value.
    control_note = ""
    if not controls.empty and "pr_auc" in controls:
        vals = pd.to_numeric(controls["pr_auc"], errors="coerce").dropna().tolist()
        if vals:
            control_note = "；负对照 AP " + "/".join(f"{v:.3f}" for v in vals[:2])
            text[0,0] = str(text[0,0]) + control_note

    display = np.where(np.isfinite(z), 1.0, 0.0)
    fig = go.Figure(go.Heatmap(z=display, x=cols, y=rows, text=text, texttemplate="%{text}", textfont=dict(size=11, color="#173f52"), colorscale=[[0,"#f7fbfe"],[.001,"#f7fbfe"],[1,"#a9d8f2"]], zmin=0, zmax=1, showscale=False, hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>"))
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="环境因子与迁移验证")
    return _layout(fig, height=330, margin_t=58)


def _norway_forward_heatmap(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "norway_forward_benchmark_folds.csv")
    fig = go.Figure()
    if df.empty or "test_window" not in df.columns:
        fig.add_annotation(text="暂无挪威前向验证结果", x=.5, y=.5, showarrow=False)
        fig.update_layout(title="挪威前向验证")
        return _layout(fig, height=225)
    columns = df["test_window"].astype(str).tolist()
    rows = ["模型 AP", "参考模型 AP", "季节基线 AP"]
    z = np.array([
        pd.to_numeric(df.get("model_average_precision", 0), errors="coerce").to_numpy(float),
        pd.to_numeric(df.get("reference_average_precision", 0), errors="coerce").to_numpy(float),
        pd.to_numeric(df.get("seasonal_average_precision", 0), errors="coerce").to_numpy(float),
    ])
    text = [[f"{v:.3f}" if np.isfinite(v) else "—" for v in row] for row in z]
    fig.add_trace(go.Heatmap(z=z, x=columns, y=rows, text=text, texttemplate="%{text}", colorscale=[[0,"#eef6fd"],[.35,"#cbe5f8"],[.7,"#72b8e4"],[1,"#1c78c0"]], zmin=0, zmax=max(.25, float(np.nanmax(z))+0.02), colorbar=dict(title="AP", thickness=10), hovertemplate="%{y}<br>%{x}<br>AP %{z:.3f}<extra></extra>"))
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="挪威前向验证")
    return _layout(fig, height=225, margin_t=50)


def render(root: Any) -> None:
    root = Path(root)
    discovery = _read_json(root / "outputs" / "discovery_card.json")
    norway_card = _read_json(root / "outputs" / "norway_forward_benchmark_card.json")
    best = discovery.get("best_candidate") or {}
    case_total, case_pending, _, case_counts = _case_state(root)
    library_count = _visual_library_count(root)
    own_runs = _registered_user_runs(root)

    st.markdown(_overview_header(best, norway_card, case_pending, library_count), unsafe_allow_html=True)

    left, right = st.columns([2.55, 1.0], gap="large")
    with left:
        st.markdown('<div class="home-panel">' + _research_flow_html() + '</div>', unsafe_allow_html=True)
    with right:
        st.plotly_chart(_case_donut(case_counts, case_total), use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown('<div class="home-panel">' + _workspace_linkage_html(own_runs, case_pending, library_count) + '</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.0, 1.05, 1.18], gap="large")
    with c1:
        st.plotly_chart(_agent_trace_figure(root), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(_sa_replay_heatmap(root), use_container_width=True, config=PLOTLY_CONFIG)
    with c3:
        st.plotly_chart(_evidence_matrix(root), use_container_width=True, config=PLOTLY_CONFIG)

    bottom_left, bottom_right = st.columns([1.42, .58], gap="large")
    with bottom_left:
        st.plotly_chart(_norway_forward_heatmap(root), use_container_width=True, config=PLOTLY_CONFIG)
    with bottom_right:
        st.markdown('<div class="home-ocean-summary"><div class="home-ocean-text">跨越海域的知识迁移<br>服务于更安全的海洋</div></div>', unsafe_allow_html=True)

    nav1, nav2, nav3, nav4 = st.columns(4, gap="small")
    with nav1:
        _workspace_jump("研究与验证", "home_to_research")
    with nav2:
        _workspace_jump("自有数据分析", "home_to_own")
    with nav3:
        _workspace_jump("现场影像甄别", "home_to_vision")
    with nav4:
        _workspace_jump("大模型结果解读", "home_to_llm")
