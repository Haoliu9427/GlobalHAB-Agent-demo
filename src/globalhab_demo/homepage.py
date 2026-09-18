"""Interactive overview for the GlobalHAB-Agent workspaces.

The homepage reads registered outputs/session state only. It visualises the
existing research, validation, case and interpretation workflow without
recomputing scientific results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}


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


def _workspace_jump(label: str, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        st.session_state["_workspace_jump"] = label
        st.rerun()


def _case_state(root: Path) -> tuple[int, int, int, dict[str, int]]:
    try:
        from globalhab_demo.case_manager import case_status_counts, list_cases

        cases = list_cases(root)
        counts = case_status_counts(root)
        pending = (
            int(counts.get("pending_review", 0))
            + int(counts.get("in_progress", 0))
            + int(counts.get("visual_defer", 0))
        )
        evidence = sum(len(c.get("evidence") or []) for c in cases)
        return len(cases), pending, evidence, {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return 0, 0, 0, {}


def _visual_library_count(root: Path) -> int:
    p = root / "data" / "field_visual" / "user_library" / "records.csv"
    if not p.exists():
        return 0
    try:
        return int(len(pd.read_csv(p)))
    except Exception:
        return 0


def _registered_user_runs(root: Path) -> int:
    base = root / "outputs" / "real_training"
    if not base.exists():
        return 0
    return len([p for p in base.glob("*") if p.is_dir() and (p / "metrics.csv").exists()])


def _layout(fig: go.Figure, *, height: int, margin_t: int = 46) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=margin_t, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, Arial", color="#234857", size=12),
        hoverlabel=dict(font_size=12, font_family="Microsoft YaHei, Arial"),
        title=dict(font=dict(size=16, color="#173f52"), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _workspace_flow_figure() -> go.Figure:
    labels = ["项目总览", "研究与验证", "自有数据分析", "现场影像甄别", "大模型结果解读"]
    source = [0, 0, 0, 0, 1, 1, 2, 3]
    target = [1, 2, 3, 4, 3, 4, 4, 4]
    values = [1] * len(source)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=22,
                thickness=18,
                line=dict(color="#b8d4da", width=1),
                label=labels,
                color=["#173f52", "#2b7c91", "#5a9fa4", "#55a6a0", "#7694b2"],
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color="rgba(74,145,157,0.18)",
                hoverinfo="skip",
            ),
        )
    )
    fig.update_layout(title="工作区联动", height=310)
    return _layout(fig, height=310)


def _case_donut(counts: dict[str, int], total: int, evidence_total: int) -> go.Figure:
    mapping = [
        ("待复核", "pending_review"),
        ("处理中", "in_progress"),
        ("视觉复核", "visual_screened"),
        ("视觉DEFER", "visual_defer"),
        ("实验室待确认", "lab_pending"),
        ("已确认", "confirmed"),
        ("已归档", "archived"),
    ]
    labels = [label for label, key in mapping if counts.get(key, 0) > 0]
    values = [counts.get(key, 0) for _, key in mapping if counts.get(key, 0) > 0]
    if not values:
        labels, values = ["暂无Case"], [1]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            sort=False,
            textinfo="none",
            hovertemplate="%{label}<br>%{value} 个<extra></extra>" if total else "%{label}<extra></extra>",
            marker=dict(line=dict(color="white", width=2)),
        )
    )
    fig.add_annotation(
        x=0.5,
        y=0.55,
        text=f"<b>{total}</b>",
        showarrow=False,
        font=dict(size=29, color="#173f52"),
    )
    fig.add_annotation(
        x=0.5,
        y=0.41,
        text=f"Case · {evidence_total} 条证据",
        showarrow=False,
        font=dict(size=11, color="#5e747d"),
    )
    fig.update_layout(title="Case 状态", showlegend=True)
    return _layout(fig, height=310)


def _agent_trace_figure(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "agent_log.csv")
    fig = go.Figure()
    if not df.empty and {"step", "pr_auc"}.issubset(df.columns):
        custom = df[[c for c in ["route", "lag_days", "model", "utility", "status"] if c in df.columns]].astype(str).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["pr_auc"],
                mode="lines+markers",
                line=dict(width=2.2, color="#2b7c91"),
                marker=dict(size=10, color=df["pr_auc"], colorscale="Teal", showscale=False, line=dict(color="white", width=1)),
                customdata=custom,
                hovertemplate=(
                    "Step %{x}<br>AP %{y:.3f}<br>"
                    "路径 %{customdata[0]} · 时滞 %{customdata[1]} d<br>"
                    "模型 %{customdata[2]} · Utility %{customdata[3]}<extra></extra>"
                ),
                name="Agent search",
            )
        )
        best_i = int(df["pr_auc"].astype(float).idxmax())
        fig.add_annotation(
            x=df.loc[best_i, "step"],
            y=df.loc[best_i, "pr_auc"],
            text="当前候选",
            showarrow=True,
            arrowhead=2,
            ax=36,
            ay=-38,
            bgcolor="rgba(255,255,255,.9)",
            bordercolor="#b9d8dc",
        )
    fig.update_xaxes(title="试验步", dtick=1, gridcolor="rgba(64,110,120,.10)")
    fig.update_yaxes(title="Average Precision", rangemode="tozero", gridcolor="rgba(64,110,120,.10)")
    fig.update_layout(title="Agent 探索轨迹", showlegend=False)
    return _layout(fig, height=340)


def _sa_replay_figure(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "sa_real_replay_timeline.csv")
    fig = go.Figure()
    if not df.empty and {"sample_date", "k_cristata_peak_cells_l"}.issubset(df.columns):
        df = df.copy()
        df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
        df = df.dropna(subset=["sample_date"]).sort_values("sample_date")
        marker_size = 8 + 2.5 * pd.to_numeric(df.get("samples", 1), errors="coerce").fillna(1).clip(1, 8)
        fig.add_trace(
            go.Scatter(
                x=df["sample_date"],
                y=df["k_cristata_peak_cells_l"],
                mode="lines+markers",
                line=dict(width=2, color="#428d94"),
                marker=dict(size=marker_size, color="#63aaa6", line=dict(color="white", width=1)),
                customdata=df[[c for c in ["samples", "locations", "k_cristata_detection_share"] if c in df.columns]].to_numpy(),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>峰值 %{y:.2e} cells/L<br>"
                    "样本 %{customdata[0]} · 站点 %{customdata[1]}<br>"
                    "检出比例 %{customdata[2]:.0%}<extra></extra>"
                ),
                name="K. cristata",
            )
        )
    fig.update_xaxes(title=None, gridcolor="rgba(64,110,120,.08)")
    fig.update_yaxes(title="cells/L", type="log", gridcolor="rgba(64,110,120,.10)")
    fig.update_layout(title="南澳真实事件回放", showlegend=False)
    return _layout(fig, height=340)


def _norway_forward_figure(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "norway_forward_benchmark_folds.csv")
    fig = go.Figure()
    if not df.empty and "test_window" in df.columns:
        series = [
            ("模型 AP", "model_average_precision"),
            ("参考模型 AP", "reference_average_precision"),
            ("季节基线 AP", "seasonal_average_precision"),
        ]
        for label, col in series:
            if col in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df["test_window"],
                        y=df[col],
                        name=label,
                        hovertemplate=f"%{{x}}<br>{label} %{{y:.3f}}<extra></extra>",
                    )
                )
    fig.update_xaxes(title=None, gridcolor="rgba(64,110,120,.05)")
    fig.update_yaxes(title="Average Precision", rangemode="tozero", gridcolor="rgba(64,110,120,.10)")
    fig.update_layout(title="挪威前向验证", barmode="group")
    return _layout(fig, height=345)


def _method_ring_figure() -> go.Figure:
    labels = [
        "Transferability<br>& Generalization",
        "时序迁移", "空间迁移", "跨区域验证", "真实事件回放",
        "温度 / MHW", "营养盐", "时滞 / TE-CTE",
        "洋流 / 输运", "空间溢出 / SDM",
        "挪威前向验证", "自有数据",
        "南澳 Karenia", "现场影像", "生物响应",
    ]
    parents = [
        "",
        labels[0], labels[0], labels[0], labels[0],
        "时序迁移", "时序迁移", "时序迁移",
        "空间迁移", "空间迁移",
        "跨区域验证", "跨区域验证",
        "真实事件回放", "真实事件回放", "真实事件回放",
    ]
    values = [12, 3, 3, 3, 3, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1]
    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            insidetextorientation="radial",
            maxdepth=3,
            hovertemplate="%{label}<extra></extra>",
            marker=dict(line=dict(color="white", width=2)),
        )
    )
    fig.update_layout(title="环境因子与迁移验证", uniformtext=dict(minsize=10, mode="hide"))
    return _layout(fig, height=470)


def render(root: Any) -> None:
    root = Path(root)
    discovery = _read_json(root / "outputs" / "discovery_card.json")
    norway = _read_json(root / "outputs" / "norway_forward_benchmark_card.json")
    best = discovery.get("best_candidate") or {}

    case_total, case_pending, evidence_total, case_counts = _case_state(root)
    library_count = _visual_library_count(root)
    own_runs = _registered_user_runs(root)

    st.markdown(
        """
        <div class="home-heading home-heading-visual">
          <div class="home-title">项目总览</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Small numeric strip: only the numbers needed to orient the charts below.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前候选", f"{best.get('route', '—')} · {best.get('lag_days', '—')} d")
    m2.metric("挪威 AP", f"{float(norway.get('model_average_precision', 0)):.3f}" if norway else "—")
    m3.metric("待处理 Case", case_pending)
    m4.metric("影像记录", library_count)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.plotly_chart(_workspace_flow_figure(), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        st.plotly_chart(_case_donut(case_counts, case_total, evidence_total), use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(_agent_trace_figure(root), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        st.plotly_chart(_sa_replay_figure(root), use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.plotly_chart(_norway_forward_figure(root), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        st.plotly_chart(_method_ring_figure(), use_container_width=True, config=PLOTLY_CONFIG)

    # Minimal navigation: the homepage is a visual overview, not another documentation page.
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        _workspace_jump("研究与验证", "home_to_research")
    with nav2:
        _workspace_jump(f"自有数据分析 · {own_runs}", "home_to_own")
    with nav3:
        _workspace_jump(f"现场影像甄别 · {case_pending}", "home_to_vision")
    with nav4:
        _workspace_jump("大模型结果解读", "home_to_llm")
