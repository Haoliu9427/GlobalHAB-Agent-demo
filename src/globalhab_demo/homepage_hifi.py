"""High-fidelity visual control-board homepage for GlobalHAB-Agent.

The homepage is a read-only projection of existing project outputs. It does not
recompute models or create scientific results. The upper half is rendered as a
compact product dashboard; the lower panels remain Plotly-based and interactive.
"""
from __future__ import annotations

import base64
import html
from urllib.parse import quote
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from globalhab_demo.display_locale import st

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": "reset",
}

BLUE = "#1d79c5"
BLUE_DARK = "#155f9d"
SKY = "#79bbed"
TEAL = "#2ab7b0"
NAVY = "#123b5a"
MUTED = "#6e8598"
GRID = "rgba(48,91,122,.12)"


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


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _case_state(root: Path) -> tuple[int, int, int, dict[str, int]]:
    try:
        from globalhab_demo.case_manager import case_status_counts, list_cases
        cases = list_cases(root)
        counts = case_status_counts(root)
        pending = sum(
            int(counts.get(k, 0))
            for k in ("pending_review", "in_progress", "visual_defer", "visual_screened", "lab_pending")
        )
        evidence = sum(len(c.get("evidence") or []) for c in cases)
        return len(cases), pending, evidence, {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return 0, 0, 0, {}


def _visual_library_count(root: Path) -> int:
    path = root / "data" / "field_visual" / "user_library" / "records.csv"
    try:
        return int(len(pd.read_csv(path))) if path.exists() else 0
    except Exception:
        return 0


def _registered_user_runs(root: Path) -> int:
    base = root / "outputs" / "real_training"
    if not base.exists():
        return 0
    return sum(1 for p in base.iterdir() if p.is_dir() and (p / "metrics.csv").exists())


def _svg_icon(kind: str) -> str:
    icons = {
        "database": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"/></svg>',
        "bars": '<svg viewBox="0 0 24 24"><path d="M5 20V11h3v9M11 20V6h3v14M17 20V3h3v17M3 20h19"/></svg>',
        "doc": '<svg viewBox="0 0 24 24"><path d="M7 3h8l4 4v14H7z"/><path d="M15 3v5h5M10 12h6M10 16h6"/></svg>',
        "image": '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M4 18l5-5 4 4 3-3 4 4"/></svg>',
        "search": '<svg viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/></svg>',
        "science": '<svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="17" r="2.5"/><circle cx="19" cy="17" r="2.5"/><path d="M10.7 7.2L6.5 14.8M13.3 7.2l4.2 7.6M7.5 17h9"/></svg>',
        "globe": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 4.2 6 4.2 9S15 18 12 21M12 3C9 6 7.8 9 7.8 12S9 18 12 21"/></svg>',
        "home": '<svg viewBox="0 0 24 24"><path d="M3 11.5L12 4l9 7.5"/><path d="M5.5 10.5V21h13V10.5M9.5 21v-6h5v6"/></svg>',
        "bulb": '<svg viewBox="0 0 24 24"><path d="M9 18h6M9.5 21h5"/><path d="M8 15c-1.5-1.2-2.5-3-2.5-5A6.5 6.5 0 1118.5 10c0 2-1 3.8-2.5 5-.8.7-1 1.3-1 2H9c0-.7-.2-1.3-1-2z"/></svg>',
    }
    return icons.get(kind, icons["doc"])


def _kpi_card(icon: str, label: str, value: str, note: str, accent: str = "blue") -> str:
    return (
        f'<div class="hf-kpi-card hf-accent-{accent}">' 
        f'<div class="hf-kpi-icon">{_svg_icon(icon)}</div>'
        f'<div class="hf-kpi-copy"><span>{html.escape(label)}</span>'
        f'<b title="{html.escape(value)}">{html.escape(value)}</b>'
        f'<small>{note}</small></div><div class="hf-kpi-arrow">›</div></div>'
    )


def _case_donut_html(counts: dict[str, int], total: int) -> str:
    groups = [
        ("待复核", counts.get("pending_review", 0) + counts.get("visual_defer", 0), "#1675bc"),
        ("处理中", counts.get("in_progress", 0) + counts.get("visual_screened", 0) + counts.get("lab_pending", 0), "#2bb7af"),
        ("已验证", counts.get("confirmed", 0), "#7cb9e8"),
        ("已归档", counts.get("archived", 0), "#93a8b9"),
    ]
    if total <= 0:
        donut = "conic-gradient(#e3f0fb 0 100%)"
    else:
        parts, cursor = [], 0.0
        for _, value, color in groups:
            frac = max(0.0, float(value) / float(total)) * 100.0
            if frac <= 0:
                continue
            parts.append(f"{color} {cursor:.3f}% {cursor + frac:.3f}%")
            cursor += frac
        if cursor < 100:
            parts.append(f"#edf4f9 {cursor:.3f}% 100%")
        donut = "conic-gradient(" + ",".join(parts) + ")"
    legend = "".join(
        f'<div class="hf-case-legend-row"><i style="background:{color}"></i><span>{name}</span><b>{value} ({(value/total*100 if total else 0):.0f}%)</b></div>'
        for name, value, color in groups
    )
    return f'''
    <div class="hf-card hf-case-panel">
      <div class="hf-card-title-row"><div class="hf-card-title">Case状态</div><a class="hf-card-action" href="?workspace=研究验证&section=风险研判&cases=1" target="_self">查看全部 →</a></div>
      <div class="hf-case-body">
        <div class="hf-donut" style="background:{donut}"><div><b>{total}</b><span>Case 总数</span></div></div>
        <div class="hf-case-legend">{legend}</div>
      </div>
    </div>'''


def _research_flow_html() -> str:
    import math
    labels = ["多源观测", "异常识别", "传播推理", "现场复核", "迁移验证"]
    sections = ["数据来源与复核", "科学解释", "探索与验证", "真实事件回放", "真实数据训练与验证"]
    colors = ["#123F59", "#176981", "#248D9C", "#50A9AE", "#397687"]
    def xy(r, degrees):
        angle = math.radians(degrees)
        return 180+r*math.cos(angle), 155+r*math.sin(angle)
    pieces=[]
    for i,(label,section,color) in enumerate(zip(labels,sections,colors)):
        start=-126+i*72+2;end=start+63
        x1,y1=xy(132,start);x2,y2=xy(132,end)
        x3,y3=xy(70,end);x4,y4=xy(70,start)
        tipx,tipy=xy(101,end+9);notchx,notchy=xy(101,start+8)
        tx,ty=xy(101,(start+end)/2)
        path=f"M{x1:.2f},{y1:.2f} A132,132 0 0 1 {x2:.2f},{y2:.2f} L{tipx:.2f},{tipy:.2f} L{x3:.2f},{y3:.2f} A70,70 0 0 0 {x4:.2f},{y4:.2f} L{notchx:.2f},{notchy:.2f} Z"
        href=f"?workspace={quote('研究验证')}&section={quote(section)}"
        pieces.append(f'<a href="{href}" target="_self" aria-label="{label}"><title>{label} · 查看对应工作区</title><path d="{path}" fill="{color}"/><text x="{tx:.2f}" y="{ty+5:.2f}" text-anchor="middle">{label}</text></a>')
    return f'''<div class="hf-card hf-research-panel hf-sector-panel">
      <div class="hf-card-title">研究思路概览</div>
      <svg class="hf-research-wheel" viewBox="20 0 320 310" role="img" aria-label="研究流程：多源观测、异常识别、传播推理、现场复核、迁移验证">
      {''.join(pieces)}
      <circle cx="180" cy="155" r="60" fill="#F0F8FC"/>
      <text class="wheel-center-title" x="180" y="151" text-anchor="middle">科学Agent</text>
      <text class="wheel-center-sub" x="180" y="174" text-anchor="middle">观测 · 推理 · 验证</text>
      </svg></div>'''


def _start_task_html() -> str:
    return f'''<section class="hf-task-launch" aria-label="发起研究任务">
      <div class="hf-launch-copy"><b>开始一项研究</b><p>提出目标，上传观测与影像。</p></div>
      <a href="?workspace={quote('数据分析')}&assistant=1" target="_self" class="hf-launch-button">发起研究任务 <span aria-hidden="true">→</span></a>
    </section>'''


def _workspace_flow_html(own_runs: int, library_count: int) -> str:
    items = [
        ("home", "项目总览", "Homepage", "teal"),
        ("doc", "研究验证", "ST / STS", "blue"),
        ("bars", "数据分析", "研究助手 · 自主配置", "teal"),
        ("image", "影像识别", f"影像 {library_count} 条", "blue"),
        ("bulb", "模型解读", "科学结论", "blue"),
    ]
    nodes: list[str] = []
    for idx, (icon, title, sub, accent) in enumerate(items):
        nodes.append(
            f'<a href="?workspace={quote(title)}" target="_self" class="hf-workspace-node hf-workspace-{accent}"><div class="hf-workspace-icon">{_svg_icon(icon)}</div>'
            f'<div><b>{title}</b><span>{sub}</span></div></a>'
        )
        if idx < len(items) - 1:
            nodes.append('<div class="hf-workspace-arrow">→</div>')
    return f'''
    <div class="hf-card hf-workspace-panel">
      <div class="hf-card-title-row hf-title-inline"><div><div class="hf-card-title">工作区联动</div><div class="hf-card-subtitle">从数据到结论的协同工作流</div></div></div>
      <div class="hf-workspace-flow">{"".join(nodes)}</div>
    </div>'''


def _hero_and_upper_html(
    root: Path,
    best: dict[str, Any],
    norway_card: dict[str, Any],
    case_total: int,
    case_pending: int,
    case_counts: dict[str, int],
    library_count: int,
    own_runs: int,
) -> str:
    route = str(best.get("route") or "—")
    lag = best.get("lag_days")
    route_label = {"downstream":"沿流传播", "local":"局地变化", "upstream":"逆流对照"}.get(route, "传播候选")
    candidate = "顺流传播" if route == "downstream" else route_label
    candidate_note = f"上下游信号相隔约{int(lag)}天 · 合成实验" if lag is not None and str(lag).isdigit() else "合成实验中的候选线索"
    cut_date = html.escape(str(best.get("cut_date") or "—"))
    ap = _safe_float(norway_card.get("model_average_precision"), np.nan)
    ap_text = f"{ap:.3f}" if np.isfinite(ap) else "—"
    delta = _safe_float(norway_card.get("relative_improvement_over_reference"), np.nan)
    reference = _safe_float(norway_card.get('reference_average_precision'), np.nan)
    delta_note = f'参考 {reference:.3f} · <span class="hf-positive">+{delta:.1%}</span>' if np.isfinite(delta) and np.isfinite(reference) else '长期前向验证'
    banner = _asset_data_uri(root / "assets" / "home_ocean_banner.png")
    banner_style = f' style="background-image:url({banner})"' if banner else ""

    kpis = "".join([
        _kpi_card("database", "发现的传播线索", candidate, candidate_note, "blue"),
        _kpi_card("bars", "挪威 AP · 真实观测", ap_text, delta_note, "blue"),
        _kpi_card("doc", "待处理 Case", str(case_pending), "现场复核与证据更新队列", "blue"),
        _kpi_card("image", "影像记录", str(library_count), "已登记现场影像记录", "blue"),
    ])
    return f'''
    <div class="hf-home-root">
      <section class="hf-home-hero">
        <div class="hf-home-title-wrap"><h1>项目总览</h1><p>多源观测 · 科学推理 · 证据落地 · 迁移验证</p></div>
        <div class="hf-home-banner"{banner_style}></div>
      </section>
      <div class="hf-kpi-grid">{kpis}</div>
      <div class="hf-upper-grid">{_research_flow_html()}<div class="hf-overview-actions">{_case_donut_html(case_counts, case_total)}{_start_task_html()}</div></div>
      {_workspace_flow_html(own_runs, library_count)}
    </div>'''


def _base_layout(fig: go.Figure, *, height: int, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=58, r=24, t=38, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, PingFang SC, Arial", color=NAVY, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor="#dbe5ee", font=dict(size=13, color=NAVY)),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=13)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color="#526b7c", size=13), title_font=dict(color="#526b7c", size=13))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color="#526b7c", size=13), title_font=dict(color="#526b7c", size=13))
    return fig


def _agent_trace_figure(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "agent_log.csv")
    fig = go.Figure()
    if not df.empty and {"step", "pr_auc"}.issubset(df.columns):
        frame = df.sort_values("step").copy()
        fig.add_trace(go.Scatter(
            x=frame["step"], y=frame["pr_auc"], mode="lines+markers", name="探索轨迹",
            line=dict(width=2.5, color="#1975a2"),
            marker=dict(size=7.5, color="#66c6d2", line=dict(color="white", width=1.2)),
            customdata=frame[['route','lag_days','model']].to_numpy(),
            hovertemplate="试验步 %{x}<br>AP %{y:.3f}<br>路径 %{customdata[0]} · 时滞 %{customdata[1]}天<br>模型 %{customdata[2]}<extra></extra>",
        ))
        idx = pd.to_numeric(frame["pr_auc"], errors="coerce").idxmax()
        best_step = int(frame.loc[idx, "step"])
        best_ap = float(frame.loc[idx, "pr_auc"])
        fig.add_trace(go.Scatter(
            x=[best_step], y=[best_ap], mode="markers", name="当前候选",
            marker=dict(size=11, color="#168c9d", line=dict(color="white", width=1.5)),
            hovertemplate="当前候选<br>试验步 %{x}<br>AP %{y:.3f}<extra></extra>",
        ))
        fig.add_annotation(
            x=best_step, y=best_ap, text=f"当前候选<br><b>AP = {best_ap:.3f}</b>", showarrow=True,
            arrowhead=2, arrowcolor="#5d8197", ax=22, ay=-40, bgcolor="rgba(255,255,255,.97)",
            borderwidth=0, font=dict(size=13, color=NAVY),
        )
    card = _read_json(root / 'outputs' / 'discovery_card.json')
    baseline = card.get('minimum_references', {}).get('strongest_simple_baseline_pr_auc')
    if baseline is not None:
        fig.add_hline(y=float(baseline), line_dash='dot', line_color='#8798a5', line_width=1,
                      annotation_text=f'简单基线 {baseline:.3f}', annotation_position='top right',
                      annotation_font=dict(size=13, color='#526b7c'))
    fig.update_xaxes(title="试验步", dtick=1, showgrid=False)
    fig.update_yaxes(title="平均精确率（AP）", rangemode="tozero", dtick=.2)
    return _base_layout(fig, height=330, showlegend=False)


def _sa_site_heatmap(root: Path) -> go.Figure:
    raw = _read_csv(root / "data" / "real_case" / "derived" / "sa_qpcr_observations.csv")
    fig = go.Figure()
    if raw.empty or not {"sample_date", "location", "k_cristata_cells_l"}.issubset(raw.columns):
        fig.add_annotation(text="暂无南澳回放数据", x=.5, y=.5, showarrow=False)
        return _base_layout(fig, height=330, showlegend=False)
    raw = raw.copy()
    raw["sample_date"] = pd.to_datetime(raw["sample_date"], errors="coerce")
    raw["k_cristata_cells_l"] = pd.to_numeric(raw["k_cristata_cells_l"], errors="coerce").where(lambda v: v >= 0)
    # Prefer repeatedly sampled sites, then high peaks. This produces a readable event replay rather than a sparse all-site matrix.
    score = raw.groupby("location").agg(samples=("k_cristata_cells_l", "size"), peak=("k_cristata_cells_l", "max"))
    score["rank"] = score["samples"] * 2 + np.log10(score["peak"] + 1)
    sites = score.sort_values("rank", ascending=False).head(10).index.tolist()
    sub = raw[raw["location"].isin(sites) & raw["sample_date"].notna()].copy()
    sub["month_day"] = sub["sample_date"].dt.strftime("%Y-%m-%d")
    pivot = sub.pivot_table(index="location", columns="month_day", values="k_cristata_cells_l", aggfunc="max")
    if pivot.shape[1] > 18:
        keep = np.linspace(0, pivot.shape[1]-1, 18).round().astype(int)
        pivot = pivot.iloc[:, np.unique(keep)]
    row_order = [s for s in sites if s in pivot.index]
    pivot = pivot.reindex(row_order)
    raw_vals = pivot.to_numpy(float)
    z = np.log10(raw_vals + 1)
    text = np.empty(raw_vals.shape, dtype=object)
    for i in range(raw_vals.shape[0]):
        for j in range(raw_vals.shape[1]):
            v = raw_vals[i, j]
            text[i, j] = "—" if v <= 0 else (f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
    fig.add_trace(go.Heatmap(
        z=np.zeros(raw_vals.shape).tolist(), x=pivot.columns.tolist(), y=[f"S{i+1}" for i in range(len(pivot))],
        colorscale=[[0,"#E8EDF1"],[1,"#E8EDF1"]],showscale=False,xgap=2,ygap=2,
        hovertemplate="%{y} · %{x}<br>无有效浓度记录，不代表零值<extra></extra>"))
    fig.add_trace(go.Heatmap(
        z=z.tolist(), x=pivot.columns.tolist(), y=[f"S{i+1}" for i in range(len(pivot))], customdata=[[[str(site), None if not np.isfinite(v) else float(v)] for v in row] for site,row in zip(pivot.index,raw_vals)], text=text.tolist(),
        colorscale=[[0,"#e9f7f4"],[.32,"#bee4d5"],[.58,"#f4dd99"],[.8,"#ef9b61"],[1,"#dc5b43"]],
        zmin=0, zmax=max(7.2, float(np.nanmax(z)) if z.size else 7.2),
        colorbar=dict(orientation="h", thickness=8, len=1., x=.5, xanchor="center", y=1.05, tickvals=[0,3,5,7], ticktext=["0","千","十万","千万"], outlinewidth=0, tickfont=dict(size=13, color='#526b7c')),
        hovertemplate="%{customdata[0]}<br>%{x}<br>K. cristata %{customdata[1]:,.0f} cells/L<extra></extra>", hoverongaps=False,
        xgap=1, ygap=1,
    ))
    fig.update_xaxes(title="采样日期", type="category", tickangle=0, showgrid=False, tickvals=pivot.columns.tolist()[::3], ticktext=[d[5:] for d in pivot.columns.tolist()[::3]])
    fig.update_yaxes(title=None, autorange="reversed", showgrid=False, tickmode="array",tickvals=[f"S{i+1}" for i in range(len(pivot))],tickfont=dict(size=9, color="#6c8193"))
    fig = _base_layout(fig, height=330, showlegend=False)
    fig.update_layout(margin=dict(l=38,r=16,t=44,b=44))
    return fig


def _evidence_matrix(root: Path) -> go.Figure:
    lag = _read_csv(root / "outputs" / "te_cte_lag_summary.csv")
    spatial = _read_csv(root / "outputs" / "spatial_durbin_effects.csv")
    species = _read_csv(root / "outputs" / "sa_real_species_summary.csv")
    norway = _read_json(root / "outputs" / "norway_forward_benchmark_card.json")
    discovery = _read_json(root / "outputs" / "discovery_card.json")

    rows = ["温度 / MHW", "营养盐", "洋流输运", "南澳真实事件", "挪威外部验证"]
    cols = ["时滞证据", "空间溢出", "真实事件", "跨区域前向"]
    display = np.zeros((len(rows), len(cols)), dtype=float)
    text = np.full((len(rows), len(cols)), "—", dtype=object)
    hover = np.full((len(rows), len(cols)), "未在当前输出中形成独立指标", dtype=object)

    if not lag.empty and {"lag_days", "mean_cte_bits", "mean_net_directionality_bits"}.issubset(lag.columns):
        row = lag.loc[pd.to_numeric(lag["mean_cte_bits"], errors="coerce").idxmax()]
        cte = float(row["mean_cte_bits"]); net = float(row["mean_net_directionality_bits"]); lagd = int(row["lag_days"])
        display[0,0] = min(1, cte / .20); text[0,0] = f"{cte:.3f} bit"; hover[0,0] = f"最佳条件传递信息流；lag={lagd} d"
        display[2,0] = min(1, abs(net) / .16); text[2,0] = f"{net:.3f} bit"; hover[2,0] = "净方向性（正向 CTE - 反向 CTE）"

    if not spatial.empty and {"variable", "effect_type", "effect_per_1sd"}.issubset(spatial.columns):
        def _sp(variable: str) -> float:
            q = spatial[(spatial["variable"] == variable) & (spatial["effect_type"] == "indirect")]
            return float(q.iloc[0]["effect_per_1sd"]) if not q.empty else np.nan
        mhw = _sp("multiscale_anomaly_score_lag14")
        nut = _sp("nutrient_context")
        cir = _sp("circulation_residence_proxy")
        for ridx, val, label in [(0,mhw,"MHW异常间接效应"),(1,nut,"营养盐间接效应"),(2,cir,"输运/滞留间接效应")]:
            if np.isfinite(val):
                display[ridx,1] = min(1, abs(val) / .11)
                text[ridx,1] = f"{val:+.3f}"
                hover[ridx,1] = f"{label}，每 1 SD 的关联尺度效应"

    if not species.empty and {"species", "detection_share", "peak_cells_l"}.issubset(species.columns):
        q = species[species["species"].astype(str).str.contains("cristata", case=False, na=False)]
        if not q.empty:
            share = float(q.iloc[0]["detection_share"]); peak = float(q.iloc[0]["peak_cells_l"])
            display[3,2] = min(1, share)
            text[3,2] = f"检出 {share:.0%}"
            hover[3,2] = f"K. cristata 峰值 {peak:,.0f} cells/L；检出样本占比 {share:.1%}"

    ap = _safe_float(norway.get("model_average_precision"), np.nan)
    recall = _safe_float(norway.get("top10_recall"), np.nan)
    if np.isfinite(ap):
        display[4,3] = min(1, ap / .23)
        text[4,3] = f"AP {ap:.3f}"
        hover[4,3] = f"挪威长期前向验证；Top10 recall {recall:.1%}" if np.isfinite(recall) else "挪威长期前向验证"

    best_ap = _safe_float((discovery.get("best_candidate") or {}).get("pr_auc"), np.nan)
    if np.isfinite(best_ap):
        display[0,3] = min(1, best_ap / .70)
        text[0,3] = f"候选 AP {best_ap:.3f}"
        hover[0,3] = "合成留出区域前向检验的最佳候选；用于方法恢复，不替代真实外部验证"

    # Blue shade shows within-cell evidence strength only; annotations retain native units.
    fig = go.Figure(go.Heatmap(
        z=display, x=cols, y=rows, text=text, customdata=hover,
        texttemplate="%{text}", textfont=dict(size=10, color=NAVY),
        colorscale=[[0,"#f5faff"],[.001,"#f5faff"],[.35,"#d4ecfa"],[.7,"#8bc6ea"],[1,"#3a91ce"]],
        zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
    ))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=9))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=9))
    return _base_layout(fig, height=306, showlegend=False)


def _norway_heatmap(root: Path) -> go.Figure:
    df = _read_csv(root / "outputs" / "norway_forward_benchmark_folds.csv")
    fig = go.Figure()
    if df.empty or "test_window" not in df.columns:
        fig.add_annotation(text="暂无挪威前向验证结果", x=.5, y=.5, showarrow=False)
        return _base_layout(fig, height=112, showlegend=False)
    columns = df["test_window"].astype(str).tolist()
    rows = ["模型 AP", "参考模型 AP", "季节基线 AP"]
    z = np.array([
        pd.to_numeric(df.get("model_average_precision"), errors="coerce").to_numpy(float),
        pd.to_numeric(df.get("reference_average_precision"), errors="coerce").to_numpy(float),
        pd.to_numeric(df.get("seasonal_average_precision"), errors="coerce").to_numpy(float),
    ])
    text = np.array([[f"{v:.3f}" if np.isfinite(v) else "—" for v in row] for row in z], dtype=object)
    vmax = max(.25, float(np.nanmax(z)) + .01)
    fig.add_trace(go.Heatmap(
        z=z, x=columns, y=rows, text=text, texttemplate="%{text}", textfont=dict(size=10, color="#173f52"),
        colorscale=[[0,"#edf6fd"],[.35,"#d0e8f8"],[.7,"#81bde6"],[1,"#1d79c5"]],
        zmin=0, zmax=vmax, colorbar=dict(title="AP", thickness=9, len=.82, x=1.015),
        xgap=2, ygap=2, hovertemplate="%{y}<br>%{x}<br>AP %{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _base_layout(fig, height=112, showlegend=False)


def _blend_hex(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    av = tuple(int(a[i:i+2], 16) for i in (1, 3, 5))
    bv = tuple(int(b[i:i+2], 16) for i in (1, 3, 5))
    cv = tuple(round(av[k] * (1 - t) + bv[k] * t) for k in range(3))
    return "#%02x%02x%02x" % cv


def _norway_panel_html(root: Path) -> str:
    df = _read_csv(root / "outputs" / "norway_forward_benchmark_folds.csv")
    if df.empty or "test_window" not in df.columns:
        return '<div class="hf-card hf-norway-card"><div class="hf-card-title">挪威前向验证</div><div class="hf-empty">暂无前向验证结果</div></div>'
    columns = df["test_window"].astype(str).tolist()
    rows = [
        ("模型 AP", pd.to_numeric(df.get("model_average_precision"), errors="coerce").to_numpy(float)),
        ("参考模型 AP", pd.to_numeric(df.get("reference_average_precision"), errors="coerce").to_numpy(float)),
        ("季节基线 AP", pd.to_numeric(df.get("seasonal_average_precision"), errors="coerce").to_numpy(float)),
    ]
    vmax = max(.25, max(float(np.nanmax(vals)) for _, vals in rows if len(vals)))
    head = ''.join(f'<div class="hf-norway-colhead">{html.escape(c)}</div>' for c in columns)
    body = []
    for label, vals in rows:
        cells = []
        for value in vals:
            if np.isfinite(value):
                t = min(1.0, float(value) / vmax)
                bg = _blend_hex("#edf6fd", "#1d79c5", t)
                fg = "#ffffff" if t > .72 else "#123b5a"
                cells.append(
                    f'<div class="hf-norway-cell" title="{html.escape(label)} · AP {value:.3f}" '
                    f'style="background:{bg};color:{fg}">{value:.3f}</div>'
                )
            else:
                cells.append('<div class="hf-norway-cell hf-na">—</div>')
        body.append(f'<div class="hf-norway-rowlabel">{html.escape(label)}</div>' + ''.join(cells))
    return f'''
    <div class="hf-card hf-norway-card">
      <div class="hf-card-title">挪威前向验证</div>
      <div class="hf-norway-content">
        <div class="hf-norway-table">
          <div class="hf-norway-corner"></div>{head}
          {''.join(body)}
        </div>
        <div class="hf-bottom-message"><img class="hf-world-watermark" alt="世界海陆轮廓" src="{_asset_data_uri(root / "assets" / "world_outline.svg")}"><div>跨越海域的知识迁移<br><b>服务于更安全的海洋</b></div></div>
      </div>
    </div>'''

def _panel_header(title: str, action: str = "", section: str = "") -> None:
    hints={"Agent探索轨迹":"规则控制器历史参照，不是本轮大模型执行结果。每个点代表一次试验。", "南澳真实事件回放":"灰格表示无有效记录，浅绿表示0；细胞/升，对数色阶；站点和日期为展示子集。", "环境因子与迁移验证":"上图为合成机制检验；下图为独立挪威真实观测任务，两者不能合并解释为实海因果证明。"}
    tooltip=html.escape(hints.get(title,"查看对应工作区了解完整结果"))
    action_html = f'<a href="?workspace={quote("研究验证")}&section={quote(section)}" target="_self">查看详情 →</a>' if section else (f'<span>{html.escape(action)}</span>' if action else "")
    notes = {'Agent探索轨迹': '规则控制器历史参照 · 合成实验', '南澳真实事件回放': '真实观测 · 细胞/升（对数色阶）· 灰格缺测', '环境因子与迁移验证': '上：合成机制检验　下：真实观测验证'}
    note = f'<div class="hf-figure-note">{notes[title]}</div>' if title in notes else ''
    st.markdown(f'<div class="hf-viz-title"><b title="{tooltip}">{html.escape(title)} ⓘ</b>{action_html}</div>{note}', unsafe_allow_html=True)


def render(root: Any) -> None:
    # The approved overview is a full-width control board. Hide Streamlit's
    # control drawer only on this workspace; it remains available elsewhere.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {display:none !important;}
        [data-testid="stAppViewContainer"] > .main {margin-left:0 !important;}
        </style>
        <div id="globalhab-build-hf2" data-build="HF3.7.3-LLM-SCIENTIFIC-20260919"></div>
        """,
        unsafe_allow_html=True,
    )
    root = Path(root)
    discovery = _read_json(root / "outputs" / "discovery_card.json")
    norway_card = _read_json(root / "outputs" / "norway_forward_benchmark_card.json")
    best = discovery.get("best_candidate") or {}
    case_total, case_pending, _, case_counts = _case_state(root)
    library_count = _visual_library_count(root)
    own_runs = _registered_user_runs(root)

    st.markdown(
        _hero_and_upper_html(root, best, norway_card, case_total, case_pending, case_counts, library_count, own_runs),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(key="hf_agent_panel"):
            _panel_header("Agent探索轨迹", section="探索与验证")
            st.plotly_chart(_agent_trace_figure(root), use_container_width=True, config=PLOTLY_CONFIG, key="hf_agent_chart")
            
    with c2:
        with st.container(key="hf_sa_panel"):
            _panel_header("南澳真实事件回放", section="真实事件回放")
            st.plotly_chart(_sa_site_heatmap(root), use_container_width=True, config=PLOTLY_CONFIG, key="hf_sa_chart")
            
    with c3:
        with st.container(key="hf_evidence_panel"):
            _panel_header("环境因子与迁移验证", section="科学解释")
            st.plotly_chart(_combined_evidence_figure(root),use_container_width=True,config=PLOTLY_CONFIG,key='hf_evidence_combined')
            

    _render_china_panel(root)


def _render_china_panel(root: Path) -> None:
    """Paired AP comparison; every seed contributes, no cherry-picked maximum."""
    with st.container(key="china_overview", border=True):
        _panel_header("中国近海 · 跨年检验", section="真实数据训练与验证")
        rows=[]
        for marker in ['ITS1','18S_V4']:
            data=_read_csv(root/'outputs'/'china_mainland'/marker/'metrics.csv')
            if data.empty:continue
            for sea in ['黄海','东海','南海']:
                group=data[data.sea.eq(sea)]
                baseline=group[group.model.eq('物种检出率基线')]['AP']
                model=group[group.model.eq('HistGradientBoosting')]['AP']
                if len(baseline) and len(model):
                    rows.append((f'{sea} · {marker.replace("_"," ")}',float(baseline.mean()),float(model.mean()),float(model.min()),float(model.max()),int(group.iloc[0]['n'])))
        if not rows:
            st.info("尚未找到中国近海验证文件。")
            return
        cols=st.columns(len(rows),gap='medium')
        for col,(label,base,model,lo,hi,n) in zip(cols,rows):
            with col:
                gain=model-base
                color='#168d96' if gain>=0 else '#c17c55'
                fig=go.Figure(go.Bar(x=[base,model],y=['检出率基线','梯度提升'],orientation='h',
                    marker_color=['#c5d8df',color],text=[f'{base:.3f}',f'{model:.3f}'],textposition='outside',cliponaxis=False,
                    hovertemplate='%{y}<br>AP %{x:.3f}<extra></extra>'))
                fig.update_layout(title=dict(text=f'<b>{label}</b><br><span style="font-size:12px;color:{color}">较基线 {gain:+.3f}</span>',font_size=17),
                    height=220,margin=dict(l=5,r=32,t=65,b=35),showlegend=False,
                    xaxis=dict(range=[0,1],tickvals=[0,.5,1],title=dict(text='平均精确率（AP，越高越好）',font_size=13),showgrid=True,gridcolor='#edf2f3'),
                    yaxis=dict(autorange='reversed'),bargap=.48,font=dict(size=13,color=NAVY),
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
                fig.update_traces(customdata=[[lo,hi,n]]*2,hovertemplate='%{y}<br>AP %{x:.3f}<br>2019–2020训练 · 2021检验<br>模型种子范围 %{customdata[0]:.3f}–%{customdata[1]:.3f}<br>检验记录 %{customdata[2]}<br>分子检出任务；不同标记分别评估<extra></extra>')
                st.plotly_chart(fig,use_container_width=True,config=PLOTLY_CONFIG,key='china_'+label)



def _evidence_summary_html(root: Path) -> str:
    lag=_read_csv(root/'outputs/te_cte_lag_summary.csv')
    spatial=_read_csv(root/'outputs/spatial_durbin_effects.csv')
    norway=_read_json(root/'outputs/norway_forward_benchmark_card.json')
    cards=[]
    if not lag.empty:
        r=lag.loc[lag.mean_cte_bits.idxmax()]
        cards.append(f'<div class="evidence-note"><span>传播时间差 · 合成实验</span><b>{int(r.lag_days)}天 <small>方向信息最强</small></b><p>顺流 {r.mean_cte_bits:.3f} / 反向 {r.mean_reverse_cte_bits:.3f} bit<br>条件信息量，用于比较方向；不是预测准确率。</p></div>')
    if not spatial.empty:
        vals=[]
        for code,label in [('multiscale_anomaly_score_lag14','异常信号'),('nutrient_context','营养盐'),('circulation_residence_proxy','输运代理')]:
            q=spatial[(spatial.variable==code)&(spatial.effect_type=='indirect')]
            if len(q): vals.append(f'<div><span>{label}</span><b>{float(q.iloc[0].effect_per_1sd)*100:+.1f}</b></div>')
        cards.append('<div class="evidence-note"><span>邻近海区关联 · 合成实验</span><div class="effect-chips">'+''.join(vals)+'</div><p>输入增加1个标准差，对应的模型概率变化（百分点）；不是实测因果效应。</p></div>')
    if norway:
        cards.append(f'<div class="evidence-note"><span>真实数据验证 · 挪威</span><b>AP {float(norway["model_average_precision"]):.3f} <small>参考模型 {float(norway["reference_average_precision"]):.3f}</small></b><p>AP衡量风险排序，越高越好；不等于准确率。</p></div>')
    return '<div class="evidence-notes">'+''.join(cards)+'</div>'


def _combined_evidence_figure(root):
    """Two independent evidence panels share one canvas and typography."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, vertical_spacing=.30, row_heights=[.56,.44])
    lag = _read_csv(root/'outputs/te_cte_lag_summary.csv')
    if not lag.empty:
        for field, label, color in [('mean_cte_bits','顺流','#168c9d'),('mean_reverse_cte_bits','反向','#8c9ba7')]:
            fig.add_trace(go.Scatter(x=lag.lag_days, y=lag[field], mode='lines+markers', name=label,
                          line=dict(color=color,width=2,dash='solid' if label=='顺流' else 'dot'), marker=dict(size=5,symbol='circle' if label=='顺流' else 'diamond'),
                          hovertemplate=label+'：%{x}天<br>条件信息量 %{y:.3f} bit<br>合成实验<extra></extra>'), row=1,col=1)
        peak=lag.loc[lag.mean_cte_bits.idxmax()]
        fig.add_vrect(x0=float(peak.lag_days)-1.7,x1=float(peak.lag_days)+1.7,fillcolor='#d7ecee',opacity=.55,line_width=0,row=1,col=1)
        fig.update_yaxes(range=[0,float(lag.mean_cte_bits.max())*1.3],nticks=3,title=None,row=1,col=1)
        fig.add_annotation(x=float(peak.lag_days),y=float(peak.mean_cte_bits),text=f'{int(peak.lag_days)}天',
                           showarrow=False,yshift=16,font=dict(size=13,color='#16768d'),row=1,col=1)
    card=_read_json(root/'outputs/norway_forward_benchmark_card.json')
    if card and card.get('samples',0)>0:
        values=[card['top10_selected']/card['samples']*100,card['top10_recall']*100]
        for y,label,v,color in [(1,'监测样本',values[0],'#9dbeca'),(0,'覆盖事件',values[1],'#168c9d')]:
            fig.add_trace(go.Bar(x=[v],y=[y],orientation='h',width=.45,marker_color=color,
                          text=[f'{v:.0f}%'],textposition='outside',cliponaxis=False,showlegend=False,
                          textfont=dict(size=13),name=label,hovertemplate=label+' %{x:.1f}%<br>挪威真实观测 · 历史前向验证<extra></extra>'),row=2,col=1)
        fig.update_yaxes(tickvals=[1,0],ticktext=['监测样本','覆盖事件'],range=[-.6,1.6],showgrid=False,row=2,col=1)
    fig=_base_layout(fig,height=330,showlegend=True)
    fig.update_layout(uniformtext_minsize=13, uniformtext_mode='show')
    fig.update_layout(margin=dict(l=70,r=24,t=38,b=44),
                       legend=dict(orientation='h',x=1,y=1.2,xanchor='right',yanchor='bottom',font_size=13))
    fig.update_xaxes(tickvals=[0,14,30,45],ticktext=['0天','14天','30天','45天'],showgrid=False,row=1,col=1)
    fig.update_xaxes(range=[0,100],tickvals=[0,50,100],ticktext=['0%','50%','100%'],showgrid=True,row=2,col=1)
    fig.add_annotation(x=0,y=1.17,xref='paper',yref='paper',text='传播时差 · 合成（bit）',showarrow=False,xanchor='left',font=dict(size=13,color=NAVY))
    fig.add_annotation(x=0,y=.42,xref='paper',yref='paper',text='有限采样 · 挪威实测',showarrow=False,xanchor='left',font=dict(size=13,color=NAVY))
    return fig


def _evidence_figure(root,mode='lag'):
    fig=go.Figure()
    if mode=='lag':
        lag=_read_csv(root/'outputs/te_cte_lag_summary.csv')
        for field,label,color in [('mean_cte_bits','顺流','#168c9d'),('mean_reverse_cte_bits','反向','#cbb799')]:
            fig.add_trace(go.Scatter(x=lag.lag_days,y=lag[field],mode='lines+markers',name=label,line=dict(color=color,width=2.5),hovertemplate=label+'：%{x}天<br>CTE %{y:.3f} bit · 合成实验<extra></extra>'))
        r=lag.loc[lag.mean_cte_bits.idxmax()]
        fig.add_vrect(x0=r.lag_days-2,x1=r.lag_days+2,fillcolor='#d9edee',opacity=.5,line_width=0,layer='below')
        fig.update_layout(xaxis_title='候选时滞（天）',yaxis_title='条件信息量（bit）',
            legend=dict(orientation='h',x=0,y=1.2),yaxis_range=[0,lag.mean_cte_bits.max()*1.25])
    else:
        card=_read_json(root/'outputs/norway_forward_benchmark_card.json')
        values=[card['top10_selected']/card['samples']*100,card['top10_recall']*100]
        fig.add_trace(go.Bar(x=values,y=['监测样本','覆盖事件'],orientation='h',marker_color=['#bdd6df','#168c9d'],text=[f'{v:.0f}%' for v in values],textposition='outside',cliponaxis=False,
            hovertemplate='%{y} %{x:.1f}%<br>挪威真实观测 · 历史前向检验<extra></extra>'))
        fig.update_layout(xaxis=dict(range=[0,65],ticksuffix='%'),yaxis_autorange='reversed',showlegend=False,bargap=.5)
    fig.update_layout(height=235,margin=dict(l=12,r=35,t=28,b=35),font=dict(size=11,color=NAVY),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor='#edf2f3',zeroline=False)
    return fig
