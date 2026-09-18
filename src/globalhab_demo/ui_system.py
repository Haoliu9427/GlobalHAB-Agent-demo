"""Shared presentation system for the GlobalHAB-Agent Streamlit app.

This module changes presentation only. It does not modify scientific values,
model routing, evidence states or validation logic.
"""
from __future__ import annotations

from typing import Iterable

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

BLUE = "#1C78C0"
SKY = "#4D9CE0"
TEAL = "#2CB7B1"
NAVY = "#173F52"
MUTED = "#6F8597"
BORDER = "#DBE5EE"
PANEL = "#FFFFFF"
PALE = "#F4F9FD"

COLORWAY = [BLUE, TEAL, SKY, "#6BB7E8", "#7ACFC3", "#355F88", "#9CBBD2"]


def install_plotly_theme() -> None:
    """Install one restrained chart template for the entire application."""
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", color=NAVY, size=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=COLORWAY,
            margin=dict(l=40, r=24, t=54, b=40),
            title=dict(font=dict(size=17, color=NAVY), x=0.02, xanchor="left"),
            legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color="#496579")),
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font=dict(color=NAVY, size=12)),
            xaxis=dict(
                showline=False,
                zeroline=False,
                gridcolor="rgba(45,92,122,.10)",
                tickfont=dict(color="#6A8092"),
                title_font=dict(color="#516E82"),
            ),
            yaxis=dict(
                showline=False,
                zeroline=False,
                gridcolor="rgba(45,92,122,.10)",
                tickfont=dict(color="#6A8092"),
                title_font=dict(color="#516E82"),
            ),
        )
    )
    pio.templates["globalhab_blue"] = template
    pio.templates.default = "globalhab_blue"
    px.defaults.template = "globalhab_blue"
    px.defaults.color_discrete_sequence = COLORWAY
    px.defaults.color_continuous_scale = ["#EFF7FE", "#C9E6F8", "#78BCE5", "#1C78C0"]


def render_top_navigation(options):
    from globalhab_demo.display_locale import st
    from urllib.parse import quote
    from html import escape
    options=list(options)
    requested=st.query_params.get("workspace")
    query=st.query_params.get("search", "").strip()
    if query:
        matches=[v for v in options if query.lower() in v.lower()]
        if matches:requested=matches[0]
        del st.query_params["search"]
        if not matches:st.session_state["nav_search_note"]="未找到工作区，请输入：总览、研究、数据、影像或解读。"
    pending=st.session_state.pop("_workspace_jump",None)
    if requested in options:
        pending=requested
        if "workspace" in st.query_params:del st.query_params["workspace"]
    if pending in options:st.session_state["workspace_mode"]=pending
    current=st.session_state.setdefault("workspace_mode",options[0])
    if current not in options:current=options[0]
    section=st.query_params.get("section")
    if section:
        st.session_state["research_section"]=section
        del st.query_params["section"]
    icons=[":material/home:",":material/science:",":material/query_stats:",":material/photo_camera:",":material/psychology:"]
    links=''.join(f'<a class="nav-link {"active" if v==current else ""}" href="?workspace={quote(v)}" target="_self">{escape(v)}</a>' for v in options)
    st.markdown(f'''<header class="ocean-product-nav"><a class="ocean-brand" href="?workspace={quote(options[0])}" target="_self"><svg viewBox="0 0 42 42"><path d="M3 16c12 0 14-15 27-11M3 23c14 0 17-18 34-11M3 30c15 0 17-15 34-13M9 35c12 0 16-10 26-12"/></svg><span><b>GlobalHAB-Agent</b><small>从多源观测到可审计风险研判的科学Agent</small></span></a><nav>{links}</nav><form class="ocean-nav-search" method="get"><input name="search" placeholder="搜索工作区…" aria-label="搜索工作区"/><button type="submit" aria-label="搜索"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M16 16l4.5 4.5"/></svg></button></form></header>''',unsafe_allow_html=True)
    def choose(value):
        st.session_state["workspace_mode"]=value
    with st.container(key="workspace_buttons"):
        columns=st.columns(len(options), gap="small")
        for i,(col,value) in enumerate(zip(columns,options)):
            with col:
                st.button(value,icon=icons[i],key="header_workspace_"+str(i),on_click=choose,args=(value,),width="stretch")
    active=options.index(current)
    st.markdown(f"<style>.st-key-workspace_buttons .st-key-header_workspace_{active} button {{background:#0879c7!important;color:#fff!important;}} .st-key-workspace_buttons .st-key-header_workspace_{active} button p {{color:#fff!important;}}</style>",unsafe_allow_html=True)
    if st.session_state.get("nav_search_note"):
        st.caption(st.session_state.pop("nav_search_note"))
    return current

def render_workspace_header(title: str, subtitle: str, *, kicker: str = "GlobalHAB-Agent") -> None:
    """Render the light workspace header used outside the homepage."""
    from globalhab_demo.display_locale import st

    st.markdown(
        f"""
        <section class="workspace-header">
          <div class="workspace-header-copy">
            <div class="workspace-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div class="workspace-header-ocean" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def compact_section_label(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="section-heading"><div class="section-title">{title}</div>{sub}</div>'
