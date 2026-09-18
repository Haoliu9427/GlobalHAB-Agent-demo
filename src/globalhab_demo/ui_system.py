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


def render_top_navigation(options: Iterable[str]) -> str:
    """Render a button-based product header and return the selected workspace.

    Buttons are used instead of ``st.radio`` so the navigation does not expose
    browser- or Streamlit-version-specific radio circles. This makes the live
    application match the approved dashboard composition more reliably.
    """
    import streamlit as st

    options = list(options)
    pending = st.session_state.pop("_workspace_jump", None)
    if pending in options:
        st.session_state["workspace_mode"] = pending
    if st.session_state.get("workspace_mode") not in options:
        st.session_state["workspace_mode"] = options[0]

    current = st.session_state["workspace_mode"]
    labels = {
        "项目总览": "▣  项目总览",
        "研究与验证": "▤  研究与验证",
        "自有数据分析": "▥  自有数据分析",
        "现场影像甄别": "▧  现场影像甄别",
        "大模型结果解读": "✺  大模型结果解读",
    }
    nav_keys = {
        "项目总览": "nav_home",
        "研究与验证": "nav_research",
        "自有数据分析": "nav_data",
        "现场影像甄别": "nav_visual",
        "大模型结果解读": "nav_llm",
    }

    def _select_workspace(value: str) -> None:
        st.session_state["workspace_mode"] = value

    with st.container(key="global_top_nav"):
        cols = st.columns([1.68, .78, .92, 1.02, 1.08, 1.20, .58], gap="small", vertical_alignment="center")
        with cols[0]:
            st.markdown(
                """
                <div class="top-brand-wrap">
                  <div class="top-brand-mark" aria-hidden="true">
                    <svg viewBox="0 0 42 42" role="img">
                      <path d="M4 15c7 0 7-7 14-7s7 7 14 7 7-7 7-7"/>
                      <path d="M4 22c7 0 7-7 14-7s7 7 14 7 7-7 7-7"/>
                      <path d="M4 29c7 0 7-7 14-7s7 7 14 7 7-7 7-7"/>
                    </svg>
                  </div>
                  <div>
                    <div class="top-brand-title">GlobalHAB-Agent</div>
                    <div class="top-brand-sub">从多源观测到可审计风险研判的科学 Agent</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for col, value in zip(cols[1:6], options):
            with col:
                st.button(
                    labels.get(value, value),
                    key=nav_keys[value],
                    use_container_width=True,
                    on_click=_select_workspace,
                    args=(value,),
                )
        with cols[6]:
            st.markdown(
                '<div class="top-brand-side">DATA<br>SCIENCE<br>FOR A HEALTHY OCEAN</div>',
                unsafe_allow_html=True,
            )

    active_key = nav_keys[current]
    st.markdown(
        f"""
        <style>
        .st-key-{active_key} button {{
          background:#1677c5 !important;
          border-color:#1677c5 !important;
          color:#fff !important;
          box-shadow:0 2px 6px rgba(29,121,197,.16) !important;
        }}
        .st-key-{active_key} button p {{color:#fff !important;}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.session_state["workspace_mode"]


def render_workspace_header(title: str, subtitle: str, *, kicker: str = "GlobalHAB-Agent") -> None:
    """Render the light workspace header used outside the homepage."""
    import streamlit as st

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
