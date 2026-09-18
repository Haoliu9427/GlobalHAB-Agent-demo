"""Integrated homepage for the GlobalHAB-Agent workspaces.

The homepage only reads registered outputs/session state. It does not recompute
scientific results or allow the language model layer to modify numerical evidence.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(value: Any, digits: int = 3, fallback: str = "—") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return fallback


def _workspace_jump(label: str, key: str) -> None:
    if st.button("进入工作区", key=key, use_container_width=True):
        st.session_state["_workspace_jump"] = label
        st.rerun()


def _case_summary(root: Path) -> tuple[int, int, int]:
    try:
        from globalhab_demo.case_manager import case_status_counts, list_cases

        cases = list_cases(root)
        counts = case_status_counts(root)
        pending = int(counts.get("pending_review", 0)) + int(counts.get("in_progress", 0)) + int(counts.get("visual_defer", 0))
        evidence = sum(len(c.get("evidence") or []) for c in cases)
        return len(cases), pending, evidence
    except Exception:
        return 0, 0, 0


def _visual_library_count(root: Path) -> int:
    p = root / "data" / "field_visual" / "user_library" / "records.csv"
    if not p.exists():
        return 0
    try:
        return int(len(pd.read_csv(p)))
    except Exception:
        return 0


def _own_data_status(root: Path) -> tuple[str, str]:
    current = st.session_state.get("user_result")
    if current is not None:
        return "本次会话已有结果", "可继续比较模型、下载记录或送入结果解读"
    base = root / "outputs" / "real_training"
    runs = [p for p in base.glob("*") if p.is_dir() and (p / "metrics.csv").exists()] if base.exists() else []
    if runs:
        return f"已登记 {len(runs)} 组历史任务", "当前没有新的用户上传结果"
    return "等待用户数据", "上传带时间、位置和观测标签的现场数据后开始分析"


def _llm_status() -> tuple[str, str]:
    decoded = st.session_state.get("llm_interpretation")
    if decoded:
        return "本次会话已有解读", "语言模型只读取已登记结果摘要，不重算科学指标"
    return "等待结果来源", "可读取研究结果、完整 Case、现场甄别或用户数据分析结果"


def _architecture_html() -> str:
    return """
    <div class="agent-architecture" role="img" aria-label="GlobalHAB-Agent方法与数据流框架">
      <div class="arch-stage arch-input">
        <div class="arch-kicker">01 · 输入</div>
        <div class="arch-title">多源观测与任务上下文</div>
        <div class="arch-body">环境时序 · SST/MHW · 营养盐 · 流场/输运 · HAB/qPCR · 经纬度与时间 · 现场影像 · 养殖信息 · 用户CSV/模型结果</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-stage">
        <div class="arch-kicker">02 · 感知与质控</div>
        <div class="arch-title">异常检测 + 数据质量门控</div>
        <div class="arch-body">多尺度异常、缺失/样本支持检查、现场影像质量、时空可用性。输出可计算的数据状态与候选事件。</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-stage arch-core">
        <div class="arch-kicker">03 · 科学推理内核</div>
        <div class="arch-title">Adaptive Router → ST / STS</div>
        <div class="arch-body"><b>ST</b>：Shock识别 + Transmission时滞/方向检验（TE/CTE、阻断预测）。<br><b>STS</b>：在ST基础上加入Spillover空间效应分解（Spatial Durbin）。路由器按数据条件选择可用分支，而非固定把所有方法全部执行。</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-stage arch-agent">
        <div class="arch-kicker">04 · Agent决策循环</div>
        <div class="arch-title">候选假设 → 试验 → 反馈</div>
        <div class="arch-body">在 route × lag × model 候选空间内按预算选择下一项试验；接收AP/校准/负对照等反馈，保留最有证据的候选，同时记录全过程。</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-stage">
        <div class="arch-kicker">05 · 证据落地</div>
        <div class="arch-title">真实回放 · 前向验证 · Case</div>
        <div class="arch-body">南澳/挪威/现场数据独立验证；候选区进入Case队列，与影像、实验室证据和生物响应沙盘连接。</div>
      </div>
      <div class="arch-arrow">→</div>
      <div class="arch-stage arch-output">
        <div class="arch-kicker">06 · 输出</div>
        <div class="arch-title">风险排序与可审计解释</div>
        <div class="arch-body">候选海域、时滞、证据强度、验证指标、现场复核状态、响应情景与结构化解读；保留适用边界，不自动等同于业务预报。</div>
      </div>
    </div>
    <div class="arch-feedback">
      <span>反馈回路</span>
      现场影像/实验室证据和用户数据可回写 Case 与训练库，形成后续模型更新；大模型仅负责解释，不进入数值计算闭环。
    </div>
    """


def render(root: Any) -> None:
    root = Path(root)
    discovery = _read_json(root / "outputs" / "discovery_card.json")
    norway = _read_json(root / "outputs" / "norway_forward_benchmark_card.json")
    sa = _read_json(root / "outputs" / "sa_real_replay_card.json")

    best = discovery.get("best_candidate") or {}
    case_total, case_pending, evidence_total = _case_summary(root)
    library_count = _visual_library_count(root)
    own_state, own_note = _own_data_status(root)
    llm_state, llm_note = _llm_status()

    st.markdown(
        """
        <div class="home-heading">
          <div class="home-title">项目总览</div>
          <div class="home-subtitle">把研究计算、用户数据、现场证据和结果解读放在同一条工作链上。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact, data-first status line. These are registered outputs, not live claims.
    lag = best.get("lag_days")
    route = best.get("route")
    ap = norway.get("model_average_precision")
    top10 = norway.get("top10_recall")
    st.markdown(
        f"""
        <div class="home-status-grid">
          <div class="home-stat"><span>当前方法候选</span><b>{html.escape(str(route or '—'))} / {html.escape(str(lag) + ' d' if lag is not None else '—')}</b><small>合成验证登记结果</small></div>
          <div class="home-stat"><span>挪威前向验证 AP</span><b>{_fmt(ap)}</b><small>Top 10%召回 {_fmt((float(top10) * 100) if top10 is not None else None, 1)}%</small></div>
          <div class="home-stat"><span>现场任务</span><b>{case_pending}</b><small>共 {case_total} 个Case · {evidence_total} 条证据</small></div>
          <div class="home-stat"><span>影像训练库</span><b>{library_count}</b><small>用户登记影像记录</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 四个工作区")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True, key="home_research_card"):
            st.markdown("#### 研究与验证")
            st.caption("从环境异常、时滞与空间溢出到真实事件回放和前向验证。")
            if best:
                st.markdown(
                    f"**最新登记候选：** `{best.get('route','—')}` 路径，`{best.get('lag_days','—')} 天`时滞，"
                    f"AP `{_fmt(best.get('pr_auc'))}`。"
                )
            if sa:
                st.caption(f"南澳回放：{sa.get('observations','—')} 条观测 / {sa.get('sampling_dates','—')} 个采样日。")
            _workspace_jump("研究与验证", "home_to_research")

        with st.container(border=True, key="home_vision_card"):
            st.markdown("#### 现场影像甄别")
            st.caption("把研究候选转成现场复核任务，登记影像、质量信息和实验室证据。")
            st.markdown(f"**待处理 Case：** {case_pending}　　**影像库：** {library_count}")
            st.caption("影像判断作为证据层，不直接覆盖研究模型的数值结果。")
            _workspace_jump("现场影像甄别", "home_to_vision")

    with c2:
        with st.container(border=True, key="home_own_card"):
            st.markdown("#### 自有数据分析")
            st.caption("将用户现场观测放入独立的训练/验证流程，避免与演示数据混在一起。")
            st.markdown(f"**状态：** {own_state}")
            st.caption(own_note)
            _workspace_jump("自有数据分析", "home_to_own")

        with st.container(border=True, key="home_llm_card"):
            st.markdown("#### 大模型结果解读")
            st.caption("读取已登记结果或完整 Case，形成面向科研与管理沟通的结构化说明。")
            st.markdown(f"**状态：** {llm_state}")
            st.caption(llm_note)
            _workspace_jump("大模型结果解读", "home_to_llm")

    st.markdown("### Agent 方法框架")
    st.markdown(
        "这里把 Agent 定义为一条可审计的科学工作链：**方法路由负责选择工具，ST/STS负责科学推理，Hypothesis Agent负责在有限预算下选择下一项试验，Case与现场证据负责把候选带回现实数据。**",
    )
    st.markdown(_architecture_html(), unsafe_allow_html=True)

    framework_png = root / "assets" / "GlobalHAB-Agent_method_framework.png"
    framework_svg = root / "assets" / "GlobalHAB-Agent_method_framework.svg"
    with st.expander("框架图文件（用于PPT或说明文档）", expanded=False):
        if framework_png.exists():
            st.image(str(framework_png), width="stretch")
            st.download_button(
                "下载PNG框架图", framework_png.read_bytes(),
                file_name="GlobalHAB-Agent_method_framework.png", mime="image/png",
                use_container_width=True, key="home_download_framework_png",
            )
        if framework_svg.exists():
            st.download_button(
                "下载SVG矢量框架图", framework_svg.read_bytes(),
                file_name="GlobalHAB-Agent_method_framework.svg", mime="image/svg+xml",
                use_container_width=True, key="home_download_framework_svg",
            )

    with st.expander("输入、输出与兼容范围", expanded=False):
        st.markdown(
            """
**输入层**：带时间与经纬度的表格/时序观测（CSV）、HAB或qPCR记录、SST/MHW与营养盐、流场或输运信息、现场影像、养殖情景参数，以及已经生成的CSV/JSON模型结果。Florida/HYCOM适配器内部还可解析小型NetCDF响应。

**中间对象**：异常事件表、路由诊断、TE/CTE时滞网络、空间效应表、候选假设与Agent日志、真实事件回放、Case证据链、影像训练库。

**输出层**：候选区域与时滞、风险排序、阻断验证指标、真实前向验证、现场复核状态、生物响应情景比较、可下载结果，以及基于这些已登记结果的大模型解读。

**边界**：不同数据源允许走不同分支；没有流场时不强行解释输运，没有连续标签时采用事件回放而不是伪装成监督训练。大模型不修改数值结果，也不替代现场/实验室确认。
            """
        )

    st.caption("Homepage读取现有工作区和已登记输出，不重新计算科学结果。")
