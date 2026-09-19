"""Shared rules-style presentation of actual LLM experiment evidence."""
from .planning_models import MODEL_NAMES, inventory
import pandas as pd
import plotly.graph_objects as go
from .display_locale import st
from .research_figures import baseline_comparison, exploration_trace, finish, TEAL, PALE, CORAL
from .research_panels import research_plot, research_table


def candidate_label(aid):
    parts = aid.split("__")
    if len(parts) != 3:
        return aid
    return " · ".join([{"local": "本地", "downstream": "顺流"}.get(parts[0],parts[0]), parts[1].replace("d", "天"), {"logistic": "逻辑回归", "random_forest": "随机森林"}.get(parts[2], MODEL_NAMES.get(parts[2],parts[2]))])


def evidence_frames(result):
    trace = []
    for row in result.get("audit", []):
        if row.get("status") == "executed" and row.get("tool") == "evaluate_hypothesis":
            feedback = row.get("result", {})
            if "pr_auc" in feedback:
                trace.append({"step": row["step"], "pr_auc": feedback["pr_auc"], "方案": candidate_label(feedback["action_id"]), "决策理由": row.get("rationale", "")})
    final = result.get("final")
    comparison = []
    if final:
        comparison = [{"方法": "季节参照", **final["seasonal_reference"]}, {"方法": "Agent冻结候选", **final["candidate"]}]
    return pd.DataFrame(trace), pd.DataFrame(comparison)


def control_figure(result, aid):
    candidates = {r["action_id"]: r for r in result.get("candidates", [])}
    candidate = candidates[aid]
    controls = [r for r in result.get("controls", []) if r["action_id"] == aid]
    rows = [("原候选", candidate["pr_auc"], TEAL)]
    names = {"reversed": "反方向对照", "permutation": "时间打乱对照"}
    rows += [(names.get(r["control"], r["control"]), r["pr_auc"], CORAL) for r in controls]
    baseline = result.get("protocol", {}).get("validation_seasonal_reference", {}).get("pr_auc")
    if baseline is not None:
        rows.append(("季节参照", baseline, PALE))
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h", marker_color=[r[2] for r in rows], text=[r[1] for r in rows], texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>验证集 AP %{x:.3f}<extra></extra>"))
    fig.update_layout(title="同一验证集 · 候选与负对照", showlegend=False, bargap=.46, xaxis=dict(title="AP ↑", range=[0,max(.1,max(r[1] for r in rows)*1.3)]), yaxis_autorange="reversed")
    return finish(fig,350)


def render_results(result):
    trace, comparison = evidence_frames(result)
    selections=[row for row in result.get("audit",[]) if row.get("tool")=="select_models" and row.get("status")=="executed"]
    if selections:
        with st.container(border=True,key="research_section_llm_plan"):
            st.markdown("### 模型选择与研究计划")
            for row in selections:
                labels=[MODEL_NAMES.get(mid,mid) for mid in row["arguments"]["model_ids"]]
                st.markdown(f"**第 {row['step']} 步 · " + " / ".join(labels) + "**")
                st.write(row.get("rationale", ""))
            st.caption("以上为模型实际提出的候选；最终结论取决于工具返回的验证证据。")
    with st.container(border=True, key="research_section_llm_comparison"):
        st.markdown("### 实验对照与探索记录")
        candidates = result.get("candidates", [])
        aid = None
        if candidates:
            ids = [r["action_id"] for r in candidates]
            frozen = (result.get("final") or {}).get("action_id")
            aid = st.selectbox("选择要查看负对照的候选方案", ids,
                index=ids.index(frozen) if frozen in ids else 0, format_func=candidate_label)
        final_fig = baseline_comparison(comparison) if not comparison.empty else None
        control_fig = control_figure(result, aid) if aid else None
        figures = [fig for fig in (final_fig, control_fig) if fig is not None]
        maximum = max([float(v) for fig in figures for bar in fig.data for v in bar.x] or [0.1])
        for fig in figures:
            fig.update_layout(height=370, margin=dict(l=130, r=52, t=68, b=58),
                              title=dict(x=0.02, xanchor="left"), bargap=0.46)
            fig.update_xaxes(range=[0, max(0.1, maximum * 1.25)], title="平均精确率 AP ↑", automargin=False)
            fig.update_yaxes(automargin=False)
        left, right = st.columns(2, gap="large")
        with left:
            if final_fig is not None:
                research_plot(final_fig, width="stretch", config={"displayModeBar":False}, key="llm_final_comparison")
                st.caption("独立测试集 · 冻结方案与季节参照；AP 越高，风险排序越好。")
            else:
                st.info("尚未完成冻结后的独立测试，暂不显示最终性能对比。")
        with right:
            if control_fig is not None:
                research_plot(control_fig, width="stretch", config={"displayModeBar":False}, key="llm_control_comparison")
                done = {r["control"] for r in result.get("controls",[]) if r["action_id"] == aid}
                missing = [{"reversed":"反方向","permutation":"时间打乱"}[k] for k in ("reversed","permutation") if k not in done]
                st.caption("验证集 · 尚未执行："+"、".join(missing) if missing else "验证集 · 两种负对照均已执行；单次对照不代表因果证明。")
            else:
                st.info("尚无已完成的候选实验，暂不显示对照图。")
        if not comparison.empty:
            research_table(comparison[["方法","pr_auc","brier_skill","ece"]].rename(columns={"pr_auc":"AP","brier_skill":"Brier技能分数","ece":"校准误差"}), hide_index=True, width="stretch")
    with st.container(border=True,key="research_section_llm_trace"):
        st.markdown("#### 完整探索轨迹")
        st.caption("只绘制实际执行的候选方案；横轴为模型规划调用步，负对照和查看数据的步骤不计入最佳候选曲线。")
        if not trace.empty:
            fig=exploration_trace(trace)
            fig.update_xaxes(title="模型规划调用步",dtick=1)
            fig.update_yaxes(title="验证集 AP ↑")
            research_plot(fig,width="stretch",config={"displayModeBar":False},key="llm_exploration_trace")
            research_table(trace.rename(columns={"step":"规划调用步","pr_auc":"验证集 AP"}),width="stretch",hide_index=True)
        else:
            st.info("尚无候选实验反馈；不会用示例曲线代替本轮结果。")
    with st.container(border=True,key="research_section_llm_audit"):
        st.markdown("#### 科学工具调用记录")
        names={"select_models":"选择预测模型", "inspect_data":"查看数据","evaluate_hypothesis":"检验候选","negative_control":"执行负对照","finish":"冻结方案"}
        statuses={"executed":"已执行","rejected":"调用被拒绝","model_service_error":"服务调用失败"}
        for row in result.get("audit",[]):
            title=f"第 {row['step']} 步 · {names.get(row.get('tool'), '模型服务')} · {statuses.get(row['status'],row['status'])}"
            with st.expander(title):
                st.write(row.get("rationale",row.get("message","调用格式未通过校验")))
                st.json({"参数":row.get("arguments",{}),"工具证据":row.get("result",{})})
