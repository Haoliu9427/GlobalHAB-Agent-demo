"""Server-configured hosted planning; no local model or rule fallback."""
import time
from .display_locale import st
from .data import generate_demo_data
from .scientific_agent import run_scientific_agent, dumps
from .real_training.remote_qwen import ready, chat
from .platform_models import catalog, connection_id, probe


def render(manual=False):
    st.markdown("### " + ("自主接入 · 智能研究" if manual else "模型驱动 · 智能研究"))
    st.caption("合成实验 · 模型提出下一步，工具计算证据，你复核结论。")
    config = {}
    with st.container(border=True):
        st.markdown("**01　选择模型**")
        if manual:
            endpoint = st.text_input("模型 API 地址", key="science_api_url", placeholder="https://服务商地址/v1")
            model = st.text_input("模型名称", key="science_api_model")
            key = st.text_input("API Key（仅本次会话）", type="password", key="science_api_key")
            config = {"base_url": endpoint.strip(), "model": model.strip(), "api_key": key.strip()}
            st.caption("使用你自己的服务配置和额度。内置大模型的密钥不会填入此处。")
        else:
            entries = catalog()
            if entries:
                by_id = {e["id"]: e for e in entries}
                if st.session_state.get("science_platform_model") not in by_id:
                    st.session_state.pop("science_platform_model", None)
                selected = st.selectbox("选择大模型", list(by_id), format_func=lambda k: by_id[k]["label"], key="science_platform_model")
                config = by_id[selected]["config"]
                st.caption("已配置大模型，可直接选择使用。")
            else:
                st.info("大模型尚未配置。请由网站维护者在 Streamlit Secrets 添加 [modelscope] 和 api_key；保存后自动显示三个 Qwen 模型。访客无需填写密钥。")
        fingerprint = connection_id(config)
        verified = st.session_state.get("science_connection", {})
        connected = verified.get("id") == fingerprint and time.time()-verified.get("at", 0) < 600
        if ready(config):
            if st.button("重新检查连接" if connected else "检查模型连接", key="science_probe", icon=":material/wifi:"):
                try:
                    with st.spinner("正在请求模型确认连接……"):
                        probe(config)
                    st.session_state["science_connection"] = {"id": fingerprint, "at": time.time()}
                    connected = True
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("Model connection check: %s", str(exc).replace(config.get("api_key", "NEVER_MATCH"), "[REDACTED]"))
                    st.session_state.pop("science_connection", None)
                    connected = False
                    st.error("模型连接未通过。请由服务配置者检查权限、额度或服务状态；本次没有启动研究。")
            if connected:
                st.success("模型连接已通过 · 研究结果仍需本轮实际检验")
            else:
                st.caption("服务已配置，尚未验证当前连接。先检查连接，再启动研究。")
    from .planning_models import inventory
    import pandas as pd
    model_rows = inventory()
    with st.expander(f"预测模型目录 · {len(model_rows)} 个条目 / {sum(r['available'] for r in model_rows)} 个适用于本轮"):
        st.dataframe(pd.DataFrame([{"模型":r["name"],"本轮状态":r["reason"]} for r in model_rows]),hide_index=True,width="stretch")
        st.caption("大模型先选2–3种预测模型，再调用实验；可根据反馈补充候选。STS交互模型复用现有交互特征，适配当前合成信号；STS-Gated TCN尚未接入此规划任务。")
    with st.container(border=True):
        st.markdown("**02　设置研究任务**")
        goal = st.text_area("你希望检验什么？", value="先比较简单预测模型、STS交互结构模型与树模型，再探索本地和上游关联及不同滞后；检查反方向和时间打乱对照，保留不支持假设的证据。", key="science_goal", height=104)
        budget = st.slider("最多进行多少次实验？", 4, 12, 8, key="science_budget", help="候选方案或单项负对照各计一次。")
        with st.expander("实验设置与数据范围"):
            seed = st.number_input("合成数据种子", 1, 9999, 42, key="science_seed")
            st.caption("每次检验才计算对应方案。验证集用于探索；方案冻结后再做独立测试。此处使用合成数据，不代表真实海域预警效果。")
        start = st.button("开始规划与科学检验", type="primary", disabled=not (ready(config) and connected), key="science_run", icon=":material/play_arrow:")
    result_key = "science_result_" + ("manual_" if manual else "platform_") + fingerprint
    if start:
        st.session_state.pop(result_key, None)
        progress = st.empty()
        try:
            def planner(messages):
                return chat(config, messages, max_tokens=1000, thinking_mode="stable")
            result = run_scientific_agent(generate_demo_data(days=720, seed=int(seed)), planner,
                goal=goal, budget=int(budget), seed=int(seed), model_label=config["model"], notify=progress.info)
            st.session_state[result_key] = result
        except Exception:
            st.error("本轮未完成，请稍后重试或检查模型服务。没有生成替代结果。")
        finally:
            progress.empty()
    st.markdown("### 03　检验过程与证据")
    result = st.session_state.get(result_key)
    if not result:
        st.markdown("**运行后会看到：** 检验目的 → 调用的工具与参数 → 实测反馈 → 下一步决策 → 冻结后的独立测试。")
        return
    st.caption(f"当前显示运行记录：模型 {result['model']} · 种子 {result['seed']} · 预算 {result['experimental_budget']}。修改设置不会改写这份结果。")
    a, b, c = st.columns(3)
    a.metric("实际实验", f"{result['experiments_executed']} / {result['experimental_budget']}")
    b.metric("模型规划调用", len(result["audit"]))
    c.metric("已报告 token", result["known_total_tokens"] if result["token_usage_complete"] else "未完整报告")
    st.caption(f"耗时 {result['elapsed_seconds']:.1f} 秒；另含季节参照拟合 {result['reference_fits']} 次、冻结后最终拟合 {result['final_fits']} 次。没有宣称节省时间或优于其他搜索方法。")
    final = result.get("final")
    if final:
        if final["conclusion"] == "证据不足，保留负结果":
            st.warning("实验流程已完成 · " + final["conclusion"])
        else:
            st.success("实验流程已完成 · " + final["conclusion"])
        st.write(f"冻结方案：{final['action_id']}；独立测试 AP {final['candidate']['pr_auc']:.3f}，季节参照 AP {final['seasonal_reference']['pr_auc']:.3f}。")
    else:
        st.warning("本轮未形成完成复核的冻结方案，保留已有记录。状态：" + result["status"])
    from .result_pool import publish_scientific_result
    publish_scientific_result(result)
    st.caption("本轮结果已同步至“模型解读”，打开后可直接选取并解读。")
    from .scientific_agent_results import render_results
    render_results(result)
    st.download_button("下载本轮可审计实验记录", dumps(result).encode("utf-8"), "scientific_agent_audit.json", "application/json", key="science_download")
