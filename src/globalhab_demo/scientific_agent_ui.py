"""Research workspace for the real LLM planner; no precomputed demo impersonation."""
from .display_locale import st
from .data import generate_demo_data
from .scientific_agent import run_scientific_agent, dumps
from .real_training.remote_qwen import settings, ready, chat


def render(manual=False):
    st.markdown("### 大模型规划 · 科学工具检验")
    st.caption("合成实验｜大模型选择下一步，科学工具计算结果，人复核结论。保留原有科研页面和工作区联动。")
    if manual:
        with st.container(border=True):
            st.markdown("**自行配置 API**")
            endpoint = st.text_input("模型 API 地址", key="science_api_url", placeholder="https://服务商地址/v1")
            model = st.text_input("规划模型名称", key="science_api_model")
            key = st.text_input("API Key（仅本次会话）", type="password", key="science_api_key")
            config = {"base_url": endpoint.strip(), "model": model.strip(), "api_key": key.strip()}
            st.caption("支持兼容 Chat Completions 的模型服务。使用独立配置，不会混用网站密钥。")
    else:
        from .real_training.local_qwen_planner import QWEN_MODELS, missing_dependencies
        chosen = st.selectbox("网站提供的 Qwen 模型", list(QWEN_MODELS), key="science_local_model")
        config = {"model": QWEN_MODELS[chosen]}
        missing = missing_dependencies()
        st.caption("直接在网站服务器运行，使用数据分析工作区同一模型目录，无需 API Key。首次使用按需加载或下载权重；较大模型需要更多内存。")
        if missing:
            st.warning("当前运行环境缺少内置模型依赖：" + "、".join(missing) + "。请按部署包 requirements.txt 安装；不能以规则结果代替模型运行。")
    can_run = ready(config) if manual else not missing
    goal = st.text_area("本轮研究问题", value="比较本地与上游环境关联及不同滞后；检查反方向和时间打乱后线索是否仍成立，保留不支持假设的证据。", key="science_goal")
    budget = st.slider("本轮实验预算（候选或单项负对照各计一次）", 4, 12, 8, key="science_budget")
    with st.expander("实验设置", expanded=False):
        seed = st.number_input("合成数据种子", 1, 9999, 42, key="science_seed")
    st.info("每次检验才计算对应方案，不预先跑完24个候选。先用验证集探索，方案冻结后再做独立测试；不保证找到预设答案，也不把关联当作因果。")
    if manual and not ready(config):
        st.warning("配置尚不完整，暂时无法运行大模型检验。请选择已配置的网站模型，或切换到“自行配置 API”填写完整信息。")
    if st.button("开始大模型规划与检验", type="primary", disabled=not can_run, key="science_run"):
        st.session_state.pop("science_result", None)
        progress = st.empty()
        try:
            if not manual:
                progress.info("正在加载网站 Qwen 模型；首次使用可能需要下载权重……")
                from .real_training.local_qwen_planner import load_model, _LOCK
                with _LOCK:
                    load_model(config["model"])
            def planner(messages):
                if manual:
                    return chat(config, messages, max_tokens=1000, thinking_mode="stable")
                from .real_training.local_qwen_planner import chat_local
                return chat_local(messages, model_id=config["model"])
            result = run_scientific_agent(generate_demo_data(days=720, seed=int(seed)), planner,
                goal=goal, budget=int(budget), seed=int(seed), model_label=config["model"], notify=progress.info)
            st.session_state["science_result"] = result
        except Exception:
            st.error("本轮未完成。请检查模型依赖、权重下载、服务器内存或 API 配置；没有生成替代结果。")
        finally:
            progress.empty()
    result = st.session_state.get("science_result")
    if not result:
        st.markdown("**运行后会看到：** 检验目的 → 调用的工具与参数 → 实测反馈 → 下一步决策 → 冻结后的独立测试。")
        return
    st.caption(f"当前显示已完成运行：模型 {result['model']} · 种子 {result['seed']} · 预算 {result['experimental_budget']}。修改设置不会改写这份结果。")
    a, b, c = st.columns(3)
    a.metric("实际实验", f"{result['experiments_executed']} / {result['experimental_budget']}")
    b.metric("模型规划调用", len(result["audit"]))
    c.metric("已报告 token", result["known_total_tokens"] if result["token_usage_complete"] else "未完整报告")
    st.caption(f"耗时 {result['elapsed_seconds']:.1f} 秒；另含季节参照拟合 {result['reference_fits']} 次、冻结后最终拟合 {result['final_fits']} 次。没有宣称节省时间或优于其他搜索方法。")
    final = result.get("final")
    if final:
        st.success("实验流程已完成 · " + final["conclusion"])
        st.write(f"冻结方案：{final['action_id']}；独立测试 AP {final['candidate']['pr_auc']:.3f}，季节参照 AP {final['seasonal_reference']['pr_auc']:.3f}。")
        st.caption(final["limit"])
        from .result_pool import register
        register("探索与验证", "大模型规划与科学检验（合成）", {"final": final, "model": result["model"], "experiments": result["experiments_executed"]})
    else:
        st.warning("本轮未形成完成复核的冻结方案，保留已有记录。状态：" + result["status"])
    for row in result["audit"]:
        with st.container(border=True):
            st.markdown(f"**第 {row['step']} 步 · {row.get('tool', '模型服务')}**")
            st.write(row.get("rationale", row.get("message", "调用格式未通过校验")))
            if row.get("arguments"):
                st.caption("参数：" + dumps(row["arguments"]))
            with st.expander("查看工具返回的证据 · " + row["status"]):
                st.json(row.get("result", {}))
    st.download_button("下载本轮可审计实验记录", dumps(result).encode("utf-8"), "scientific_agent_audit.json", "application/json", key="science_download")
