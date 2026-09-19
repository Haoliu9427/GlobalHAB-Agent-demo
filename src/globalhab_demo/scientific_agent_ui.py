"""Server-configured hosted planning; no local model or rule fallback."""
import time
from .display_locale import st
from .data import generate_demo_data
from .scientific_agent import run_scientific_agent, dumps
from .real_training.remote_qwen import ready, chat
from .platform_models import catalog, connection_id, probe


def _publish_scientific_result(result, manual=False):
    """Publish evidence through the stable session record schema, not a cached
    helper import. No provider credentials are accepted or copied here."""
    import json
    import hashlib
    import datetime
    fields = ("model", "goal", "status", "seed", "data_sha256", "experimental_budget",
              "experiments_executed", "known_total_tokens", "elapsed_seconds",
              "candidates", "controls", "final")
    payload = {key: result.get(key) for key in fields}
    payload["planning_record"] = [{key: row[key] for key in
        ("step", "tool", "arguments", "rationale", "status") if key in row}
        for row in result.get("audit", [])]
    payload["scope"] = "合成实验；验证阶段与独立测试分别记录。不得把候选相关性当作因果，未完成的检验不能视为通过。"
    text = json.dumps(payload, ensure_ascii=False, default=str)
    source = "自主接入" if manual else "模型驱动"
    kind = "大模型规划与科学检验"
    rid = hashlib.sha256((source + kind + text).encode()).hexdigest()[:16]
    pool = st.session_state.setdefault("result_pool", {})
    if rid not in pool:
        pool[rid] = {"id": rid, "source": source, "evidence_type": kind,
                     "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                     "summary": text, "data_type": "模拟数据"}
        while len(pool) > 30:
            pool.pop(next(iter(pool)))
        st.session_state["_llm_source_jump"] = "本会话结果汇总"
        st.session_state["_pool_selection_jump"] = rid
        # Older result-pool renderers also honor this established widget key.
        st.session_state["pool_selection"] = [rid]
    return rid


def connection_error_hint(exc):
    """Describe known failures without reflecting provider bodies or credentials."""
    import re
    text = str(exc)
    match = re.search(r"HTTP\s*(\d{3})", text)
    if match:
        code = match.group(1)
        reasons = {
            "400": "请求参数不被服务接受，请核对模型名称及 Chat Completions 兼容性。",
            "401": "密钥无效或已失效，请使用当前服务商签发的 API Key。",
            "402": "账户余额或可用额度不足，请在服务商控制台检查。",
            "403": "当前账户没有调用权限，请检查服务是否开通及模型授权。",
            "404": "接口路径或模型不存在，请核对 API 地址和模型名称。",
            "429": "调用过于频繁或额度受限，请稍后重试并检查服务额度。",
        }
        reason = reasons.get(code, "服务暂时不可用，请稍后重试或检查服务商状态。")
        return "连接未通过（HTTP " + code + "）：" + reason
    if "HTTPS" in text or "内网" in text or "本地" in text:
        return "连接未通过：请填写公网 HTTPS 服务地址，不能使用本地或内网地址。"
    if "解析" in text and "域名" in text:
        return "连接未通过：无法解析 API 域名，请检查地址拼写。"
    if "超时" in text or "网络" in text:
        return "连接未通过：请求超时或网络不可达，请稍后重试。"
    if "结构" in text or "空的" in text or "响应" in text:
        return "连接未通过：服务未返回可用的 Chat Completions 文本，请核对接口和模型。"
    return "连接未通过：服务未返回有效响应，请检查接口兼容性或稍后重试。"


def render(manual=False):
    st.markdown("### " + ("自主接入 · 智能研究" if manual else "模型驱动 · 智能研究"))
    st.caption("合成实验 · 模型提出下一步，工具计算证据，你复核结论。")
    st.caption("页面版本：" + UI_REVISION)
    config = {}
    with st.container(border=True):
        st.markdown("**01　选择模型**")
        check_requested = False
        if manual:
            with st.form("science_manual_connection"):
                endpoint = st.text_input("模型 API 地址", key="science_api_url", placeholder="https://api.deepseek.com")
                model = st.text_input("模型名称", key="science_api_model", placeholder="例如 deepseek-flash")
                key = st.text_input("API Key（仅本次会话）", type="password", key="science_api_key")
                st.caption("填写后点击下方按钮提交并验证，无需在 Secrets 中重复配置。使用该服务商自己的密钥和额度。")
                check_requested = st.form_submit_button("保存并检查连接", icon=":material/wifi:")
            config = {"base_url": endpoint.strip(), "model": model.strip(), "api_key": key.strip()}
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
        if not manual and ready(config):
            check_requested = st.button("重新检查连接" if connected else "检查模型连接", key="science_probe", icon=":material/wifi:")
        if check_requested:
            st.session_state.pop("science_connection", None)
            connected = False
            missing = [label for field, label in (("base_url", "模型 API 地址"), ("model", "模型名称"), ("api_key", "API Key")) if not config.get(field)]
            if missing:
                st.error("请填写：" + "、".join(missing) + "。无需添加 Secrets。")
            else:
                try:
                    with st.spinner("正在请求模型确认连接……"):
                        probe(config)
                    st.session_state["science_connection"] = {"id": fingerprint, "at": time.time()}
                    connected = True
                except Exception as exc:
                    st.error(connection_error_hint(exc))
        if connected:
            st.success("模型连接已通过 · 可以开始研究")
        elif ready(config):
            st.caption("配置已填写，尚未通过连接检查。请先检查连接，再启动研究。")
        elif manual:
            st.caption("请填写三项配置，然后点击“保存并检查连接”。")
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
    _publish_scientific_result(result, manual=manual)
    st.caption("本轮结果已同步至“模型解读”，打开后可直接选取并解读。")
    from .release_ui import load_view
    render_results = load_view("scientific_agent_results", UI_REVISION).render_results
    render_results(result)
    st.download_button("下载本轮可审计实验记录", dumps(result).encode("utf-8"), "scientific_agent_audit.json", "application/json", key="science_download")

UI_REVISION = 'HF3.9.13-LIVE-CAMERA-20260920'
