from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
FIELD = ROOT / "src" / "globalhab_demo" / "field_visual.py"
LLM = ROOT / "src" / "globalhab_demo" / "real_training" / "result_interpreter.py"


def test_case_evidence_function_not_shadowed_in_app():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "evidence_rows":
                    bad.append(getattr(node, "lineno", None))
    assert not bad, f"evidence_rows function is shadowed at lines {bad}"
    text = APP.read_text(encoding="utf-8")
    assert "sandbox_evidence_rows" in text
    assert "rows = evidence_rows(current_case)" in text


def test_case_context_cards_use_project_kpi_style():
    app = APP.read_text(encoding="utf-8")
    field = FIELD.read_text(encoding="utf-8")
    assert 'key="field_task_from_research"' in app
    assert 'kpi_grid([' in app
    assert 'key="vision_case_context"' in field
    assert '_render_context_kpis([' in field


def test_workspace_hero_copy_regression():
    app = APP.read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    assert "OWN DATA WORKBENCH" in app
    assert "<h1>自有数据分析</h1>" in app
    assert "让大模型解释已经计算完成的科学结果，而不是替代模型计算" not in llm


def test_batch_review_and_case_lifecycle_ui_present():
    app = APP.read_text(encoding="utf-8")
    field = FIELD.read_text(encoding="utf-8")
    assert "批量生成现场复核任务并前往任务队列" in app
    assert "Case / 现场任务" in app
    assert "永久删除Case" in app
    assert "取消复核" in app and "归档" in app
    assert "登记并处理下一个" in field
    assert "待复核任务队列" in field


def test_visual_workspace_tabs_and_learning_cards_are_consistent():
    field = FIELD.read_text(encoding="utf-8")
    learning = (ROOT / "src" / "globalhab_demo" / "visual_learning.py").read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert '"现场影像甄别"' in field and '"我的影像数据"' in field and '"模型训练与版本"' in field
    assert '① 现场影像甄别' not in field
    assert '② 我的影像数据' not in field
    assert '③ 模型训练与版本' not in field
    assert '_render_learning_kpis' in learning
    assert 'vision_library_summary' in learning
    assert 'vision_training_summary' in learning
    assert 'vision_library_recent_card' in learning
    assert 'vision_training_candidate_card' in learning
    assert '.st-key-vision_workspace_tabs' in css


def test_own_data_hero_and_sidebar_polish_regression():
    app = APP.read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert '同一套数据完成质量检查、模型比较、时间留出与可选未来预测；所有结果与项目固定证据分开保存。' not in app
    assert 'sidebar-brand-card' in app
    assert 'sidebar-section-title' in app
    assert 'label_visibility="collapsed"' in app
    assert '.sidebar-brand-card' in css
    assert '.st-key-workspace_nav [role="radiogroup"]' in css


def test_workspace_copy_and_balanced_card_layout_regression():
    field = FIELD.read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert "### 01 · 拍照或上传" not in field
    assert "### 02 · 现场信息" not in field
    assert "### 拍照或上传" in field
    assert "### 现场信息" in field
    assert "科学边界：照片用于水色、浑浊、泡沫/漂浮物等视觉现象筛查与复核优先级" not in field
    assert '<p class="tagline">可读取项目固定证据' in llm
    assert 'st.columns([1, 1], gap="large")' in llm
    assert "st.columns([1,1],gap='large')" in workbench
    assert '.camera-permission-cn' in css
    assert '此应用需要使用您的摄像头。' in css
    assert ':has(.st-key-vision_input_card)' in css
    assert ':has(.st-key-obs_upload_card)' in css
    assert ':has(.st-key-llm_source_card)' in css


def test_strict_equal_height_cards_and_compact_remote_service():
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert "endpoint_col, model_col = st.columns([1.25, 1], gap='small')" in workbench
    assert "action_a, action_b, action_c = st.columns(3, gap='small')" in workbench
    assert "远程调用与隐私说明" in workbench
    assert "height=108" in workbench
    assert "height=100" in llm
    assert '.st-key-obs_service_card,.st-key-obs_models_card{min-height:690px!important;}' in css
    assert '[data-testid="stColumn"]:has(.st-key-llm_source_card)' in css
    assert '[data-testid="stVerticalBlockBorderWrapper"]' in css


def test_content_density_and_user_facing_llm_prompt_examples():
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    assert "为什么AP不高但仍有价值" not in llm
    assert "请指出最值得进一步复核的证据" in llm
    assert "摘要预览" in llm
    assert "下载摘要" in llm
    assert "compact-card-footer" in llm
    assert "字段与数据要求" in workbench
    assert "本次分析说明" in workbench
    assert "远程调用与隐私说明" in workbench
    # Old verbose permanent headings should remain removed from the main card surface.
    for text in ["#### 当前输入", "#### 输出结构", "#### 必要信息", "#### 本次输出", "#### 连接状态", "#### 远程发送范围", "#### 运行前检查", "#### 分析记录"]:
        assert text not in llm + workbench


def test_final_pair_balance_and_matched_upload_actions():
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert "prep_a, prep_b = st.columns([1, 1], gap='small')" in workbench
    assert "real_training_upload_status" in workbench
    assert "字段与数据要求" in workbench
    assert "本次分析说明" in workbench
    assert "原始文件默认不发送" in llm
    assert "45,000字符" in llm
    assert 'grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;' in css


def test_parent_owned_equal_height_cards_and_content_fill():
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert "st.container(border=False,key='obs_upload_card')" in workbench
    assert "st.container(border=False,key='obs_task_card')" in workbench
    assert 'st.container(border=False, key="llm_source_card")' in llm
    assert 'st.container(border=False, key="llm_mode_card")' in llm
    assert "compact-card-footer" in workbench
    assert "compact-card-footer" in llm
    assert '[data-testid="stColumn"]:has(.st-key-obs_upload_card)' in css
    assert '[data-testid="stColumn"]:has(.st-key-llm_source_card)' in css
    assert 'compact balanced cards' in css
    assert 'min-height:0!important;' in css


def test_compact_balanced_copy_and_no_oversized_final_floor():
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    # Main cards stay concise; detailed guidance moves into collapsed expanders.
    assert "字段与数据要求" in workbench
    assert "本次分析说明" in workbench
    assert "查看完整结果摘要" in llm
    assert 'expanded=False' in llm
    assert "#### 当前输入" not in llm
    assert "#### 发送设置" not in llm
    assert "#### 输出结构" not in llm
    assert "#### 必要信息" not in workbench
    assert "#### 自动检查" not in workbench
    assert "#### 验证规则" not in workbench
    assert "compact balanced cards" in css
    assert "min-height:0!important;" in css
    assert ".compact-card-footer" in css


def test_remote_service_defaults_and_bottom_run_alignment_regression():
    workbench = (ROOT / "src" / "globalhab_demo" / "real_training" / "user_workbench.py").read_text(encoding="utf-8")
    css = (ROOT / "assets" / "interface.css").read_text(encoding="utf-8")
    assert "'DeepSeek':{'base_url':'https://api.deepseek.com','model':'deepseek-flash'}" in workbench
    assert "st.session_state['user_api_model']=preset['model']" in workbench
    assert "card-flex-spacer" in workbench
    assert ':has(.st-key-obs_service_card)' in css
    assert '.st-key-obs_models_card [data-testid="stElementContainer"]:has(.card-flex-spacer)' in css


def test_llm_source_card_uses_space_for_summary_preview_regression():
    llm = LLM.read_text(encoding="utf-8")
    assert 'height=245' in llm
    assert '查看完整结果摘要' in llm
    assert '摘要预览（只读）' in llm
    assert 'preview = summary[:1800]' in llm
