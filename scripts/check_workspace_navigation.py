"""Verify independent workspace navigation and visible pretrained model names."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
a = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180)

# User-data workspace should not trigger the research exploration run.
a.session_state['workspace_mode'] = '自有数据分析'
a.session_state['own_observations_workspace_open'] = True
a.run()
assert not a.exception, [e.message for e in a.exception]
assert 'exploration' not in a.session_state
catalogs = [d.value for d in a.dataframe if '模型' in d.value.columns]
assert any(
    {'Chronos-Bolt-small', 'Qwen2.5-0.5B-Instruct'} <= set(table['模型'].astype(str))
    for table in catalogs
)
assert all(a.button(key=key) for key in ['nav_home', 'nav_research', 'nav_data', 'nav_visual', 'nav_llm'])

# Field visual workspace should load independently. Camera/file input is not
# programmatically populated here; feature logic is covered by non-UI tests.
a.session_state['workspace_mode'] = '现场影像甄别'
a.run()
assert not a.exception, [e.message for e in a.exception]
assert 'exploration' not in a.session_state
assert a.button(key='vision_run')

# LLM workspace remains independent and the field-visual result is registered.
a.session_state['workspace_mode'] = '大模型结果解读'
a.run()
assert not a.exception, [e.message for e in a.exception]
assert 'exploration' not in a.session_state
assert '最近一次现场影像甄别' in a.selectbox(key='llm_result_source').options
assert a.button(key='llm_generate').disabled

# Research workspace still exposes the registered foundation models.
a.session_state['workspace_mode'] = '研究与验证'
a.run()
assert not a.exception, [e.message for e in a.exception]
for name in ['Qwen2.5-0.5B-Instruct', 'Chronos-Bolt-small']:
    a.selectbox(key='published_foundation_model').set_value(name).run()
    assert not a.exception, [e.message for e in a.exception]
    tables = [d.value for d in a.dataframe if 'model' in d.value.columns]
    assert any(name in d.model.values for d in tables)
    assert all(not d.model.isin(['LLM', 'LLM+EcoFusion']).any() for d in tables)

result = {
    'exceptions': 0,
    'independent_workspace_skips_exploration': True,
    'field_visual_workspace': True,
    'llm_interpretation_workspace': True,
    'visible_models': ['Qwen2.5-0.5B-Instruct', 'Chronos-Bolt-small'],
    'upload_logic_changed': False,
}
try:
    (ROOT / 'validation/workspace_navigation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
except OSError as exc:
    # Verification remains useful when the deployed package is mounted read-only.
    print(f"warning: could not persist workspace_navigation.json: {exc}")
print(result)
