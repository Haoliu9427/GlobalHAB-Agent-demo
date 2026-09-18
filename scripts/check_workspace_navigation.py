"""Verify independent workspace navigation and visible pretrained model names."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
a = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180)

# User-data workspace should not trigger the research exploration run.
a.session_state['workspace_mode'] = '自有数据分析'
a.run()
assert not a.exception, [e.message for e in a.exception]
assert 'exploration' not in a.session_state
assert {'Chronos-Bolt-small', 'Qwen2.5-0.5B-Instruct'} <= set(a.multiselect(key='user_models').options)

# Field visual workspace should load independently. Camera/file input is not
# programmatically populated here; feature logic is covered by non-UI tests.
a.radio(key='workspace_mode').set_value('现场影像甄别').run()
assert not a.exception, [e.message for e in a.exception]
assert 'exploration' not in a.session_state
assert a.button(key='vision_run')

# LLM workspace remains independent and the field-visual result is registered.
a.radio(key='workspace_mode').set_value('大模型结果解读').run()
assert not a.exception, [e.message for e in a.exception]
assert 'exploration' not in a.session_state
assert '最近一次现场影像甄别' in a.selectbox(key='llm_result_source').options
assert a.button(key='llm_generate').disabled

# Research workspace still exposes the registered foundation models.
a.radio(key='workspace_mode').set_value('研究与验证').run()
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
(ROOT / 'validation/workspace_navigation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(result)
