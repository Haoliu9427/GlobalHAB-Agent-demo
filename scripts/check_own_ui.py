"""UI interaction test with explicit synthetic upload, never a research result."""
import tempfile,json,sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
source='''import sys,io,runpy
from pathlib import Path
from unittest.mock import patch
import streamlit as st
ROOT=Path(ROOT_VALUE)
sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training.own_observations import render
fixture=runpy.run_path(str(ROOT/'tests/test_own_observations.py'))['fixture']
with patch.object(st,'file_uploader',return_value=io.BytesIO(fixture())):
    render()
'''.replace('ROOT_VALUE',repr(str(ROOT)))
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'app.py';p.write_text(source)
    a=AppTest.from_file(str(p),default_timeout=120).run()
    assert not a.exception,[e.message for e in a.exception]
    a.text_area(key='user_description').set_value('Synthetic interface test only').run()
    a.button(key='user_run').click().run()
    assert not a.exception,[e.message for e in a.exception]
    assert 'user_result' in a.session_state
    rows=len(a.session_state['user_result'][1][0]);assert rows==9
    a.selectbox(key='user_horizon').set_value(14).run()
    assert any('旧结果' in i.value for i in a.info)
    (ROOT/'validation/own_ui_smoke.json').write_text(json.dumps({'synthetic_fixture':True,'exceptions':0,'training_result_rows':rows,'stale_result_hidden':True}))
    print('UI upload, model execution, result download state and stale-result hiding passed')
