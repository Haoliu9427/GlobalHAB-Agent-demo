"""Exercise mainland controls in the complete app, without remote writes."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180).run()
assert not a.exception,[e.message for e in a.exception]
a.radio(key='real_data_mode').set_value('中国近海调查').run()
assert not a.exception,[e.message for e in a.exception]
checks=[]
for marker in ['ITS1','18S V4']:
    a.selectbox(key='mainland_marker').set_value(marker).run()
    for region in ['渤海','黄海','东海','南海']:
        a.selectbox(key='mainland_sea').set_value(region).run()
        assert not a.exception,[e.message for e in a.exception]
        checks.append({'marker':marker,'sea':region,'exceptions':0})
(ROOT/'validation/mainland_app.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False))
