"""Local application smoke test; does not connect to or update deployed app."""
from pathlib import Path
import json,sys
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180).run()
result={'exceptions':[e.message for e in at.exception],'tabs':[t.label for t in at.tabs]}
print(json.dumps(result,ensure_ascii=False,indent=2))
sys.exit(1 if result['exceptions'] else 0)
