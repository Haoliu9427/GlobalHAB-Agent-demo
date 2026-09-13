"""Two new actual pretrained adapters, synthetic smoke fixtures only."""
from pathlib import Path
import sys,runpy,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training.user_engine import build
from globalhab_demo.real_training.own_observations import foundation_score
fixture=runpy.run_path(str(ROOT/'tests/test_own_observations.py'))['fixture']
ds,ix,m=build(fixture(),7,True);rows=[]
for n in ['Chronos-Bolt-tiny','SmolLM2-360M-Instruct']:
    try:
        p,meta=foundation_score(n,ds,np.r_[ix['test'][:1],ix['future']],7,'Synthetic software check, not an ecological result',print)
        assert len(p)==2 and np.isfinite(p).all()
        rows.append({'name':n,'status':'passed_real_weights','rows':2,**meta})
    except Exception as e:rows.append({'name':n,'status':'failed','reason':str(e)})
    (ROOT/'validation/expanded_foundation_smoke.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False));print(rows[-1],flush=True)
assert all(r['status']=='passed_real_weights' for r in rows)
