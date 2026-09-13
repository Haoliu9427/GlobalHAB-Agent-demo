"""Real pretrained-weight adapter smoke test on synthetic software fixtures only."""
from pathlib import Path
import sys,json,runpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training.own_observations import prepare,foundation_score
fixture=runpy.run_path(str(ROOT/'tests/test_own_observations.py'))['fixture']
ds,ix,m=prepare(fixture());result=[]
for name in ['Chronos-Bolt-small','Qwen2.5-0.5B-Instruct']:
    try:
        score,meta=foundation_score(name,ds,ix['calibration'][:2],7,'Synthetic software test binary event; NOT real HAB evidence',print)
        assert len(score)==2 and np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
        result.append(dict(name=name,status='passed_real_weights',rows=2,**meta))
    except Exception as exc:result.append(dict(name=name,status='failed',error=str(exc)))
    (ROOT/'validation/own_foundation_smoke.json').write_text(json.dumps({'purpose':'Synthetic software smoke test, no scientific performance claim','checks':result},ensure_ascii=False,indent=2))
    print(result[-1],flush=True)
assert all(r['status']=='passed_real_weights' for r in result)
