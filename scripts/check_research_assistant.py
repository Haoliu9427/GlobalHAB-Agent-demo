import sys,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.research_assistant import execute_task,inspect_data
import pandas as pd,numpy as np
from PIL import Image
rng=np.random.default_rng(9);rows=[]
for s in range(4):
 for i,d in enumerate(pd.date_range('2020-01-01',periods=420)):
  v=np.sin(i/8+s)+rng.normal(0,.4)
  rows.append(dict(station_id=str(s),date=str(d.date()),available_at=str(d.date()),latitude=25+s,longitude=120,observed_event=int(v>.5),value=v,source='synthetic software test only'))
raw=pd.DataFrame(rows).to_csv(index=False).encode()
plan=dict(goal='测试未来7天',description='仅用于软件测试的合成事件，value为无单位信号',horizon=7,budget=3,future=True,mode='forecast',image_station='0',image_date='2021-02-23',sea_surface=True)
b=io.BytesIO();Image.fromarray(rng.integers(40,180,(80,80,3),dtype=np.uint8)).save(b,format='PNG')
r=execute_task(raw,plan,b.getvalue(),print,root=ROOT)
assert r['selected'] and len(r['forecast']) and not r['table'].empty
assert r['evidence']['station_date_match']
assert len(r['selection'])>=2
plan['mode']='anomaly';rr=execute_task(raw,plan,None)
assert rr['table'].empty and not rr['review'].empty
print('PASS',r['selected'],len(r['forecast']))
open(ROOT/'validation/agent_smoke.json','w').write(json.dumps({'fixture':'synthetic software testing only','selected':r['selected'],'forecast_rows':len(r['forecast']),'events':r['events'],'anomaly_fallback':True},ensure_ascii=False,indent=2))
