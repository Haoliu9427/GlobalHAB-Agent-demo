import sys,json,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import cv2,numpy as np
from globalhab_demo.visual_examples import sample_image,analyze_video
from globalhab_demo.field_visual import load_image,make_result
results={}
for kind in ['蓝绿色','绿色','红棕色']:
 r=make_result(load_image(sample_image(kind)),{'sea_surface_confirmed':True},root=root,requested_mode='规则基线')
 results[kind]=r['effective_visual_anomaly_score']
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'test.mp4';w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
 for i in range(30):w.write(np.full((240,320,3),(130,100+i,40),dtype=np.uint8))
 w.release();v=analyze_video(p.read_bytes(),root)
 assert len(v['frames'])==12
 results['video_frames']=len(v['frames'])
from streamlit.testing.v1 import AppTest
a=AppTest.from_file(str(root/'app.py'),default_timeout=180)
a.session_state['workspace_mode']='研究验证';a.run();assert not a.exception
for k in ['贝类养殖','海藻养殖','捕捞资源']:
 a.radio(key='bio_object').set_value(k).run();assert not a.exception,k
results['biological_objects']='passed'
(root/'validation/multimodal_features.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(results)
