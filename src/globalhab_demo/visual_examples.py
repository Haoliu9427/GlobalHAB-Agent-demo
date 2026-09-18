"""Synthetic diagnostic colour fields, not photographs or labelled HAB samples."""
import io
import numpy as np
from PIL import Image

def sample_image(kind='蓝绿色'):
    colors={'蓝绿色':[35,115,135],'绿色':[70,130,65],'红棕色':[135,75,45]}
    rng=np.random.default_rng(12);y,x=np.mgrid[:320,:480]
    waves=8*np.sin(x/22+y/9)+rng.normal(0,3,(320,480))
    arr=np.clip(np.array(colors[kind])[None,None,:]+waves[:,:,None],0,255).astype('uint8')
    b=io.BytesIO();Image.fromarray(arr).save(b,format='PNG');return b.getvalue()

def render_examples(root):
    from globalhab_demo.display_locale import st
    from globalhab_demo.field_visual import load_image,make_result
    from globalhab_demo.result_pool import register
    with st.container(border=True):
        left,right=st.columns([1,1],gap='large')
        with left:
            st.markdown('#### 选择测试水色')
            kind=st.selectbox('选择示例',['蓝绿色','绿色','红棕色'],key='example_visual_kind')
            raw=sample_image(kind);st.image(raw,width=400,caption='合成水色测试图 · 不是实海照片')
        with right:
            st.markdown('#### 筛查结果')
            st.caption('检查水色特征与规则筛查流程，不提供真实藻华标签。')
            if st.button('分析示例图像',key='example_visual_run',type='primary'):
                r=make_result(load_image(raw),{'sea_surface_confirmed':True,'notes':'合成水色测试图'},root=root,requested_mode='规则基线')
                st.session_state['example_visual_result']=(kind,r)
                register('示例图像','合成水色测试；非实海验证',r)
            saved=st.session_state.get('example_visual_result')
            if saved and saved[0]==kind:
                r=saved[1];st.metric('视觉异常指数',f"{r['effective_visual_anomaly_score']:.2f}")
                st.write(r['priority_reason'])
                with st.expander('质量与特征明细'):st.json(r)
            else:st.info('选择一种水色，点击分析查看结果。')


def analyze_video(raw,root):
    import cv2,tempfile,os
    from globalhab_demo.field_visual import make_result
    if len(raw)>80*1024*1024:raise ValueError('短视频上限80MB。')
    name=None;cap=None;rows=[]
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as f:f.write(raw);name=f.name
        cap=cv2.VideoCapture(name)
        fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps<=0 or count<1:raise ValueError('无法读取视频，请上传标准MP4。')
        duration=count/fps
        if duration>120:raise ValueError('请截取2分钟以内的视频。')
        for idx in np.unique(np.linspace(0,int(count)-1,min(12,int(count))).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,bgr=cap.read()
            if not ok:
                rows.append({'second':float(idx/fps),'score':None,'status':'解码失败'});continue
            frame=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));frame.thumbnail((1280,720))
            r=make_result(frame,{'sea_surface_confirmed':True},root=root,requested_mode='规则基线')
            rows.append({'second':float(idx/fps),'score':r['effective_visual_anomaly_score'],'status':r['screening_priority'],'quality_suitable':r['quality'].get('suitable',False)})
        return {'duration_seconds':duration,'frames':rows,'mode':'最多12个均匀抽样帧的规则筛查；不代表逐帧识别，不确诊藻华。'}
    finally:
        if cap:cap.release()
        if name:os.unlink(name)

def render_video(root):
    from globalhab_demo.display_locale import st
    import pandas as pd,plotly.express as px,hashlib,json
    from globalhab_demo.result_pool import register
    st.caption('上传拍摄好的短视频：最多2分钟、80MB，均匀抽取最多12帧。不支持网页内实时录像或逐帧跟踪。')
    file=st.file_uploader('海面短视频',type=['mp4','mov','avi'],key='video_upload')
    if not file:return
    raw=file.getvalue();sig=hashlib.sha256(raw).hexdigest();st.video(raw)
    confirm=st.checkbox('视频主体为海面或水体',key='video_water_confirm')
    if st.button('分析视频',disabled=not confirm):
        try:
            with st.spinner('抽帧与筛查…'):r=analyze_video(raw,root)
            r['input_sha256']=sig;st.session_state['video_result']=(sig,r);register('视频筛查','抽帧规则筛查',r)
        except Exception as exc:st.error(str(exc))
    saved=st.session_state.get('video_result')
    if saved and saved[0]==sig:
        r=saved[1];df=pd.DataFrame(r['frames'])
        st.plotly_chart(px.line(df,x='second',y='score',markers=True,labels={'second':'视频时间（秒）','score':'视觉异常指数'}),use_container_width=True)
        st.caption(r['mode'])
        with st.expander('各帧结果'):st.dataframe(df,hide_index=True)
        st.download_button('下载视频筛查记录',json.dumps(r,ensure_ascii=False),file_name='video_screening.json')
