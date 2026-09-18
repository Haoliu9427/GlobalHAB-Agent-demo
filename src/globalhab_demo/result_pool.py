"""Session-scoped, bounded output registry. Never stores credentials or raw images."""
import json,hashlib,datetime
from globalhab_demo.display_locale import st

def register(source,kind,payload):
    text=json.dumps(payload,ensure_ascii=False,default=str)
    digest=hashlib.sha256((source+kind+text).encode()).hexdigest()[:16]
    pool=st.session_state.setdefault('result_pool',{})
    if digest not in pool:
        pool[digest]={'id':digest,'source':source,'evidence_type':kind,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),'summary':text[:12000]}
        while len(pool)>30:pool.pop(next(iter(pool)))
    return digest

def collect():
    r=st.session_state.get('field_visual_result')
    if r:register('影像识别','视觉筛查；非藻种或毒素确诊',r)
    return st.session_state.get('result_pool',{})

def render_selection():
    import pandas as pd
    pool=collect()
    st.caption('本次会话自动收集的结果；最多保留30份。下载可保存，重启或会话结束后可能清空。不会自动发送给远程模型。')
    if not pool:
        st.info('还没有任务输出。完成数据分析、影像筛查或进入研究工作区后，结果会自动出现在这里。');return ''
    ids=st.multiselect('选择需要一起解读的结果',list(pool),default=list(pool)[-3:],format_func=lambda k:pool[k]['source']+' · '+pool[k]['time'][11:19]+' · '+pool[k]['evidence_type'],key='pool_selection')
    st.dataframe(pd.DataFrame([{'来源':item['source'],'证据类型':item['evidence_type'],'记录时间（UTC）':item['time']} for item in pool.values()]),hide_index=True)
    st.download_button('下载本会话结果汇总',json.dumps(list(pool.values()),ensure_ascii=False,indent=2),file_name='session_result_pool.json')
    return '\n\n'.join('来源：'+pool[k]['source']+'\n证据类型：'+pool[k]['evidence_type']+'\n'+pool[k]['summary'] for k in ids)
