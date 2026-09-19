"""Session-scoped, bounded output registry. Never stores credentials or raw images."""
import json,hashlib,datetime
from globalhab_demo.display_locale import st

def register(source,kind,payload,limit=12000):
    text=json.dumps(payload,ensure_ascii=False,default=str)
    digest=hashlib.sha256((source+kind+text).encode()).hexdigest()[:16]
    pool=st.session_state.setdefault('result_pool',{})
    if digest not in pool:
        pool[digest]={'id':digest,'source':source,'evidence_type':kind,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),'summary':text if limit is None else text[:limit]}
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
    pending=st.session_state.pop('_pool_selection_jump',None)
    if pending and pending in pool:
        st.session_state['pool_selection']=[pending]
    elif 'pool_selection' in st.session_state:
        st.session_state['pool_selection']=[k for k in st.session_state['pool_selection'] if k in pool]
    ids=st.multiselect('选择需要一起解读的结果',list(pool),default=None if "pool_selection" in st.session_state else list(pool)[-3:],format_func=lambda k:pool[k]['source']+' · '+pool[k]['time'][11:19]+' · '+pool[k]['evidence_type'],key='pool_selection')
    st.dataframe(pd.DataFrame([{'来源':item['source'],'证据类型':item['evidence_type'],'记录时间（UTC）':item['time']} for item in pool.values()]),hide_index=True)
    st.download_button('下载本会话结果汇总',json.dumps(list(pool.values()),ensure_ascii=False,indent=2),file_name='session_result_pool.json')
    return '\n\n'.join('来源：'+pool[k]['source']+'\n证据类型：'+pool[k]['evidence_type']+'\n'+pool[k]['summary'] for k in ids)


def publish_scientific_result(result):
    """Queue newly produced science evidence for interpretation, once per run."""
    payload={k:result.get(k) for k in ("model","goal","status","seed","data_sha256","experimental_budget","experiments_executed","known_total_tokens","elapsed_seconds","candidates","controls","final")}
    payload["planning_record"]=[{k:row[k] for k in ("step","tool","arguments","rationale","status") if k in row} for row in result.get("audit",[])]
    payload["scope"]="合成实验；验证阶段与独立测试分别记录。不得把候选相关性当作因果，未完成的检验不能视为通过。"
    before=set(st.session_state.get("result_pool",{}))
    rid=register("智能研究","大模型规划与科学检验（合成）",payload,limit=None)
    if rid not in before:
        st.session_state['_llm_source_jump']='本会话结果汇总'
        st.session_state['_pool_selection_jump']=rid
    return rid
