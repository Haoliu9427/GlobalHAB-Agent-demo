"""Independent data workbench: model inventory, verification and forecasts."""
import hashlib,json
import pandas as pd
from .model_registry import MODELS,FOUNDATIONS,FUSIONS,OTHER,missing,expand,SCIENCE_COLUMNS
from .user_engine import build,run

def render():
    import streamlit as st
    upload=st.file_uploader('现场观测CSV',type=['csv'],key='real_training_upload')
    st.download_button('下载字段模板',data='station_id,date,available_at,latitude,longitude,observed_event,value,source,temperature,salinity,dissolved_oxygen\n',file_name='field_observations_template.csv')
    task=st.radio('分析任务',['历史预测验证','历史验证 + 最新时点未来预测'],key='user_task')
    horizon=st.selectbox('预测时效（天）',[7,14,30],key='user_horizon')
    description=st.text_area('物种、事件阈值及value的含义/单位',max_chars=500,key='user_description')
    st.caption('未知标签不作为阴性。未来预测以每站最新有标签观测为起点，并不自动等于今天；历史不足或无法留出时会说明原因。')
    data=upload.getvalue() if upload else None;m=None
    if data:
        try:
            _,_,m=build(data,horizon,task!='历史预测验证')
            st.success('数据检查通过');st.write('各阶段样本量',m['counts'])
        except Exception as exc:st.error('数据检查：'+str(exc))
    records=[];allowed=[]
    for n in MODELS:
        reason='可选择（尚未针对本次数据运行）'
        if missing(n):reason='需要安装：'+', '.join(missing(n))
        elif m and any(x.startswith('Chronos') for x in expand([n])) and not m['chronos_eligible']:reason='不适用：需要逐日连续历史及准确第N天复测'
        elif m and n.startswith('STS-') and not m['science_eligible']:reason='不适用：缺少已对齐上游/输运字段或其可用时间'
        else:allowed.append(n)
        records.append({'模型':n,'类型':'融合' if n in FUSIONS else '基础模型' if n in FOUNDATIONS else '预测模型/基线','当前条件':reason})
    with st.expander('项目完整模型目录与适用条件',expanded=False):
        st.dataframe(pd.DataFrame(records),hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame([{'模型/方法':n,'用途与入口':v} for n,v in OTHER.items()]),hide_index=True,use_container_width=True)
        st.caption('同一家族在不同历史任务中的参数配置不作为新模型重复计数。STS字段：'+', '.join(SCIENCE_COLUMNS)+', upstream_available_at。上游对齐需由数据提供者完成。')
        st.write('基础模型为3种Chronos规格、3种Qwen规格和SmolLM2。首次运行需要下载权重；大规格需要更多内存。目录可选择不代表已在所有主机或数据集上验证。')
    # Only reset unavailable entries when the new data invalidate their prerequisites.
    if 'user_models' in st.session_state:st.session_state['user_models']=[n for n in st.session_state['user_models'] if n in allowed]
    selected=st.multiselect('选择本次运行模型（可多选）',allowed,default=[n for n in ['Logistic','HistGradientBoosting'] if n in allowed],key='user_models')
    epochs=st.slider('时序模型训练轮数上限',5,50,20,key='user_epochs')
    if selected:st.caption('实际运行（含融合组件与季节基线）：'+', '.join(expand(['Seasonal Climatology']+selected)))
    signature=hashlib.sha256((data or b'')+json.dumps([task,horizon,description,selected,epochs],ensure_ascii=False).encode()).hexdigest()
    if st.button('开始分析',key='user_run'):
        st.session_state.pop('user_result',None)
        if not data or m is None:st.error('请上传满足数据检查条件的CSV。');return
        progress=st.empty()
        try:
            with st.spinner('正在计算本次数据的结果…'):
                result=run(data,selected,horizon,description,epochs,task!='历史预测验证',progress.write)
            st.session_state['user_result']=(signature,result);progress.success('本次分析完成')
        except Exception as exc:progress.error('本次分析未完成：'+str(exc));return
    saved=st.session_state.get('user_result')
    if saved and saved[0]==signature:
        table,forecast,explanation,manifest,archive=saved[1]
        st.caption('本次输入SHA256：'+manifest['input_sha256'])
        a,b,c=st.tabs(['历史验证结果','未来预测','结果解读'])
        with a:
            st.dataframe(table,hide_index=True,use_container_width=True)
            st.caption('AP越高越好；Brier、ECE越低越好。基础模型仅一次固定推理；其他模型三个种子。')
        with b:
            if forecast.empty:st.info('本次仅执行历史验证。')
            else:
                st.dataframe(forecast[['site','origin_date','label_date','model','seed','probability','outside_training_range_features','missing_fraction']],hide_index=True,use_container_width=True)
                st.caption('无未来真实标签，不显示未来准确率。模型保持训练期权重，使用站点最新历史输入。')
        with c:st.text(explanation)
        st.download_button('下载本次完整结果',data=archive,file_name='GlobalHAB_analysis_results.zip')
    elif saved:st.info('输入或设置已改变，旧结果已隐藏，请重新运行。')
