"""Independent data workbench: model inventory, verification and forecasts."""
import hashlib,json
import pandas as pd
from .model_registry import MODELS,FOUNDATIONS,FUSIONS,OTHER,missing,expand,SCIENCE_COLUMNS,REMOTE
from .user_engine import build,run

def render():
    import streamlit as st
    data_col,task_col=st.columns([1,1],gap='large')
    with data_col, st.container(border=True,key='obs_upload_card'):
        st.markdown('### 观测数据')
        st.caption('上传现场记录，建立本次分析的数据集。')
        upload=st.file_uploader('现场观测CSV',type=['csv'],key='real_training_upload')
        prep_a, prep_b = st.columns([1, 1], gap='small')
        with prep_a:
            st.download_button('下载字段模板',data='station_id,date,available_at,latitude,longitude,observed_event,value,source,temperature,salinity,dissolved_oxygen\n',file_name='field_observations_template.csv',use_container_width=True)
        with prep_b:
            status_text = ('已选择 · '+str(round(upload.size/1024,1))+' KB') if upload else '等待上传CSV'
            st.button(status_text, disabled=True, use_container_width=True, key='real_training_upload_status')
        st.markdown('#### 数据结构')
        field_a, field_b = st.columns(2, gap='small')
        with field_a:
            st.caption('定位与时间')
            st.write('站点 · 日期 · 可用时间 · 经纬度')
        with field_b:
            st.caption('标签与观测')
            st.write('事件标签 · value · 数据来源')
        st.markdown('#### 上传后自动检查')
        check_a, check_b = st.columns(2, gap='small')
        with check_a:
            st.caption('时间与留出')
            st.write('可用时间 · 历史跨度 · 留出窗口')
        with check_b:
            st.caption('标签与字段')
            st.write('未知标签 · 缺失值 · 必需字段')
        st.caption('温度、盐度、溶解氧等环境变量可作为可选协变量；通过检查后再进入统一验证和模型比较。')
        if upload:st.caption('当前文件：'+upload.name)
    with task_col, st.container(border=True,key='obs_task_card'):
        st.markdown('### 预测任务')
        task_top, horizon_top = st.columns([1.55, 0.75], gap='small')
        with task_top:
            task=st.radio('分析任务',['历史预测验证','历史验证 + 最新时点未来预测'],key='user_task')
        with horizon_top:
            horizon=st.selectbox('预测时效（天）',[7,14,30],key='user_horizon')
        description=st.text_area('物种、事件阈值及value的含义/单位',max_chars=500,height=108,key='user_description')
        st.markdown('#### 本次输出')
        if task=='历史预测验证':
            st.markdown('- 同一历史留出集上的模型比较\n- AP、Brier、ECE与样本/事件支持\n- 训练范围与数据质量说明')
        else:
            st.markdown('- 历史验证指标与模型比较\n- 各站最新时点的未来风险概率\n- 超出训练范围特征与缺失比例提示')
        st.markdown('#### 分析前检查')
        pre_a, pre_b = st.columns(2, gap='small')
        with pre_a:
            st.caption('验证方式')
            st.write('按时间留出 · 同一测试集比较')
            st.caption('未知标签')
            st.write('保留为未知，不自动记作阴性')
        with pre_b:
            st.caption('未来预测起点')
            st.write('每站最新有标签观测')
            st.caption('结果保存')
            st.write('指标表 · 预测表 · 运行清单')
        st.caption(f'当前设置：{task} · {horizon}天。运行后会生成独立结果记录，不覆盖项目固定证据。')
    st.caption('未知标签不作为阴性。未来预测以每站最新有标签观测为起点，并不自动等于今天；历史不足或无法留出时会说明原因。')
    data=upload.getvalue() if upload else None;m=None
    if data:
        try:
            _,_,m=build(data,horizon,task!='历史预测验证')
            st.success('数据检查通过');st.write('各阶段样本量',m['counts'])
        except Exception as exc:st.error('数据检查：'+str(exc))
    from .remote_qwen import settings,ready,chat,explain
    def clear_key():
        st.session_state['user_api_key']=''
        st.session_state.pop('qwen_explanation',None)
        st.session_state.pop('user_result',None)
    def change_provider():
        provider=st.session_state['remote_provider']
        st.session_state['user_api_url']='https://api.deepseek.com' if provider=='DeepSeek' else ''
        st.session_state['user_api_model']=''
        st.session_state['user_api_key']=''
        st.session_state.pop('remote_model_list',None)
        st.session_state.pop('user_result',None)
        st.session_state.pop('qwen_explanation',None)
    service_col,models_col=st.columns([1,1],gap='large')
    with service_col, st.container(border=True,key='obs_service_card'):
        st.markdown('### 远程大模型服务')
        st.caption('连接DeepSeek、Qwen或其他兼容服务。仅运行本地模型时无需填写。')
        mode=st.radio('服务配置来源',['自行填写','使用服务器配置'],key='remote_mode',horizontal=True)
        if mode=='自行填写':
            provider=st.selectbox('模型服务',['自定义兼容服务','DeepSeek','Qwen'],key='remote_provider',on_change=change_provider)
            endpoint_col, model_col = st.columns([1.25, 1], gap='small')
            with endpoint_col:
                endpoint=st.text_input('API地址',placeholder='https://服务域名/compatible-mode/v1',key='user_api_url')
            with model_col:
                model_id=st.text_input('模型名称',placeholder='填写服务商提供的模型ID',key='user_api_model')
            key=st.text_input('API Key',type='password',key='user_api_key')
            remote={'base_url':endpoint.strip().rstrip('/'),'model':model_id.strip(),'api_key':key.strip()}
        else:
            remote=settings()
        action_a, action_b, action_c = st.columns(3, gap='small')
        with action_a:
            st.button('清除凭证',on_click=clear_key,use_container_width=True)
        with action_b:
            if st.button('读取模型',disabled=not remote.get('base_url') or not remote.get('api_key'),use_container_width=True):
                try:
                    from .remote_qwen import list_models
                    st.session_state['remote_model_list']=list_models(remote)
                except ValueError as exc:st.error(str(exc))
        with action_c:
            if st.button('测试连接',disabled=not ready(remote),use_container_width=True):
                try:
                    chat(remote,[{'role':'user','content':'Reply OK.'}]);st.success('服务可访问；正式运行仍会检查输出格式。')
                except ValueError as exc:st.error(str(exc))
        if st.session_state.get('remote_model_list'):
            with st.expander('服务返回的模型ID',expanded=False):
                st.write(st.session_state['remote_model_list'])
        st.markdown('#### 连接状态')
        if ready(remote):
            st.success('连接参数已完整 · '+remote['model']+'。可先测试连接，也可仅使用本地模型继续分析。')
        else:
            missing_parts=[]
            if not remote.get('base_url'):missing_parts.append('API地址')
            if not remote.get('model'):missing_parts.append('模型名称')
            if not remote.get('api_key'):missing_parts.append('API Key')
            st.info('远程服务尚未启用'+('：缺少'+'、'.join(missing_parts) if missing_parts else '。'))
        consent=st.checkbox('允许本次使用远程服务发送上述数据',key='remote_consent')
        st.markdown('#### 本次远程调用范围')
        st.markdown('- 预测：仅发送特征名、历史数值与事件说明。\n- 解读：仅发送本次结构化结果摘要。\n- 默认不发送站点ID、未来标签或完整原始CSV。')
        with st.expander('远程调用与隐私说明',expanded=False):
            if mode=='自行填写' and provider=='DeepSeek':
                st.caption('DeepSeek使用官方API基础地址；模型ID请以账户当前可用列表为准。')
            st.caption('凭证仅供当前会话使用，不保存到工程、结果包或服务器配置。API地址填写兼容接口基础地址，不含/chat/completions。')
            st.caption('调用可能产生API费用；请仅发送你有权使用的数据。')
    with models_col, st.container(border=True,key='obs_models_card'):
        st.markdown('### 模型与运行')
        st.caption('选择参与本次分析的模型，使用同一组验证样本比较。')
        records=[];allowed=[]
        for n in MODELS:
            reason='可选择（尚未针对本次数据运行）'
            if REMOTE in expand([n]) and not ready(remote):
                reason='未连接：请填写远程API配置'
                if n==REMOTE:allowed.append(n)
            elif missing(n):reason='需要安装：'+', '.join(missing(n))
            elif m and any(x.startswith('Chronos') for x in expand([n])) and not m['chronos_eligible']:reason='不适用：需要逐日连续历史及准确第N天复测'
            elif m and n.startswith('STS-') and not m['science_eligible']:reason='不适用：缺少已对齐上游/输运字段或其可用时间'
            else:allowed.append(n)
            records.append({'模型':n,'类型':'融合' if n in FUSIONS else '基础模型' if n in FOUNDATIONS else '预测模型/基线','当前条件':reason})
        with st.expander('项目完整模型目录与适用条件',expanded=False):
            st.dataframe(pd.DataFrame(records),hide_index=True,use_container_width=True)
            st.dataframe(pd.DataFrame([{'模型/方法':n,'用途与入口':v} for n,v in OTHER.items()]),hide_index=True,use_container_width=True)
            st.caption('同一家族在不同历史任务中的参数配置不作为新模型重复计数。STS字段：'+', '.join(SCIENCE_COLUMNS)+', upstream_available_at。上游对齐需由数据提供者完成。')
            st.write('本地基础模型为3种Chronos规格、3种Qwen规格和SmolLM2。首次运行需要下载权重；大规格需要更多内存。目录可选择不代表已在所有主机或数据集上验证。')
        # Only reset unavailable entries when the new data invalidate their prerequisites.
        if 'user_models' in st.session_state:st.session_state['user_models']=[n for n in st.session_state['user_models'] if n in allowed]
        selected=st.multiselect('选择本次运行模型（可多选）',allowed,default=[n for n in ['Logistic','HistGradientBoosting'] if n in allowed],key='user_models')
        epochs=st.slider('时序模型训练轮数上限',5,50,20,key='user_epochs')
        if selected:st.caption('实际运行（含融合组件与季节基线）：'+', '.join(expand(['Seasonal Climatology']+selected)))
        st.markdown('#### 运行前检查')
        check_a, check_b = st.columns(2, gap='small')
        with check_a:
            st.caption('验证策略')
            st.write('历史时间留出 · 同一测试样本比较')
            st.caption('当前任务')
            st.write(task)
        with check_b:
            st.caption('运行模型')
            st.write(f'{len(selected)} 个用户选择 + 季节基线')
            st.caption('预测时效 / 训练预算')
            st.write(f'{horizon} 天 · 上限 {epochs} 轮')
        st.markdown('#### 分析记录')
        st.markdown('- 保存模型指标、验证样本与数据质量信息。\n- 如启用未来预测，同时记录训练范围外特征与缺失比例。\n- 所有输出与项目固定证据分开保存，可单独下载。')
        signature=hashlib.sha256((data or b'')+json.dumps([task,horizon,description,selected,epochs,remote.get('model'),remote.get('base_url'),hashlib.sha256(remote.get('api_key','').encode()).hexdigest(),mode],ensure_ascii=False).encode()).hexdigest()
        if st.button('开始分析',key='user_run',type='primary',use_container_width=True):
            st.session_state.pop('user_result',None)
            if not data or m is None:st.error('请上传满足数据检查条件的CSV。');return
            if any(REMOTE in expand([n]) for n in selected) and not consent:
                st.error('请先勾选远程数据发送授权。');return
            progress=st.empty()
            try:
                with st.spinner('正在计算本次数据的结果…'):
                    result=run(data,selected,horizon,description,epochs,task!='历史预测验证',progress.write,remote_config=remote)
                st.session_state['user_result']=(signature,result);progress.success('本次分析完成')
            except Exception as exc:progress.error('本次分析未完成：'+str(exc));return
    saved=st.session_state.get('user_result')
    if saved and saved[0]==signature:
        table,forecast,explanation,manifest,archive=saved[1]
        if any(REMOTE in expand([n]) for n in selected):st.caption('本次远程模型：'+remote['model'])
        st.caption('本次输入SHA256：'+manifest['input_sha256'])
        st.markdown('### 本次分析结果')
        summary_cols=st.columns(3)
        summary_cols[0].metric('参与评价的模型',int(table['model'].nunique()))
        summary_cols[1].metric('预测时效',str(horizon)+'天')
        summary_cols[2].metric('未来预测记录',len(forecast))
        a,b,c=st.tabs(['历史验证结果','未来预测','结果解读'])
        with a:
            st.dataframe(table,hide_index=True,use_container_width=True)
            st.caption('AP越高越好；Brier、ECE越低越好。基础模型仅一次固定推理；其他模型三个种子。')
        with b:
            if forecast.empty:st.info('本次仅执行历史验证。')
            else:
                st.dataframe(forecast[['site','origin_date','label_date','model','seed','probability','outside_training_range_features','missing_fraction']],hide_index=True,use_container_width=True)
                st.caption('无未来真实标签，不显示未来准确率。模型保持训练期权重，使用站点最新历史输入。')
        with c:
            st.text(explanation)
            if st.button('用大模型解读本次结果',disabled=not ready(remote) or not consent):
                try:
                    with st.spinner('大模型正在解读本次结果…'): st.session_state['qwen_explanation']=(signature,explain(remote,explanation))
                except ValueError as exc:st.error(str(exc))
            decoded=st.session_state.get('qwen_explanation')
            if decoded and decoded[0]==signature:
                st.caption('大模型生成解读；不改变上方计算结果。')
                st.write(decoded[1])
                st.download_button('下载大模型解读',decoded[1],file_name='qwen_interpretation.txt')
        st.download_button('下载本次完整结果',data=archive,file_name='GlobalHAB_analysis_results.zip')
    elif saved:st.info('输入或设置已改变，旧结果已隐藏，请重新运行。')
