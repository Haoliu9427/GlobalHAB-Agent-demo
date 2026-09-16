"""Independent data workbench: model inventory, verification and forecasts."""
import hashlib,json,html
import pandas as pd
from .model_registry import MODELS,FOUNDATIONS,FUSIONS,OTHER,missing,expand,SCIENCE_COLUMNS,REMOTE
from .user_engine import build,run

def render():
    import streamlit as st
    data_col,task_col=st.columns([1,1],gap='large')
    with data_col, st.container(border=False,key='obs_upload_card'):
        st.markdown('### 观测数据')
        upload=st.file_uploader('现场观测CSV',type=['csv'],key='real_training_upload')
        prep_a, prep_b = st.columns([1, 1], gap='small')
        with prep_a:
            st.download_button('下载字段模板',data='station_id,date,available_at,latitude,longitude,observed_event,value,source,temperature,salinity,dissolved_oxygen\n',file_name='field_observations_template.csv',use_container_width=True)
        with prep_b:
            status_text = ('已选择 · '+str(round(upload.size/1024,1))+' KB') if upload else '等待上传CSV'
            st.button(status_text, disabled=True, use_container_width=True, key='real_training_upload_status')
        st.caption('必需：站点、日期、可用时间、经纬度、事件标签、value与来源；环境变量可选。')
        with st.expander('字段与数据要求',expanded=False):
            st.write('按行记录观测；未知标签可留空。上传后自动检查时间可用性、留出条件、缺失值和必需字段。')
        st.markdown('<div class="compact-card-footer">上传后先完成数据检查，再进入统一验证与模型比较。</div>', unsafe_allow_html=True)
    with task_col, st.container(border=False,key='obs_task_card'):
        st.markdown('### 预测任务')
        task_top, horizon_top = st.columns([1.55, 0.75], gap='small')
        with task_top:
            task=st.radio('分析任务',['历史预测验证','历史验证 + 最新时点未来预测'],key='user_task')
        with horizon_top:
            horizon=st.selectbox('预测时效（天）',[7,14,30],key='user_horizon')
        description=st.text_area('物种、事件阈值及value的含义/单位',max_chars=500,height=108,key='user_description')
        st.caption('输出：同一留出集模型比较、AP / Brier / ECE、样本/事件支持与数据质量。')
        with st.expander('本次分析说明',expanded=False):
            st.write('历史按时间留出；未知标签保留为未知。选择未来预测时，从各站最新有标签观测起算，并记录超范围特征与缺失比例。')
        st.markdown(f'<div class="compact-card-footer">当前：{task} · {horizon}天 · 结果独立保存。</div>', unsafe_allow_html=True)
    st.caption('历史不足或无法形成时间留出时，系统会明确说明原因。')
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
        st.session_state.pop('remote_action_notice',None)
        st.session_state.pop('user_result',None)
    def change_provider():
        provider=st.session_state['remote_provider']
        st.session_state['user_api_url']='https://api.deepseek.com' if provider=='DeepSeek' else ''
        st.session_state['user_api_model']=''
        st.session_state['user_api_key']=''
        st.session_state.pop('remote_model_list',None)
        st.session_state.pop('user_result',None)
        st.session_state.pop('qwen_explanation',None)
        st.session_state.pop('remote_action_notice',None)
    service_col,models_col=st.columns([1,1],gap='large')
    with service_col, st.container(border=False,key='obs_service_card'):
        st.markdown('### 远程大模型服务')
        mode=st.radio('服务配置来源',['自行填写','使用服务器配置'],key='remote_mode',horizontal=True)
        if mode=='自行填写':
            provider=st.selectbox('模型服务',['自定义兼容服务','DeepSeek','Qwen'],key='remote_provider',on_change=change_provider)
            presets={
                'DeepSeek':{'base_url':'https://api.deepseek.com','model':'deepseek-flash'},
                'Qwen':{'base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','model':'qwen-plus'},
            }
            preset=presets.get(provider,{'base_url':'','model':''})
            # Apply provider defaults to Streamlit state before the widgets are created.
            # This prevents a visible browser-restored value from diverging from the
            # backend value used by ready(), which previously left “测试连接” disabled.
            if preset['base_url'] and not str(st.session_state.get('user_api_url','')).strip():
                st.session_state['user_api_url']=preset['base_url']
            if preset['model'] and not str(st.session_state.get('user_api_model','')).strip():
                st.session_state['user_api_model']=preset['model']
            endpoint_col, model_col = st.columns([1.25, 1], gap='small')
            with endpoint_col:
                endpoint=st.text_input('API地址',placeholder='https://服务域名/compatible-mode/v1',key='user_api_url')
            with model_col:
                model_id=st.text_input('模型名称',placeholder='填写服务商提供的模型ID',key='user_api_model')
            key=st.text_input('API Key',type='password',key='user_api_key')
            effective_url=(endpoint.strip() or preset['base_url']).rstrip('/')
            effective_model=model_id.strip() or preset['model']
            remote={'base_url':effective_url,'model':effective_model,'api_key':key.strip()}
        else:
            provider='服务器配置'
            remote=settings()

        # Resolve the effective model independently of the browser widget state.
        # Streamlit can visually restore a text_input value while the backend state is
        # briefly empty on rerun; use provider defaults / discovered models as a safe
        # fallback so a valid DeepSeek/Qwen setup never leaves "测试连接" disabled.
        if not str(remote.get('model','')).strip():
            discovered=[str(x).strip() for x in (st.session_state.get('remote_model_list') or []) if str(x).strip()]
            fallback_model=(discovered[0] if discovered else '')
            if not fallback_model and mode=='自行填写':
                fallback_model=presets.get(provider,{}).get('model','')
            remote={**remote,'model':fallback_model}

        # Keep all action feedback outside the three narrow button columns.
        # This avoids Streamlit alerts being squeezed into a tall, thin block.
        remote_sig=hashlib.sha256(json.dumps({
            'base_url':remote.get('base_url',''),
            'model':remote.get('model',''),
            'api_key_hash':hashlib.sha256(remote.get('api_key','').encode()).hexdigest(),
        },sort_keys=True).encode()).hexdigest()
        action_a, action_b, action_c = st.columns(3, gap='small')
        with action_a:
            st.button('清除凭证',on_click=clear_key,use_container_width=True)
        with action_b:
            if st.button('读取模型',disabled=not remote.get('base_url') or not remote.get('api_key'),use_container_width=True):
                try:
                    from .remote_qwen import list_models
                    models=list_models(remote)
                    st.session_state['remote_model_list']=models
                    st.session_state['remote_action_notice']={
                        'signature':remote_sig,'kind':'info',
                        'text':'已读取可用模型 · '+str(len(models))+' 个'
                    }
                except ValueError as exc:
                    st.session_state['remote_action_notice']={
                        'signature':remote_sig,'kind':'error','text':'读取模型失败 · '+str(exc)
                    }
        with action_c:
            can_test=bool(remote.get('base_url') and remote.get('api_key'))
            if st.button('测试连接',disabled=not can_test,use_container_width=True):
                try:
                    test_remote=dict(remote)
                    if not str(test_remote.get('model','')).strip():
                        discovered=[str(x).strip() for x in (st.session_state.get('remote_model_list') or []) if str(x).strip()]
                        if discovered:
                            test_remote['model']=discovered[0]
                        elif mode=='自行填写':
                            test_remote['model']=presets.get(provider,{}).get('model','')
                    if not str(test_remote.get('model','')).strip():
                        raise ValueError('请先填写模型名称，或点击“读取模型”后选择可用模型。')
                    chat(test_remote,[{'role':'user','content':'Reply OK.'}])
                    provider_label=(provider if mode=='自行填写' else '服务器配置')
                    st.session_state['remote_action_notice']={
                        'signature':remote_sig,'kind':'success',
                        'text':'连接成功 · '+provider_label+' · '+str(test_remote.get('model',''))
                    }
                except ValueError as exc:
                    st.session_state['remote_action_notice']={
                        'signature':remote_sig,'kind':'error','text':'连接失败 · '+str(exc)
                    }
        notice=st.session_state.get('remote_action_notice')
        if notice and notice.get('signature')==remote_sig:
            kind=notice.get('kind','info')
            text=html.escape(str(notice.get('text','')))
            st.markdown(f'<div class="remote-status-strip {kind}"><span class="remote-status-dot"></span>{text}</div>',unsafe_allow_html=True)
        elif ready(remote):
            st.markdown('<div class="remote-status-strip ready"><span class="remote-status-dot"></span>参数已就绪 · '+html.escape(str(remote.get('model','')))+'</div>',unsafe_allow_html=True)
        else:
            missing_parts=[]
            if not remote.get('base_url'):missing_parts.append('API地址')
            if not remote.get('model'):missing_parts.append('模型名称')
            if not remote.get('api_key'):missing_parts.append('API Key')
            missing_text='缺少 '+'、'.join(missing_parts) if missing_parts else '尚未启用'
            st.markdown('<div class="remote-status-strip muted"><span class="remote-status-dot"></span>'+html.escape(missing_text)+'</div>',unsafe_allow_html=True)
        if st.session_state.get('remote_model_list'):
            with st.expander('服务返回的模型ID',expanded=False):
                st.write(st.session_state['remote_model_list'])
        consent=st.checkbox('允许本次使用远程服务发送上述数据',key='remote_consent')
        with st.expander('远程调用与隐私说明',expanded=False):
            if mode=='自行填写' and provider=='DeepSeek':
                st.caption('DeepSeek使用官方API基础地址；模型ID请以账户当前可用列表为准。')
            st.caption('凭证仅供当前会话使用，不保存到工程、结果包或服务器配置。API地址填写兼容接口基础地址，不含/chat/completions。')
            st.caption('调用可能产生API费用；请仅发送你有权使用的数据。')
    with models_col, st.container(border=False,key='obs_models_card'):
        st.markdown('### 模型与运行')
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
        st.caption(f'验证：时间留出 · {len(selected)}个选择模型 + 季节基线 · {horizon}天 · 上限{epochs}轮。')
        # Keep the primary action anchored to the bottom of the matched card.
        signature=hashlib.sha256((data or b'')+json.dumps([task,horizon,description,selected,epochs,remote.get('model'),remote.get('base_url'),hashlib.sha256(remote.get('api_key','').encode()).hexdigest(),mode],ensure_ascii=False).encode()).hexdigest()
        with st.container(border=False,key='user_run_zone'):
            st.markdown('<div class="compact-card-footer">运行后保存指标、验证样本、预测结果与数据质量记录。</div>', unsafe_allow_html=True)
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
