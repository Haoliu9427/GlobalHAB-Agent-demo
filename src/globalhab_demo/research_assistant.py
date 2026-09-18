"""Bounded research agent: inspect, plan, act, observe, freeze, verify.
No LLM impersonation; no test-feedback selection; no invented multimodal probabilities.
"""
from pathlib import Path
import io,json,time,hashlib,zipfile,re
import numpy as np
import pandas as pd
from globalhab_demo.display_locale import st

VERSION='research-agent-1'
REQUIRED=['station_id','date','available_at','latitude','longitude','observed_event','value','source']

def parse_goal(goal):
    windows=[int(n) for n in re.findall(r'(\d+)\s*天',goal)]
    screening=('异常' in goal or '筛查' in goal) and not any(w in goal for w in ['预测','未来','预警'])
    return {'intent':'历史异常筛查' if screening else '预测与验证','requested_horizons':sorted(set(windows)),
            'scope':'支持观测预测验证、历史异常筛查及现场影像证据关联。其他研究问题需要转入自主配置或人工分析。'}

def inspect_data(raw):
    if len(raw)>20*1024*1024:raise ValueError('CSV超过20MB，请缩小任务范围。')
    frame=pd.read_csv(io.BytesIO(raw),dtype={'station_id':str})
    missing=set(REQUIRED)-set(frame)
    if missing:raise ValueError('缺少字段：'+', '.join(sorted(missing)))
    if len(frame)>100000:raise ValueError('网页任务最多10万条观测。')
    if frame.empty:raise ValueError('CSV没有观测记录。')
    frame['date']=pd.to_datetime(frame.date,errors='coerce')
    if frame.date.isna().any():raise ValueError('日期无法识别，请使用YYYY-MM-DD。')
    if frame.station_id.isna().any():raise ValueError('站点编号不能为空。')
    if frame.duplicated(['station_id','date']).any():raise ValueError('同站点同日有重复记录，请先确认合并方式。')
    if not frame.observed_event.dropna().isin([0,1]).all():raise ValueError('已核验事件标签只能是0或1，未知留空。')
    return frame,{'rows':len(frame),'stations':int(frame.station_id.nunique()),'start':str(frame.date.min().date()),'end':str(frame.date.max().date()),'unknown_labels':int(frame.observed_event.isna().sum()),'value_missing':int(frame.value.isna().sum())}

def anomaly_review(frame):
    d=frame.copy();d['value']=pd.to_numeric(d.value,errors='coerce')
    # Descriptive retrospective screening only: does not claim future prediction.
    median=d.groupby('station_id').value.transform('median')
    deviation=(d.value-median).abs();mad=deviation.groupby(d.station_id).transform('median')
    d['robust_deviation']=deviation/(1.4826*mad.replace(0,np.nan))
    return d[['station_id','date','value','robust_deviation']].sort_values('robust_deviation',ascending=False,na_position='last')

def execute_task(raw,plan,image_bytes=None,notify=lambda s:None,root=None):
    from .real_training.user_engine import build,run
    from .real_training.models import Preprocessor,tabular
    from .real_training.model_registry import estimator
    from .real_training.evaluation import metrics
    from sklearn.metrics import average_precision_score
    from threadpoolctl import threadpool_limits
    t0=time.perf_counter();events=[]
    def event(action,reason,status='done',**extra):
        row=dict(step=len(events)+1,action=action,reason=reason,status=status,elapsed_seconds=round(time.perf_counter()-t0,3),**extra)
        events.append(row);notify(action+'：'+reason)
    frame,checks=inspect_data(raw)
    event('检查数据',f"{checks['rows']}条观测；{checks['stations']}个站点",details=checks)
    visual=None
    if image_bytes:
        from .field_visual import load_image,make_result
        try:
            visual=make_result(load_image(image_bytes),{'sea_surface_confirmed':plan['sea_surface'],'location':plan['image_station'],'capture_date':plan['image_date']},root=root,requested_mode='规则基线')
            event('影像筛查',visual['priority_reason'],backend=visual['backend'])
        except Exception as exc:
            visual={'error':str(exc),'screening_priority':'需重拍或重新上传'}
            event('影像筛查失败','保留观测分析；未将失败影像作为有效证据','failed',error=str(exc))
    selected=None;selection=[];model_archive=None;table=pd.DataFrame();forecast=pd.DataFrame();interpretation=''
    if plan['mode']=='anomaly':
        review=anomaly_review(frame)
        event('调整分析路径','按已确认方案执行历史异常筛查；不训练预测模型、不报告预测准确率')
        interpretation='按站点中位数绝对偏差筛查历史异常。零离散度或缺失值不产生有效异常分数；结果需要现场复核。'
    else:
        ds,ix,manifest=build(raw,plan['horizon'],False)
        event('封存评估划分','候选选择只读取训练和验证窗口；测试窗口保持封存',splits=manifest['sha256'])
        prep=Preprocessor().fit(ds.X[ix['train']]);X=tabular(prep.transform(ds.X));y=ds.rows.y.to_numpy()
        baseline=float(average_precision_score(y[ix['validation']],np.full(len(ix['validation']),y[ix['train']].mean())))
        candidates=['Logistic','HistGradientBoosting','Extra Trees'][:plan['budget']]
        # Third experiment is justified by actual validation variability or calibration feedback.
        for name in candidates:
            if name=='Extra Trees' and selection:
                leader=max(selection,key=lambda r:r['AP'])
                if leader['AP']>baseline+.05 and leader['ECE']<=.10 and leader['AP_range']<=.03:
                    event('提前结束探索','验证表现达到预设稳定性与校准门槛，不消耗第三个模型预算');break
                event('追加对照','验证提升、校准或种子稳定性尚未同时满足门槛，运行额外树模型')
            scores=[];start=time.perf_counter()
            try:
                with threadpool_limits(limits=2):
                    for seed in [17,42,73]:
                        model=estimator(name,seed,len(ix['train'])).fit(X[ix['train']],y[ix['train']])
                        scores.append(metrics(y[ix['validation']],model.predict_proba(X[ix['validation']])[:,1]))
                row={'model':name,**{k:float(np.mean([r[k] for r in scores])) for k in ['AP','Brier','ECE']},'AP_range':float(np.ptp([r['AP'] for r in scores])),'seconds':time.perf_counter()-start}
                selection.append(row);event('验证候选',name,metrics=row)
            except Exception as exc:event('候选失败',name+'未进入候选排名','failed',error=str(exc))
        if not selection:raise ValueError('所有候选均失败，没有生成成功结论。')
        chosen=sorted(selection,key=lambda r:(-r['AP'],r['Brier'],r['seconds']))[0];selected=chosen['model']
        event('冻结模型',selected+'：按验证AP排序，平分时比较Brier与耗时；不再依据测试结果改选')
        table,forecast,interpretation,manifest,model_archive=run(raw,[selected],plan['horizon'],plan['description'],future=plan['future'],notify=notify)
        event('独立检验','冻结候选与季节基线在相同留出集评估；保留全部三个种子和负结果')
        review=pd.DataFrame()
    linked=bool(image_bytes and plan['image_station'] in set(frame.station_id) and ((frame.station_id==plan['image_station']) & frame.date.dt.date.astype(str).eq(plan['image_date'])).any())
    joint='未提供影像：本次仅分析观测数据。'
    if image_bytes:
        joint='影像与观测缺少同站点同日对应，仅保留独立证据。'
        if linked:
            joint='影像已与同站点同日观测对应。视觉异常只用于提示复核，不调整模型概率。'
            if not forecast.empty:
                aligned=forecast[forecast.site.eq('station:'+plan['image_station']) & pd.to_datetime(forecast.origin_date).dt.date.astype(str).eq(plan['image_date']) & forecast.model.eq(selected)]
                if not aligned.empty:
                    probability=float(aligned.probability.mean())
                    joint+=f' 对应起报时点的模型概率均值为{probability:.1%}；视觉复核优先级为{visual.get("screening_priority","未知")}。两者目标不同，不能互相确证或否定。'
    evidence={'joint_summary':joint,'mode' :'late_evidence_association','image_attached':bool(image_bytes),'station_date_match':linked,'image_station':plan['image_station'],'image_date':plan['image_date'],'image_sha256':hashlib.sha256(image_bytes).hexdigest() if image_bytes else None,'visual':visual,'scope':'影像与观测仅作证据关联，不改变预测概率；未训练端到端多模态预测器。'}
    if image_bytes:event('关联多模态证据','站点与日期匹配，可并列复核' if linked else '缺少同站点同日记录，仅展示独立影像证据',matched=linked)
    event('交付结果','保存计划、数据哈希、逐步日志及真实执行结果')
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        if model_archive:
            with zipfile.ZipFile(io.BytesIO(model_archive)) as source:
                for name in source.namelist():z.writestr('model_results/'+name,source.read(name))
        for name,obj in [('plan',plan),('trajectory',events),('evidence',evidence),('data_checks',checks)]:z.writestr(name+'.json',json.dumps(obj,ensure_ascii=False,indent=2,default=str))
        z.writestr('validation_selection.csv',pd.DataFrame(selection).to_csv(index=False))
        if not review.empty:z.writestr('historical_anomalies.csv',review.to_csv(index=False))
    return dict(events=events,table=table,forecast=forecast,interpretation=interpretation,evidence=evidence,selection=selection,selected=selected,review=review,archive=archive.getvalue())


def demonstration_csv():
    """Reproducible synthetic onboarding data, never presented as field evidence."""
    rng=np.random.default_rng(9);rows=[]
    for station in range(4):
        for i,date in enumerate(pd.date_range('2020-01-01',periods=420)):
            value=np.sin(i/8+station)+rng.normal(0,.4)
            rows.append(dict(station_id=str(station),date=str(date.date()),available_at=str(date.date()),latitude=25+station,longitude=120,observed_event=int(value>.5),value=value,source='SYNTHETIC_ONBOARDING_DEMO'))
    return pd.DataFrame(rows).to_csv(index=False).encode()


def render(root):
    st.markdown('### 研究助手')
    st.caption('提出目标 → 检查证据 → 确认方案 → 执行与复核')
    source_mode=st.radio('从哪里开始？',['使用示例数据','上传自己的数据'],horizontal=True,key='ra_source_mode')
    demo=source_mode=='使用示例数据'
    if demo:
        st.info('无需准备文件：使用4个站点的合成示例，体验完整分析流程。示例结果不代表真实海域性能。')
    st.caption('①确认目标和时效　②生成并确认计划　③查看结果。照片可不上传。')
    with st.container(border=True):
        goal=st.text_area('你希望解决什么问题？',placeholder='例如：比较未来7天的站点风险，并结合现场照片确定需要复核的站点。',value='比较站点未来风险，检验模型表现。' if demo else '',key='ra_goal_'+source_mode)
        c1,c2=st.columns(2)
        with c1:
            upload=None
            if demo:
                st.markdown('#### 示例已就绪')
                st.write('4个虚拟站点 · 420天 · 1,680条观测')
                st.download_button('下载合成示例CSV',demonstration_csv(),file_name='SYNTHETIC_DEMO.csv')
            else:
                upload=st.file_uploader('观测数据 · CSV',type=['csv'],key='ra_data')
            st.download_button('下载数据模板',','.join(REQUIRED)+'\n',file_name='observations_template.csv')
        with c2:
            photo=st.file_uploader('现场影像 · 可选',type=['png','jpg','jpeg'],key='ra_image')
            use_example_image=st.checkbox('同时体验合成水色图筛查',key='ra_example_image') if demo else False
            if use_example_image:st.caption('合成测试图，仅体验筛查流程，不是真实观测，也不作为模型训练证据。')
            station=st.text_input('影像站点编号',key='ra_station')
            date=st.date_input('拍摄日期',key='ra_date')
            surface=st.checkbox('照片主体是海面或水体',value=True,key='ra_surface')
    c1,c2,c3=st.columns(3)
    with c1:horizon=st.selectbox('确认预测时效',[7,14,30],format_func=lambda n:f'{n}天',key='ra_horizon')
    with c2:budget=st.selectbox('最多比较模型数',[2,3],index=1,key='ra_budget')
    with c3:future=st.checkbox('同时预测各站点下一采样时点',key='ra_future')
    description=st.text_input('事件与变量定义',placeholder='物种、事件阈值，以及value的含义和单位',value='合成演示事件：value为无单位信号，大于0.5记为事件；非真实藻华数据。' if demo else '',key='ra_definition_'+source_mode)
    st.caption('根据数据检查与实验反馈安排下一步；需要指定模型或远程API时，可切换到“自主配置”。')
    if not demo and not upload:
        st.info('没有CSV？可切换到使用示例数据。只有照片时，请使用顶部的影像识别工作区。');return
    raw=demonstration_csv() if demo else upload.getvalue();image_bytes=photo.getvalue() if photo else None
    if use_example_image and not photo:
        from globalhab_demo.visual_examples import sample_image
        image_bytes=sample_image('绿色')
    if image_bytes and len(image_bytes)>15*1024*1024:st.error('影像超过15MB，请缩小后上传。');return
    signature=hashlib.sha256(raw+(image_bytes or b'')+json.dumps([goal,horizon,budget,future,description,station,str(date),surface,VERSION],ensure_ascii=False).encode()).hexdigest()
    if st.button('检查数据并生成计划',type='primary',key='ra_plan'):
        st.session_state.pop('ra_result',None)
        try:
            frame,checks=inspect_data(raw)
            parsed=parse_goal(goal)
            mode='forecast';reason='使用训练窗口拟合，验证窗口选择候选，独立校准后进行最终检验。'
            from .real_training.user_engine import build
            try:build(raw,horizon,False)
            except Exception as exc:mode='anomaly';reason='无法建立合格预测划分：'+str(exc)+'。建议先做历史异常筛查。'
            if parsed['intent']=='历史异常筛查':mode='anomaly';reason='按需求回顾站点历史异常，不执行未来预测。'
            if parsed['requested_horizons'] and parsed['requested_horizons']!=[horizon]:
                st.error('需求中的时间窗口与所选时效不一致，请调整后重新生成计划。');return
            st.session_state['ra_plan_record']=(signature,dict(data_origin='synthetic_demo' if demo else 'user_upload',goal=goal,parsed_goal=parsed,description=description,horizon=horizon,budget=budget,future=future,mode=mode,reason=reason,input_sha256=hashlib.sha256(raw).hexdigest(),image_station=station,image_date=str(date),sea_surface=surface,version=VERSION,checks=checks))
        except Exception as exc:st.error(str(exc))
    saved=st.session_state.get('ra_plan_record')
    if not saved or saved[0]!=signature:
        st.info('请生成或更新计划。修改目标、数据或参数后，旧结果将隐藏。');return
    plan=saved[1]
    with st.container(border=True):
        st.markdown('#### 待确认方案')
        st.write(plan['reason'])
        st.write('模型预算：'+str(budget)+'类，每类3个随机种子。' if plan['mode']=='forecast' else '本次仅回顾历史异常，不提供预测准确率或未来预测。')
        if image_bytes:st.write('影像使用现有规则筛查器检查质量与视觉异常；按站点、日期关联，不参与预测模型选优。')
        confirm=st.checkbox('我确认目标、事件定义和分析方案',key='ra_confirm_'+signature[:12])
        run_now=st.button('确认并执行',disabled=not confirm or not goal.strip() or (plan['mode']=='forecast' and not description.strip()),key='ra_execute')
    if run_now:
        progress=st.status('研究任务执行中',expanded=True)
        try:
            result=execute_task(raw,plan,image_bytes,progress.write,root)
            st.session_state['ra_result']=(signature,result)
            from globalhab_demo.result_pool import register
            register('研究助手','合成示例' if demo else '用户观测',{'metrics':result['table'].to_dict('records'),'plan':plan,'evidence':result['evidence']})
            progress.update(label='研究任务完成',state='complete',expanded=False)
        except Exception as exc:
            progress.update(label='任务未完成',state='error');st.error(str(exc))
    result=st.session_state.get('ra_result')
    if not result or result[0]!=signature:return
    r=result[1]
    if demo:st.warning('以下为合成示例运行结果，仅供体验流程。')
    tabs=st.tabs(['分析结论','执行轨迹','多模态证据'])
    with tabs[0]:
        from globalhab_demo.result_charts import render_metrics
        render_metrics(r['table'],r['interpretation'])
        if r['table'].empty:st.write(r['interpretation'])
        if not r['review'].empty:st.dataframe(r['review'].head(30),hide_index=True)
        if not r['forecast'].empty:
            capacity=st.slider('优先复核比例（不重新训练）',5,50,20,5,key='ra_capacity')
            f=r['forecast'];f=f[f.model.eq(r['selected'])].groupby(['site','origin_date','label_date'],as_index=False).probability.mean().sort_values('probability',ascending=False)
            st.dataframe(f.head(max(1,int(np.ceil(len(f)*capacity/100)))),hide_index=True)
            st.caption('三个种子概率均值；无未来真实标签，不报告未来准确率。')
    with tabs[1]:
        for e in r['events']:
            with st.expander(str(e['step'])+' · '+e['action']+' · '+e['status']):st.json(e)
        if r['selection']:st.dataframe(pd.DataFrame(r['selection']),hide_index=True)
    with tabs[2]:
        if photo:st.image(image_bytes,width=400)
        st.write(r['evidence']['joint_summary'])
        st.caption(r['evidence']['scope'])
        st.write('站点与日期匹配：'+('是' if r['evidence']['station_date_match'] else '否 / 未提供影像'))
        if r['evidence']['visual']:
            visual=r['evidence']['visual']
            st.metric('影像复核优先级',visual.get('screening_priority','未完成'))
            st.write(visual.get('priority_reason',visual.get('error','')))
            with st.expander('影像检查明细'):st.json(visual)
    st.download_button('下载本次任务与执行证据',r['archive'],file_name='GlobalHAB_agent_task.zip')
