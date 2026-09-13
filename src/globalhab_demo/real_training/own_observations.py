"""Session-local model comparisons for uploaded, labelled field observations."""
import io,json,time,hashlib,zipfile,tempfile,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from .data import field,split
from .models import Preprocessor,tabular,classical,TemporalEstimator
from .evaluation import metrics,Calibrator
CLASSICAL=['Logistic','RandomForest','HistGradientBoosting','EcoTemporalNet']
FOUNDATION=['Chronos-Bolt-small','Qwen2.5-0.5B-Instruct']

def prepare(content,horizon=7):
    if len(content)>20*1024*1024:raise ValueError('上传上限20MB，请使用本地脚本处理更大数据。')
    frame=pd.read_csv(io.BytesIO(content),dtype={'station_id':str})
    if len(frame)>100000:raise ValueError('网页最多处理100000条原始观测；不会截取部分数据冒充完整比较。')
    # Preserve identifiers such as 001 when the shared adapter reads CSV.
    frame['station_id']='station:'+frame['station_id'].fillna('').astype(str)
    if (frame.station_id=='station:').any():raise ValueError('station_id不能为空')
    ds=field(io.StringIO(frame.to_csv(index=False)),horizon)
    frame['date']=pd.to_datetime(frame.date);frame=frame.dropna(subset=['observed_event'])
    histories=[];regular=[]
    groups={s:g.sort_values('date').reset_index(drop=True) for s,g in frame.groupby('station_id')}
    for row in ds.rows.itertuples():
        g=groups[row.site];end=int(g.date.searchsorted(row.origin_date,side='right'));g=g.iloc[max(0,end-ds.X.shape[1]):end]
        gaps=g.date.diff().dt.total_seconds().div(86400).fillna(0).to_numpy()
        histories.append(np.stack([g.observed_event.to_numpy(dtype=float),gaps],axis=1))
        regular.append(bool(np.all(gaps[1:]==1) and (row.label_date-row.origin_date).days==horizon))
    ds.X=np.concatenate([ds.X,np.asarray(histories,dtype='float32')],axis=2)
    ds.features+=['past_verified_event','gap_days']
    ix,manifest=split(ds)
    manifest.update(input_sha256=hashlib.sha256(content).hexdigest(),horizon_days=horizon,
        features=ds.features,chronos_eligible=bool(all(regular)),
        target='First observed event label at horizon to horizon+3 days; not any event within the interval',
        histories='Past verified events included; target labels excluded. Chronos requires daily histories and exact target day.')
    return ds,ix,manifest

def dependencies(name):
    modules={'EcoTemporalNet':['torch'],'Chronos-Bolt-small':['torch','chronos'],'Qwen2.5-0.5B-Instruct':['torch','transformers']}.get(name,[])
    return [x for x in modules if importlib.util.find_spec(x) is None]

def foundation_score(name,ds,ids,horizon,description,notify=lambda x:None):
    import torch
    torch.set_num_threads(2);torch.manual_seed(42)
    start=time.perf_counter();event_index=ds.features.index('past_verified_event')
    if name=='Chronos-Bolt-small':
        from chronos import BaseChronosPipeline
        pipe=BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-small',device_map='cpu')
        scores=[];levels=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
        for offset in range(0,len(ids),64):
            context=torch.tensor(ds.X[ids[offset:offset+64],:,event_index],dtype=torch.float32)
            q,_=pipe.predict_quantiles(context,prediction_length=horizon,quantile_levels=levels)
            if torch.is_tensor(q):q=q.detach().cpu().numpy()
            for values in np.asarray(q)[:,horizon-1,:]:
                # Quantile interpolation is a score, calibrated on user's pre-test window.
                scores.append(float(1-np.interp(.5,np.maximum.accumulate(values),levels,left=.1,right=.9)))
            notify(f'Chronos 已处理 {min(offset+64,len(ids))}/{len(ids)} 条')
        meta={'model_id':'amazon/chronos-bolt-small','parameters':sum(p.numel() for p in pipe.model.parameters()),
            'revision':getattr(pipe.model.config,'_commit_hash',None),'input':'past verified binary events only; daily cadence',
            'protocol':'horizon-day quantile interpolation at 0.5; pre-test Platt calibration'}
    else:
        from transformers import AutoTokenizer,AutoModelForCausalLM
        model_id='Qwen/Qwen2.5-0.5B-Instruct'
        tok=AutoTokenizer.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32).eval()
        labels=[tok.encode(str(i),add_special_tokens=False) for i in [0,1]]
        if any(len(x)!=1 for x in labels):raise ValueError('Tokenizer必须支持单token二分类标签')
        prompt=('Predict the binary observed_event at the first sampled date between '+str(horizon)+' and '+str(horizon+3)+
            ' days after the last history row. Only past observations are supplied, oldest first. '
            'past_verified_event contains historical labels, gap_days contains elapsed days. Null means missing, never absence. '
            'Return only 1 or 0. Task definition is user supplied: '+description+'\n')
        scores=[];tokens=0
        for j,i in enumerate(ids):
            history=[[float(v) if np.isfinite(v) else None for v in row] for row in ds.X[i]]
            text=prompt+json.dumps({'features':ds.features,'history':history},ensure_ascii=False,allow_nan=False)
            chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
            inputs=tok(chat,return_tensors='pt');tokens+=int(inputs.input_ids.numel())
            if inputs.input_ids.shape[1]>4096:raise ValueError('单条历史超过4096token；未截断或丢弃样本，请减少可选变量。')
            with torch.no_grad():
                logits=model(**inputs).logits[0,-1,[labels[0][0],labels[1][0]]]
                scores.append(float(torch.softmax(logits.float(),dim=-1)[1]))
            if j%10==0 or j==len(ids)-1:notify(f'Qwen 已处理 {j+1}/{len(ids)} 条')
        meta={'model_id':model_id,'parameters':sum(p.numel() for p in model.parameters()),'revision':getattr(model.config,'_commit_hash',None),
            'prompt_template':prompt,'input_tokens':tokens,'protocol':'Frozen zero-shot token01 score; pre-test Platt calibration'}
    meta.update(seconds=time.perf_counter()-start,seed=42,seed_note='Frozen deterministic inference, not multiple retrainings',pretraining_overlap='Not independently excluded')
    return np.asarray(scores),meta

def execute(content,selected,horizon=7,description='',epochs=20,notify=lambda x:None):
    if not selected or set(selected)-set(CLASSICAL+FOUNDATION):raise ValueError('请选择已支持的模型')
    if not description.strip() or len(description)>500:raise ValueError('请用1—500字说明物种、事件阈值和value的含义/单位')
    if epochs<1 or epochs>50:raise ValueError('epochs需在1—50之间')
    ds,ix,manifest=prepare(content,horizon)
    for name in selected:
        missing=dependencies(name)
        if missing:raise ValueError(name+'缺少依赖：'+','.join(missing)+'；请按自有观测说明安装可选运行环境。')
    if 'Chronos-Bolt-small' in selected and not manifest['chronos_eligible']:
        raise ValueError('Chronos要求每个样本历史为连续逐日观测，且目标恰好在所选第N天；不对不规则采样补造标签。其他模型可处理间隔特征。')
    ids=np.unique(np.concatenate([ix[k] for k in ix if k!='train']))
    if set(selected)&set(FOUNDATION) and len(ids)>2000:raise ValueError('基础模型网页运行上限2000条评估历史；本次未运行，也未抽样缩小留出集。请缩小上传研究范围后建立新任务。')
    start=time.perf_counter();prep=Preprocessor().fit(ds.X[ix['train']]);X=prep.transform(ds.X);Z=tabular(X);y=ds.rows.y.to_numpy()
    manifest.update(description=description,models=selected,seeds=[17,42,73],foundation_seed=42,
        parameter_selection='Fixed classical hyperparameters; TCN early stopping on validation only; calibration separate',epochs=epochs,
        preprocessing_seconds=time.perf_counter()-start,framework='User-data evaluation, independent from frozen published experiments')
    reports=[];preds=[];audits=[]
    with tempfile.TemporaryDirectory(prefix='globalhab_user_') as tmp:
        out=Path(tmp);import joblib
        joblib.dump(prep,out/'preprocessor.joblib')
        for name in selected:
            for seed in ([42] if name in FOUNDATION else [17,42,73]):
                notify(f'正在运行 {name} · 种子 {seed}');t0=time.perf_counter();parameters=None
                if name in FOUNDATION:
                    s,meta=foundation_score(name,ds,ids,horizon,description,notify);raw=np.full(len(y),np.nan);raw[ids]=s;parameters=meta['parameters'];audits.append(meta)
                elif name=='EcoTemporalNet':
                    net=TemporalEstimator(len(ds.features),width=16,seed=seed,epochs=epochs).fit(X[ix['train']],y[ix['train']],X[ix['validation']],y[ix['validation']])
                    raw=net.predict_proba(X);parameters=net.parameters
                    import torch
                    torch.save(net.net.state_dict(),out/f'{name}_{seed}.pt')
                    (out/f'{name}_{seed}.json').write_text(json.dumps({'features':len(ds.features),'width':16,'seed':seed,'best_epoch':net.best_epoch}))
                else:
                    model=classical(name,seed).fit(Z[ix['train']],y[ix['train']]);raw=model.predict_proba(Z)[:,1]
                    joblib.dump(model,out/f'{name}_{seed}.joblib')
                    if name=='Logistic':parameters=int(model.coef_.size+model.intercept_.size)
                if not np.isfinite(raw[ids]).all():raise ValueError(name+'产生无效结果；整个比较未标为成功')
                cal=Calibrator().fit(raw[ix['calibration']],y[ix['calibration']]);joblib.dump(cal,out/f'calibrator_{name}_{seed}.joblib')
                elapsed=time.perf_counter()-t0
                for part in ['test','spatial_test']:
                    if part not in ix or not len(ix[part]):continue
                    take=ix[part];p=cal.predict(raw[take]);reports.append(dict(model=name,seed=seed,split=part,seconds=elapsed,parameters=parameters,**metrics(y[take],p)))
                    rows=ds.rows.iloc[take].copy();rows['probability']=p;rows['model']=name;rows['seed']=seed;rows['split']=part;preds.append(rows)
        table=pd.DataFrame(reports)
        table.to_csv(out/'metrics.csv',index=False);pd.concat(preds).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
        (out/'protocol.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(out/'foundation_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
        for k,take in ix.items():ds.rows.iloc[take].to_csv(out/f'{k}_rows.csv',index=False)
        (out/'README.txt').write_text('本包为自有观测独立运行结果。输入CSV未存入结果包，请自行保留。protocol.json记录输入哈希。基础模型权重不包含；下载的权重缓存在运行主机。所有模型在同一留出集上评估，不能用测试集反复调参后宣称独立验证。joblib/pt仅加载自己信任的结果。未证明稳定提升时保留负结果。')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for f in out.iterdir():z.write(f,f.name)
        return table,manifest,buffer.getvalue()

def render():
    import streamlit as st
    st.write('上传带有事件标签的连续观测，选择模型，在相同时间留出集上训练和比较。')
    upload=st.file_uploader('现场观测CSV',type=['csv'],key='real_training_upload')
    st.download_button('下载字段模板',data='station_id,date,available_at,latitude,longitude,observed_event,value,source,temperature,salinity,dissolved_oxygen\n',file_name='field_observations_template.csv')
    horizon=st.selectbox('预测目标距当前的天数',[7,14,30],key='own_horizon')
    desc=st.text_area('事件与变量定义',placeholder='例如：物种、observed_event=1的判定阈值、value的含义和单位',max_chars=500,key='own_description')
    st.caption('目标为所选天数至其后3天内首次复测的事件标签；不是这段时间内任意一天发生事件。未知标签留空，不作为阴性。')
    left,right=st.tabs(['训练与验证','基础模型对照'])
    with left:
        basic=st.multiselect('训练模型',CLASSICAL,default=['Logistic','HistGradientBoosting'],key='own_basic')
        epochs=st.slider('TCN训练轮数上限',5,50,20,key='own_epochs')
        st.caption('三个固定种子；TCN参数少于50000，早停仅使用验证期。')
        run_basic=st.button('运行训练与验证',key='own_run_basic')
    with right:
        foundation=st.multiselect('基础模型',FOUNDATION,default=FOUNDATION,key='own_foundation')
        st.caption('Chronos使用连续逐日的历史事件序列；Qwen使用字段及历史观测。模型在运行主机本地推理，首次使用需下载权重。')
        refs=st.multiselect('同场训练对照',CLASSICAL,default=['Logistic'],key='own_refs')
        for name in foundation:
            missing=dependencies(name)
            if missing:st.info(name+'尚未安装：'+', '.join(missing)+'。安装方法见 OWN_OBSERVATIONS_GUIDE.md。')
        run_foundation=st.button('运行基础模型对照',key='own_run_foundation')
    content=upload.getvalue() if upload else None
    signature=hashlib.sha256((content or b'')+json.dumps([horizon,desc,basic,foundation,refs,epochs]).encode()).hexdigest()
    if content:
        try:
            _,_,info=prepare(content,horizon)
            st.write('数据检查通过：',info['counts'])
            with st.expander('留出日期与样本校验'):st.json(info)
        except Exception as exc:st.error('数据检查未通过：'+str(exc));st.session_state.pop('own_result',None);return
    if run_basic or run_foundation:
        st.session_state.pop('own_result',None)
        if not content:st.error('请先上传CSV。');return
        selection=basic if run_basic else list(dict.fromkeys(refs+foundation))
        if run_foundation and not foundation:st.error('请选择至少一个基础模型。');return
        progress=st.empty()
        try:
            with st.spinner('正在运行，完成后可下载完整结果。'):
                result,manifest,archive=execute(content,selection,horizon,desc,epochs,progress.write)
            st.session_state['own_result']=(signature,result,manifest,archive)
            progress.success('运行完成')
        except Exception as exc:progress.error('运行未完成：'+str(exc));return
    cached=st.session_state.get('own_result')
    if cached and cached[0]==signature:
        _,result,manifest,archive=cached
        st.subheader('本次运行结果')
        st.dataframe(result,hide_index=True,use_container_width=True)
        st.caption('基础模型为一次固定推理；传统模型/TCN为三个种子。AP越高越好，Brier与ECE越低越好；不自动宣称优于基线。')
        st.download_button('下载本次完整结果',data=archive,file_name='GlobalHAB_user_observations_results.zip')
    elif cached:st.info('数据或设置已改变，请重新运行；旧结果未作为当前结果展示。')
