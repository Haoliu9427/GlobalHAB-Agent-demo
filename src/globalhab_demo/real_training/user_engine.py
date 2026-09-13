"""Common-split evaluation and genuinely unlabelled, latest-origin forecasts."""
import io,json,time,hashlib,tempfile,zipfile,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss
from .own_observations import prepare,foundation_score
from .models import Preprocessor,tabular,TemporalEstimator
from .evaluation import Calibrator,metrics,block_delta
from .model_registry import *

def build(content,horizon,future=False):
    ds,ix,m=prepare(content,horizon)
    d=pd.read_csv(io.BytesIO(content),dtype={'station_id':str});d['site']='station:'+d.station_id.astype(str)
    d['date']=pd.to_datetime(d.date);d=d.dropna(subset=['observed_event'])
    d['season_sin']=np.sin(2*np.pi*d.date.dt.dayofyear/365.25);d['season_cos']=np.cos(2*np.pi*d.date.dt.dayofyear/365.25)
    d['past_verified_event']=d.observed_event
    science=all(c in d for c in SCIENCE_COLUMNS+['upstream_available_at'])
    if science:
        availability=pd.to_datetime(d.upstream_available_at,errors='coerce')
        science=bool(availability.notna().all() and (availability<=d.date).all() and np.isfinite(d[SCIENCE_COLUMNS].to_numpy(dtype=float)).all())
    m['science_eligible']=science
    if science:ds.features+= [c for c in SCIENCE_COLUMNS if c not in ds.features]
    groups={s:g.sort_values('date').reset_index(drop=True) for s,g in d.groupby('site')}
    future_rows=[]
    if future:
        for site,g in groups.items():
            if len(g)<12:raise ValueError('未来预测站点历史不足12次：'+site)
            r=g.iloc[-1];future_rows.append(dict(sample_id='future:'+site+':'+str(r.date.date()),site=site,origin_date=r.date,
                label_date=r.date+pd.Timedelta(days=horizon),y=np.nan,latitude=r.latitude,longitude=r.longitude,source=r.source,
                last_observed_event=int(r.observed_event),target_value=np.nan))
        ix['future']=np.arange(len(ds.rows),len(ds.rows)+len(future_rows))
        ds.rows=pd.concat([ds.rows,pd.DataFrame(future_rows)],ignore_index=True)
    X=[];regular=True
    for row in ds.rows.itertuples():
        g=groups[row.site];end=int(g.date.searchsorted(row.origin_date,side='right'));g=g.iloc[end-12:end].copy()
        g['gap_days']=g.date.diff().dt.total_seconds().div(86400).fillna(0)
        regular=regular and bool((g.gap_days.iloc[1:]==1).all() and row.label_date-row.origin_date==pd.Timedelta(days=horizon))
        X.append(g[ds.features].to_numpy(dtype='float32'))
    ds.X=np.asarray(X);m.update(features=ds.features,chronos_eligible=regular,future_origins=len(future_rows))
    return ds,ix,m

def light_fit(X,y,train,val,seed,epochs):
    import torch,copy
    from torch import nn
    torch.manual_seed(seed);torch.set_num_threads(2)
    class Light(nn.Module):
        def __init__(self):
            super().__init__();self.a=nn.Conv1d(X.shape[2],4,3);self.b=nn.Conv1d(4,3,3,dilation=2);self.head=nn.Linear(3,1)
        def forward(self,x):
            h=torch.relu(self.a(nn.functional.pad(x.transpose(1,2),(2,0))))
            h=torch.relu(self.b(nn.functional.pad(h,(4,0))))
            return self.head(h[:,:,-1]).squeeze(-1)
    net=Light();opt=torch.optim.Adam(net.parameters(),lr=.01);t=torch.tensor(X);best=float('inf');state=None
    for epoch in range(epochs):
        net.train();opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(t[train]),torch.tensor(y[train],dtype=torch.float32));loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(t[val])).numpy()
        score=log_loss(y[val],p,labels=[0,1])
        if score<best:best=score;state=copy.deepcopy(net.state_dict())
    net.load_state_dict(state);net.eval()
    def predict(v):
        with torch.no_grad():return torch.sigmoid(net(torch.tensor(v))).numpy()
    return predict,net

def sts_arrays(X,features):
    def col(n):return X[:,:,features.index(n)]
    common=[col(n) for n in ['nitrate','phosphate','silicate','season_sin','season_cos']]
    local=np.stack([col('local_raw_signal')]+common,axis=-1)
    up=np.stack([col('upstream_raw_signal')]+common,axis=-1)
    gate=col('circulation_residence_proxy')[:,-1]
    transmitted=col('upstream_raw_signal')[:,-1]*(1/(1+np.exp(-gate)))
    nutrient=.5*col('nitrate')[:,-1]+.3*col('phosphate')[:,-1]+.2*col('silicate')[:,-1]
    skip=np.column_stack([transmitted,nutrient,transmitted*nutrient,col('season_sin')[:,-1],col('season_cos')[:,-1]])
    return local,up,gate,skip

def run(content,selected,horizon=7,description='',epochs=20,future=False,notify=lambda s:None):
    if not description.strip() or len(description)>500:raise ValueError('请填写事件阈值、物种与value含义（1—500字）。')
    if not selected:raise ValueError('请选择模型')
    if horizon not in [7,14,30] or not 1<=epochs<=50:raise ValueError('不支持的时效或训练轮数')
    names=expand(['Seasonal Climatology']+selected);ds,ix,m=build(content,horizon,future)
    for n in names:
        if missing(n):raise ValueError(n+'缺少依赖：'+', '.join(missing(n)))
    if any(n.startswith('Chronos') for n in names) and not m['chronos_eligible']:raise ValueError('Chronos需要连续逐日历史且目标在准确第N天。不会补造标签。')
    if any(n.startswith('STS-') for n in names) and not m['science_eligible']:raise ValueError('STS模型需要'+','.join(SCIENCE_COLUMNS)+'及不晚于当前日期的upstream_available_at；当前数据不适用。')
    eval_ids=np.unique(np.concatenate([ids for k,ids in ix.items() if k!='train']))
    if any(n in FOUNDATIONS for n in names) and len(eval_ids)>2000:raise ValueError('基础模型评估超过2000条网页上限；不会缩小同一留出集，请定义较小研究任务。')
    t0=time.perf_counter();prep=Preprocessor().fit(ds.X[ix['train']]);X=prep.transform(ds.X);Z=tabular(X);y=ds.rows.y.to_numpy()
    m.update(selected=selected,executed=names,seeds=[17,42,73],foundation_seed=42,description=description,epochs=epochs,
       preprocessing_seconds=time.perf_counter()-t0,forecast_training='Frozen train-partition model; latest observed history, no future target label',
       adaptations='Classifier families retrained on user features; lightweight TCN input dimension adapted; STS uses supplied prealigned signals and train-scaled environmental features, not synthetic constants.',
       cost_note='seconds sums component costs for fusion and may count reused components; not total task wall time',
       date_note='Future origin is latest labelled observation per station, not necessarily today. Generic target: first sampled event at N to N+3 days; forecasts are conditional on sampling in that window.')
    bounds=[]
    for j in range(len(ds.features)):
        values=ds.X[ix['train'],:,j].ravel();values=values[np.isfinite(values)]
        bounds.append(np.quantile(values,[.01,.99]) if len(values) else [np.nan,np.nan])
    bounds=np.asarray(bounds);last=ds.X[:,-1,:]
    outside=((last<bounds[:,0])|(last>bounds[:,1])).sum(axis=1)
    missing_fraction=(~np.isfinite(ds.X)).mean(axis=(1,2))
    tables=[];preds=[];costs={};scores={};params={};audits=[];choices=[];paired=[];explainers=[];training_warnings=[]
    with tempfile.TemporaryDirectory(prefix='hab_user_') as tmp:
        out=Path(tmp);joblib.dump(prep,out/'preprocessor.joblib')
        for n in names:
            for seed in ([42] if n in FOUNDATIONS else [17,42,73]):
                notify(n+' · '+str(seed));start=time.perf_counter();parameter=None
                if n in FOUNDATIONS:
                    p,meta=foundation_score(n,ds,eval_ids,horizon,description,notify);raw=np.full(len(y),np.nan);raw[eval_ids]=p;parameter=meta['parameters'];audits.append(meta)
                elif n in FUSIONS:
                    a,b=FUSIONS[n];sa=42 if a in FOUNDATIONS else seed
                    pa=scores[(a,sa)];pb=scores[(b,seed)];v=ix['validation']
                    weight=min([0.,.25,.5,.75,1.],key=lambda w:log_loss(y[v],w*pa[v]+(1-w)*pb[v],labels=[0,1]))
                    raw=weight*pa+(1-weight)*pb;choices.append(dict(model=n,seed=seed,first_component=a,first_weight=weight,selection='validation'))
                    parameter=None
                elif n in BASELINES:
                    if n=='Seasonal Climatology':B=ds.X[:,-1,[ds.features.index('season_sin'),ds.features.index('season_cos')]]
                    else:B=ds.X[:,-1,[ds.features.index('past_verified_event'),ds.features.index('season_sin'),ds.features.index('season_cos')]]
                    model=estimator('Logistic',seed,len(ix['train'])).fit(B[ix['train']],y[ix['train']]);raw=model.predict_proba(B)[:,1];joblib.dump(model,out/f'{n}_{seed}.joblib')
                elif n in TABULAR or n=='STS-Interaction GLM':
                    B=sts_arrays(X,ds.features)[3] if n.startswith('STS-') else Z
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model=estimator('Logistic' if n.startswith('STS-') else n,seed,len(ix['train'])).fit(B[ix['train']],y[ix['train']])
                    training_warnings.extend({'model':n,'seed':seed,'warning':str(w.message)} for w in caught)
                    raw=model.predict_proba(B)[:,1];joblib.dump(model,out/f'{n}_{seed}.joblib')
                    if hasattr(model,'coef_'):parameter=int(model.coef_.size+model.intercept_.size)
                    if n in TABULAR:explainers.append((log_loss(y[ix['validation']],raw[ix['validation']],labels=[0,1]),n,seed,model))
                elif n=='EcoTemporalNet':
                    net=TemporalEstimator(len(ds.features),16,seed,epochs).fit(X[ix['train']],y[ix['train']],X[ix['validation']],y[ix['validation']]);raw=net.predict_proba(X);parameter=net.parameters
                    import torch
                    torch.save(net.net.state_dict(),out/f'{n}_{seed}.pt')
                elif n=='Lightweight TCN':
                    predict,net=light_fit(X,y,ix['train'],ix['validation'],seed,epochs);raw=predict(X);parameter=sum(p.numel() for p in net.parameters())
                    import torch
                    torch.save(net.state_dict(),out/f'{n}_{seed}.pt')
                elif n=='STS-Gated TCN':
                    from ..sts_gated_tcn import _fit,_forward
                    arrays=sts_arrays(X,ds.features);best=None
                    for budget in sorted(set([max(1,epochs//2),epochs])):
                        weights,_=_fit(*[a[ix['train']] for a in arrays],y[ix['train']],seed,budget)
                        pr=_forward(*arrays,weights)[0];loss=log_loss(y[ix['validation']],pr[ix['validation']],labels=[0,1])
                        if best is None or loss<best[0]:best=(loss,weights,pr,budget)
                    _,weights,raw,budget=best;parameter=sum(a.size for a in weights.values());np.savez(out/f'{n}_{seed}.npz',**weights)
                    choices.append(dict(model=n,seed=seed,epochs=budget,selection='validation'))
                if not np.isfinite(raw[eval_ids]).all():raise ValueError(n+'输出无效；比较没有完成')
                scores[(n,seed)]=raw;cal=Calibrator().fit(raw[ix['calibration']],y[ix['calibration']]);joblib.dump(cal,out/f'calibrator_{n}_{seed}.joblib')
                seconds=time.perf_counter()-start
                if n in FUSIONS:
                    a,b=FUSIONS[n];seconds+=costs[(a,42 if a in FOUNDATIONS else seed)]+costs[(b,seed)]
                costs[(n,seed)]=seconds
                for part in ['test','spatial_test','future']:
                    if part not in ix or not len(ix[part]):continue
                    ids=ix[part];p=cal.predict(raw[ids]);rows=ds.rows.iloc[ids].copy();rows['probability']=p;rows['model']=n;rows['seed']=seed;rows['split']=part;rows['outside_training_range_features']=outside[ids];rows['missing_fraction']=missing_fraction[ids];preds.append(rows)
                    if part=='future':continue
                    tables.append(dict(model=n,seed=seed,split=part,seconds=seconds,parameters=parameter,**metrics(y[ids],p)))
                    if n!='Seasonal Climatology':
                        reference=next(f for f in preds if f.model.iloc[0]=='Seasonal Climatology' and f.seed.iloc[0]==seed and f.split.iloc[0]==part)
                        blocks=rows.origin_date.dt.to_period('Q').astype(str).to_numpy()
                        ci=block_delta(y[ids],p,reference.probability.to_numpy(),blocks,200,seed)
                        paired.append(dict(model=n,seed=seed,split=part,blocks=len(set(blocks)),reference='Seasonal Climatology',**ci))
        table=pd.DataFrame(tables);pred=pd.concat(preds);ci=pd.DataFrame(paired)
        # Deterministic interpretation of this run, not a language-model narrative.
        lines=['本次数据SHA256：'+m['input_sha256'],f"训练/验证/校准/测试样本：{m['counts']}", '所有成绩由本次数据重新计算。基础模型为固定推理一次，其他模型三个种子；种子不等于独立海域。']
        means=table[table.split=='test'].groupby('model')[['AP','Brier','ECE']].mean()
        lines+=['各模型的测试期表现（描述性比较，不能据此反复选参）：',means.to_string()]
        lines.append('本次测试期平均AP最高：'+str(means.AP.idxmax())+'；Brier最低：'+str(means.Brier.idxmin())+'；ECE最低：'+str(means.ECE.idxmin())+'。这是事后描述，不等于训练期选模或稳定胜出。')
        for n in means.index:
            q=ci[(ci.model==n)&(ci.split=='test')] if len(ci) else pd.DataFrame()
            stable=not any(w['model']==n for w in training_warnings) and len(q)>0 and (q.blocks>=4).all() and (q.delta_ap_low>0).all()
            lines.append(n+('：本协议下各已运行种子对季节基线的AP差值区间下限大于0；仍需检查Brier/ECE。' if stable else '：未证明跨采样块的稳定优势，保留该结果。'))
        for choice in choices:
            if 'first_weight' in choice:
                lines.append(choice['model']+'，种子'+str(choice['seed'])+'：验证期选择的'+choice['first_component']+'权重为'+str(choice['first_weight'])+('，该组件未被融合采用。' if choice['first_weight']==0 else '。'))
        if future:lines+=['未来预测无真实标签，不能计算未来AP或准确率。起报日期见各站origin_date；输出为已校准模型概率，不是死亡率或运营指令。']
        lines.append('逐条结果附带超出训练期1%—99%范围的字段数及历史缺测比例。这些是分布变化提示，不是错误概率或可靠性保证。')
        lines+=['相关特征、分布变化和观测选择会影响表现，以上不证明因果关系。STS的输入必须来自已对齐的真实上游观测，不自动推断传播网络。']
        importance=[]
        if explainers:
            _,ename,eseed,emodel=min(explainers,key=lambda item:item[0])
            ecal=joblib.load(out/f'calibrator_{ename}_{eseed}.joblib');ids=ix['test']
            original=metrics(y[ids],ecal.predict(emodel.predict_proba(Z[ids])[:,1]))['AP']
            for j,feature in enumerate(ds.features):
                changed=X[ids].copy();changed[:,:,j]=0;changed[:,:,j+len(ds.features)]=0
                after=metrics(y[ids],ecal.predict(emodel.predict_proba(tabular(changed))[:,1]))['AP']
                importance.append(dict(model=ename,seed=eseed,feature=feature,AP_drop=original-after,model_selection='validation log loss'))
            top=sorted(importance,key=lambda r:r['AP_drop'],reverse=True)[:3]
            lines.append('输入遮蔽敏感性（验证期选定模型'+ename+'）：'+str([(r['feature'],round(r['AP_drop'],4)) for r in top])+ '。这是移除输入后的AP变化，不是因果贡献；负值表示移除后表现更好。')
        else:lines.append('本次未运行可用的表格模型，不生成输入贡献排名。')
        pd.DataFrame(importance).to_csv(out/'input_sensitivity.csv',index=False)
        interpretation='\n\n'.join(lines)
        pd.DataFrame(training_warnings).to_csv(out/'training_warnings.csv',index=False)
        if training_warnings:interpretation+='\n\n训练警告（含可能未收敛）：'+str(training_warnings)
        table.to_csv(out/'metrics.csv',index=False);pred[pred.split!='future'].to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
        pred[pred.split=='future'].to_csv(out/'future_predictions.csv',index=False);ci.to_csv(out/'paired_intervals.csv',index=False)
        pd.DataFrame(choices).to_csv(out/'model_selection.csv',index=False)
        (out/'interpretation.txt').write_text(interpretation);(out/'protocol.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
        (out/'foundation_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
        for k,ids in ix.items():ds.rows.iloc[ids].to_csv(out/f'{k}_rows.csv',index=False)
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():z.write(p,p.name)
        return table,pred[pred.split=='future'],interpretation,m,buffer.getvalue()
