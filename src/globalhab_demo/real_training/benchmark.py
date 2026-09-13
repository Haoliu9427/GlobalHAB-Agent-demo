"""Train/validation/calibration/test pipeline with exact common sample IDs."""
import json,hashlib,time,platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,average_precision_score
from .data import split
from .models import Preprocessor,TemporalEstimator,classical,tabular
from .evaluation import metrics,Calibrator,block_delta,conformal
from .agent import choose_capacity

def run(ds,out,dates=None,seeds=(17,42,73),epochs=20,foundation=False,bootstrap=200,holdout=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ix,manifest=split(ds,dates,holdout)
    prep=Preprocessor().fit(ds.X[ix['train']]);X=prep.transform(ds.X);Z=tabular(X);y=ds.rows.y.to_numpy()
    joblib.dump(prep,out/'preprocessor.joblib')
    manifest.update(features=ds.features,sequence_length=ds.X.shape[1],raw_data_hash=hashlib.sha256(ds.X.tobytes()).hexdigest(),
        seeds=list(seeds),environment={'python':platform.python_version(),'platform':platform.platform()},
        parameter_cap=50000,evaluation='Retrospective frozen test; no test-based tuning; rolling histories include only already observed outcomes.')
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    for k,v in ix.items():ds.rows.iloc[v].to_csv(out/f'{k}_rows.csv',index=False)
    availability=[]
    for part,ids in ix.items():
        for j,feature in enumerate(ds.features):
            values=ds.X[ids,-1,j];values=values[np.isfinite(values)]
            availability.append(dict(partition=part,feature=feature,missing_fraction=1-len(values)/len(ids),
                observed_count=len(values),median=float(np.median(values)) if len(values) else None))
    pd.DataFrame(availability).to_csv(out/'input_availability.csv',index=False)
    train,val,cal=ix['train'],ix['validation'],ix['calibration']
    report=[];predictions=[];stability=[];importance=[];selected=[];controls=[]
    fscore=None;fmeta=None
    if foundation:
        if ds.task!='china_hk_7d': raise ValueError('Foundation benchmark requires the equally spaced weekly HK task.')
        from .foundation import chronos_scores
        start=min(val)
        try:
            suffix,fmeta=chronos_scores(ds.X[start:]);fscore=np.full(len(X),np.nan);fscore[start:]=suffix
            (out/'foundation_status.json').write_text(json.dumps({'status':'completed',**fmeta},indent=2))
        except Exception as e:
            (out/'foundation_status.json').write_text(json.dumps({'status':'not_run','reason':str(e)},indent=2))
            print('Foundation unavailable:',e,flush=True)
    for seed in seeds:
        raw={};costs={};params={};models={}
        print(ds.task,'seed',seed,'training',len(train),flush=True)
        # Seasonal and persistence references trained only on train; still calibrate all equally.
        months=ds.rows.origin_date.dt.month.to_numpy()
        seasonal={m:(y[train][months[train]==m].sum()+1)/(np.sum(months[train]==m)+2) for m in range(1,13)}
        raw['Seasonal']=np.array([seasonal[m] for m in months]);costs['Seasonal']=0.;params['Seasonal']=12
        raw['Persistence']=.02+.96*ds.rows.last_observed_event.to_numpy();costs['Persistence']=0.;params['Persistence']=1
        for name in ['Logistic','RandomForest','HistGradientBoosting']:
            start=time.perf_counter();m=classical(name,seed).fit(Z[train],y[train])
            raw[name]=m.predict_proba(Z)[:,1];models[name]=m;costs[name]=time.perf_counter()-start
            params[name]=int(m.coef_.size+m.intercept_.size) if name=='Logistic' else None
            joblib.dump(m,out/f'{name}_{seed}.joblib')
        # Adaptive experiment choice observes validation feedback only; first small,
        # then tries more capacity if underfitting or less capacity if overfitting.
        tried=[];feedback=[];best=None
        for step in range(2):
            width,reason=choose_capacity(None if step==0 else {k:feedback[-1][k] for k in ['train_loss','val_loss']})
            start=time.perf_counter();net=TemporalEstimator(len(ds.features),width,seed,epochs).fit(X[train],y[train],X[val],y[val])
            pr=net.predict_proba(X);elapsed=time.perf_counter()-start
            entry={'step':step+1,'seed':seed,'width':width,'train_loss':float(log_loss(y[train],pr[train])),
                'val_loss':float(log_loss(y[val],pr[val])),'val_AP':float(average_precision_score(y[val],pr[val])),
                'parameters':net.parameters,'best_epoch':net.best_epoch,'seconds':elapsed,
                'decision_reason':reason,
                'visible_partitions':['train','validation'],'test_visible':False}
            feedback.append(entry);tried.append((net,pr,elapsed,entry))
            if best is None or entry['val_loss']<best[3]['val_loss']:best=tried[-1]
        with (out/'real_agent_log.jsonl').open('a') as f:
            for row in feedback:f.write(json.dumps(row)+'\n')
        net,pr,_,entry=best;raw['EcoTemporalNet']=pr;costs['EcoTemporalNet']=sum(v[2] for v in tried);params['EcoTemporalNet']=net.parameters
        import torch
        torch.save(net.net.state_dict(),out/f'EcoTemporalNet_{seed}.pt')
        (out/f'EcoTemporalNet_{seed}.json').write_text(json.dumps({'features':len(ds.features),'width':entry['width'],'seed':seed}))
        pd.DataFrame(net.learning_curve).to_csv(out/f'learning_curve_{seed}.csv',index=False)
        weights=np.linspace(0,1,5)
        w=min(weights,key=lambda v:log_loss(y[val],v*raw['EcoTemporalNet'][val]+(1-v)*raw['HistGradientBoosting'][val]))
        raw['EcoFusion']=w*raw['EcoTemporalNet']+(1-w)*raw['HistGradientBoosting'];costs['EcoFusion']=costs['EcoTemporalNet']+costs['HistGradientBoosting'];params['EcoFusion']=net.parameters
        selected.append({'seed':seed,'temporal_weight':float(w),'selected_width':entry['width'],'selection_partition':'validation'})
        if fscore is not None:
            raw['Chronos']=fscore;costs['Chronos']=fmeta['seconds'];params['Chronos']=fmeta['parameters']
            fw=min(weights,key=lambda v:log_loss(y[val],v*fscore[val]+(1-v)*raw['EcoFusion'][val]))
            raw['Chronos+EcoFusion']=fw*fscore+(1-fw)*raw['EcoFusion'];costs['Chronos+EcoFusion']=costs['EcoFusion']+fmeta['seconds'];params['Chronos+EcoFusion']=fmeta['parameters']+net.parameters
            selected[-1]['chronos_weight']=float(fw)
        calibrated={};calibrators={}
        for name,pr in raw.items():
            calibrator=Calibrator().fit(pr[cal],y[cal]);valid=np.isfinite(pr)
            pp=np.full(len(pr),np.nan);pp[valid]=calibrator.predict(pr[valid]);calibrated[name]=pp;calibrators[name]=calibrator
            for splitname in ['test','spatial_test']:
                if splitname not in ix or len(ix[splitname])<1:continue
                ids=ix[splitname];met=metrics(y[ids],pp[ids])
                report.append(dict(model=name,seed=seed,split=splitname,seconds=costs[name],parameters=params[name],**met,
                    **conformal(y[cal],pp[cal],y[ids],pp[ids])))
                q=ds.rows.iloc[ids].copy();q['model']=name;q['seed']=seed;q['split']=splitname;q['probability']=pp[ids];q['raw_score']=pr[ids];predictions.append(q)
                for yr in sorted(q.origin_date.dt.year.unique()):
                    local=ids[ds.rows.iloc[ids].origin_date.dt.year.to_numpy()==yr]
                    stability.append(dict(model=name,seed=seed,split=splitname,year=int(yr),**metrics(y[local],pp[local])))
        joblib.dump(calibrators,out/f'calibrators_{seed}.joblib')
        # Pair every seed/model on identical test rows; block by calendar quarter.
        ids=ix['test'];blocks=ds.rows.iloc[ids].origin_date.dt.to_period('Q').astype(str).to_numpy()
        for name in ['EcoTemporalNet','EcoFusion']+(['Chronos','Chronos+EcoFusion'] if fscore is not None else []):
            ci=block_delta(y[ids],calibrated[name][ids],calibrated['HistGradientBoosting'][ids],blocks,bootstrap,seed)
            controls.append(dict(model=name,seed=seed,reference='HistGradientBoosting',**ci))
        # Feature ablation is predictive sensitivity, never a causal effect.
        rng=np.random.default_rng(seed)
        for fi,feature in enumerate(ds.features):
            changed=X[ids].copy();changed[:,:,fi]=0.;changed[:,:,fi+len(ds.features)]=0.
            pp=calibrators['EcoTemporalNet'].predict(net.predict_proba(changed))
            importance.append(dict(seed=seed,feature=feature,AP_drop=metrics(y[ids],calibrated['EcoTemporalNet'][ids])['AP']-metrics(y[ids],pp)['AP'],interpretation='masking sensitivity; correlated covariates and distribution shift limit interpretation'))
        for label,changed in [('missing20',X[ids].copy()),('noise01',X[ids].copy())]:
            if label=='missing20':
                m=rng.random(changed[:,:,:len(ds.features)].shape)<.2
                changed[:,:,:len(ds.features)][m]=0.;changed[:,:,len(ds.features):][m]=0.
            else: changed[:,:,:len(ds.features)]+=rng.normal(0,.1,changed[:,:,:len(ds.features)].shape)
            pp=calibrators['EcoTemporalNet'].predict(net.predict_proba(changed))
            stability.append(dict(model='EcoTemporalNet',seed=seed,split=label,year=0,**metrics(y[ids],pp)))
        if 'temperature' in ds.features:
            f=ds.features.index('temperature');cut=np.nanquantile(ds.X[train,-1,f],.9)
            hot=ids[ds.X[ids,-1,f]>=cut]
            for name in calibrated:
                if len(hot):stability.append(dict(model=name,seed=seed,split='observed_high_temperature',year=0,**metrics(y[hot],calibrated[name][hot])))
    pd.DataFrame(report).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(stability).to_csv(out/'stability.csv',index=False)
    pd.DataFrame(importance).to_csv(out/'explainability.csv',index=False)
    pd.DataFrame(controls).to_csv(out/'paired_block_bootstrap.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'training_selection.csv',index=False)
    result=pd.DataFrame(report)
    summary=result.groupby(['model','split'])[['AP','Brier','ECE','seconds','top10_recall']].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    ci=pd.DataFrame(controls)
    status={}
    for name,g in ci.groupby('model'):
        stable=bool((g.delta_ap_low>0).all())
        status[name]='positive_under_this_protocol' if stable else 'no_stable_superiority_demonstrated'
    (out/'conclusions.json').write_text(json.dumps({'task':ds.task,'vs_HistGradientBoosting':status,
        'criterion':'AP paired calendar-quarter bootstrap lower bound > 0 for every seed; Brier/ECE tradeoffs shown separately',
        'limits':ds.note,'calibration':'Separate pre-test chronological window; empirical conformal coverage not an IID guarantee.'},indent=2))
    print(summary.to_string(),flush=True)
    return result
