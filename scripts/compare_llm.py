"""Frozen language-model baseline and validation-selected hybrid on exact HK split."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from globalhab_demo.real_training.data import hong_kong,split
from globalhab_demo.real_training.foundation import hf_llm_scores,ollama_scores
from globalhab_demo.real_training.inference import Predictor
from globalhab_demo.real_training.evaluation import Calibrator,metrics,block_delta,conformal

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--backend',choices=['huggingface','ollama'],default='huggingface');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.output.exists() and not a.resume:p.error('Choose a new output directory, or --resume for an interrupted run.')
    if (a.output/'metrics.csv').exists():p.error('Completed evidence cannot be overwritten.')
    if a.resume and a.backend!='huggingface':p.error('Checkpoint resume is implemented for the Hugging Face backend.')
    ds=hong_kong(ROOT/'data/training_real/raw/hong_kong_red_tide.csv')
    m=json.loads((a.experiment/'split_manifest.json').read_text());ix,check=split(ds,m['cutoffs'])
    if check['sha256']!=m['sha256'] or hashlib.sha256(ds.X.tobytes()).hexdigest()!=m['raw_data_hash']:
        raise ValueError('Dataset or partition drift: comparison rejected')
    a.output.mkdir(parents=True,exist_ok=True)
    status={'status':'running','model':a.model,'parent_experiment':a.experiment.name,'same_split_sha256':check['sha256']}
    statusfile=a.output/'status.json';statusfile.write_text(json.dumps(status,indent=2))
    ids=np.unique(np.concatenate([ix[k] for k in ['validation','calibration','test']]))
    try:
        if a.backend=='huggingface':scores,meta=hf_llm_scores(ds.X[ids],ds.features,a.model,checkpoint=a.output/'score_checkpoint.json')
        else:
            scores,audit=ollama_scores(ds.X[ids],ds.features,a.model)
            pd.DataFrame(audit).to_csv(a.output/'local_llm_audit.csv',index=False)
            meta={'model_id':a.model,'seconds':sum(r['seconds'] for r in audit),'parameters':None}
        raw=np.full(len(ds.X),np.nan);raw[ids]=scores;y=ds.rows.y.to_numpy()
        rows=ds.rows.iloc[ids].copy();rows['raw_score']=scores;rows.to_csv(a.output/'raw_predictions.csv',index=False)
        result=[];pred=[];ci=[];selected=[]
        val,cal,test=ix['validation'],ix['calibration'],ix['test']
        baselines=pd.read_csv(a.experiment/'predictions.csv.gz')
        blocks=ds.rows.iloc[test].origin_date.dt.to_period('Q').astype(str).to_numpy()
        for seed in m['seeds']:
            eco=Predictor(a.experiment,seed).raw(ds.X)
            w=min(np.linspace(0,1,5),key=lambda z:log_loss(y[val],z*raw[val]+(1-z)*eco[val]))
            selected.append({'seed':seed,'llm_weight':float(w),'selection_partition':'validation'})
            for name,pr in [('LLM',raw),('LLM+EcoFusion',w*raw+(1-w)*eco)]:
                c=Calibrator().fit(pr[cal],y[cal]);pp=c.predict(pr[test]);cp=c.predict(pr[cal])
                result.append(dict(model=name,seed=seed,split='test',seconds=meta['seconds'],parameters=meta['parameters'],
                    **metrics(y[test],pp),**conformal(y[cal],cp,y[test],pp)))
                q=ds.rows.iloc[test].copy();q['model']=name;q['seed']=seed;q['split']='test';q['probability']=pp;pred.append(q)
                for ref in ['Logistic','RandomForest','HistGradientBoosting','EcoTemporalNet']:
                    b=baselines[(baselines.seed==seed)&(baselines.model==ref)&(baselines.split=='test')].set_index('sample_id').loc[q.sample_id]
                    ci.append(dict(model=name,seed=seed,reference=ref,**block_delta(y[test],pp,b.probability.to_numpy(),blocks,200,seed)))
        pd.DataFrame(result).to_csv(a.output/'metrics.csv',index=False)
        pd.concat(pred).to_csv(a.output/'predictions.csv.gz',index=False)
        pd.DataFrame(ci).to_csv(a.output/'paired_block_bootstrap.csv',index=False)
        pd.DataFrame(selected).to_csv(a.output/'training_selection.csv',index=False)
        status.update(status='completed',**meta,seed_note='LLM is frozen deterministic; hybrid varies with 3 independently trained Eco models. Not 3 independently trained LLMs.',
            prompt_selection='One fixed prompt, not test tuned; model choice fixed before comparison.')
        print(pd.DataFrame(result).groupby('model')[['AP','Brier','ECE']].mean(),flush=True)
    except Exception as exc:
        status.update(status='failed',reason=str(exc));raise
    finally:statusfile.write_text(json.dumps(status,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
