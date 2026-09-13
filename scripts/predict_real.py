"""Predict supplied histories, never silently substitute demonstration values."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from globalhab_demo.real_training.inference import Predictor

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',choices=['EcoFusion','EcoTemporalNet','Logistic','RandomForest','HistGradientBoosting'],default='EcoFusion')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model=Predictor(a.experiment,a.seed);payload=json.loads(a.history.read_text())
    if payload['features']!=model.manifest['features']:raise ValueError('Feature names/order do not match trained model')
    X=np.asarray(payload['histories'],dtype=np.float32)
    if not np.isfinite(X).any():raise ValueError('All inputs are missing')
    probabilities=model.predict(X,a.model)
    result={'task':model.manifest['task'],'model':a.model,'seed':a.seed,'probabilities':probabilities.tolist(),
        'missing_fraction':float((~np.isfinite(X)).mean()),'meaning':model.manifest['note'],
        'operational_status':'Research inference; deployment requires local prospective validation and monitoring.'}
    policy=a.experiment/'warning_policy.csv'
    if policy.exists():
        import pandas as pd
        policies=pd.read_csv(policy);row=policies[(policies.model==a.model)&(policies.seed==a.seed)]
        if len(row)==1:
            threshold=float(row.iloc[0].threshold)
            result['research_alarm_threshold']=threshold
            result['exceeds_research_threshold']=(probabilities>threshold).tolist()
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(a.output)
if __name__=='__main__':main()
