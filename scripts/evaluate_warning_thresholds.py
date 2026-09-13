"""Freeze a calibration-window alarm threshold, assess subsequent real labels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from globalhab_demo.real_training.data import hong_kong,habsos,split
from globalhab_demo.real_training.inference import Predictor

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);p.add_argument('--calibration-alert-budget',type=float,default=.1);a=p.parse_args()
    if not 0<a.calibration_alert_budget<1:p.error('Budget must be between 0 and 1')
    m=json.loads((a.experiment/'split_manifest.json').read_text());raw=ROOT/'data/training_real/raw'
    if m['task']=='china_hk_7d':ds=hong_kong(raw/'hong_kong_red_tide.csv')
    elif m['task'].startswith('habsos_'):ds=habsos(raw/'habsos.csv.gz',int(m['task'].split('_')[1][:-1]))
    else:p.error('For field experiments use the original verified input; this audit covers bundled archives.')
    ix,check=split(ds,m['cutoffs'])
    if check['sha256']!=m['sha256']:raise ValueError('Partition changed')
    d=pd.read_csv(a.experiment/'predictions.csv.gz');policies=[];results=[]
    for seed in m['seeds']:
        model=Predictor(a.experiment,seed)
        for name in ['Logistic','RandomForest','HistGradientBoosting','EcoTemporalNet','EcoFusion']:
            cp=model.predict(ds.X[ix['calibration']],name)
            threshold=float(np.quantile(cp,1-a.calibration_alert_budget,method='higher'))
            policies.append({'model':name,'seed':seed,'threshold':threshold,'operator':'>','calibration_alert_fraction':float((cp>threshold).mean()),'selected_on':'calibration'})
            for part,g in d[(d.seed==seed)&(d.model==name)].groupby('split'):
                alert=g.probability.to_numpy()>threshold;y=g.y.to_numpy().astype(bool)
                tp=int((alert&y).sum());fp=int((alert&~y).sum());fn=int((~alert&y).sum());tn=int((~alert&~y).sum())
                results.append({'model':name,'seed':seed,'split':part,'threshold':threshold,'n':len(g),'alerts':int(alert.sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,
                    'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':tp/(tp+fn) if tp+fn else np.nan,
                    'false_positive_rate':fp/(fp+tn) if fp+tn else np.nan,'alert_fraction':float(alert.mean())})
    pd.DataFrame(policies).to_csv(a.experiment/'warning_policy.csv',index=False)
    pd.DataFrame(results).to_csv(a.experiment/'warning_threshold_results.csv',index=False)
    print('Saved fixed pre-test threshold assessment:',a.experiment)
if __name__=='__main__':main()
