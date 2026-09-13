"""Post-fit paired comparisons only. Never change or choose fitted models."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from globalhab_demo.real_training.evaluation import block_delta

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);a=p.parse_args()
    d=pd.read_csv(a.experiment/'predictions.csv.gz');result=[]
    for (seed,splitname),g in d.groupby(['seed','split']):
        for name in ['EcoTemporalNet','EcoFusion','Chronos','Chronos+EcoFusion']:
            v=g[g.model==name].set_index('sample_id').sort_index()
            if v.empty:continue
            for ref in ['Logistic','RandomForest','HistGradientBoosting']:
                b=g[g.model==ref].set_index('sample_id').sort_index()
                if not v.index.equals(b.index):raise ValueError('Unpaired sample IDs')
                ci=block_delta(v.y.to_numpy(),v.probability.to_numpy(),b.probability.to_numpy(),pd.to_datetime(v.origin_date).dt.to_period('Q').astype(str).to_numpy(),500,int(seed))
                result.append(dict(model=name,reference=ref,seed=seed,split=splitname,**ci))
    pd.DataFrame(result).to_csv(a.experiment/'comparative_audit.csv',index=False)
if __name__=='__main__':main()
