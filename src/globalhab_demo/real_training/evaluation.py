"""Calibrated risk metrics and paired block uncertainty, without test selection."""
import numpy as np
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.linear_model import LogisticRegression

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if len(y)==0: return {}
    ece=0
    for i in range(10):
        m=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        if m.any(): ece+=m.mean()*abs(p[m].mean()-y[m].mean())
    k=max(1,int(np.ceil(.1*len(y))));top=np.argsort(-p,kind='stable')[:k]
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
        AP=float(average_precision_score(y,p)) if y.sum()>0 else float('nan'),
        Brier=float(brier_score_loss(y,p)),ECE=float(ece),
        ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
        top10_recall=float(y[top].sum()/y.sum()) if y.sum() else float('nan'),
        top10_precision=float(y[top].mean()))

def logit(p):
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p)).reshape(-1,1)
class Calibrator:
    def fit(self,p,y):
        self.model=LogisticRegression(C=1.,max_iter=1000).fit(logit(p),y);return self
    def predict(self,p): return self.model.predict_proba(logit(p))[:,1]

def block_delta(y,p,baseline,blocks,repeats=200,seed=42):
    rng=np.random.default_rng(seed);u=np.unique(blocks);indices={b:np.flatnonzero(blocks==b) for b in u};v=[]
    for _ in range(repeats):
        ix=np.concatenate([indices[b] for b in rng.choice(u,size=len(u),replace=True)])
        if len(np.unique(y[ix]))<2: continue
        v.append(average_precision_score(y[ix],p[ix])-average_precision_score(y[ix],baseline[ix]))
    if not v:return dict(delta_ap_low=None,delta_ap_high=None,valid_replicates=0)
    return dict(delta_ap_low=float(np.quantile(v,.025)),delta_ap_high=float(np.quantile(v,.975)),valid_replicates=len(v))

def conformal(cal_y,cal_p,y,p,alpha=.1):
    # Empirical split-conformal diagnostic; time dependence invalidates IID coverage guarantees.
    s=np.where(cal_y==1,1-cal_p,cal_p)
    q=np.quantile(s,min(1,np.ceil((len(s)+1)*(1-alpha))/len(s)),method='higher')
    include0=p<=q;include1=(1-p)<=q
    return dict(empirical_coverage=float(np.where(y==1,include1,include0).mean()),
        ambiguous_fraction=float((include0 & include1).mean()),empty_fraction=float((~include0 & ~include1).mean()),
        nominal_coverage=1-alpha,threshold=float(q))
