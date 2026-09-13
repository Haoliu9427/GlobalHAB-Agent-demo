"""Small missingness-aware multiscale TCN. No future inputs or test fitting."""
import random
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

class Preprocessor:
    def fit(self,X):
        z=X.reshape(-1,X.shape[-1]).copy()
        z[~np.isfinite(z)]=np.nan
        self.median=np.array([np.nanmedian(z[:,i]) if np.isfinite(z[:,i]).any() else 0 for i in range(z.shape[-1])])
        filled=np.where(np.isfinite(z),z,self.median)
        self.scale=StandardScaler().fit(filled)
        return self
    def transform(self,X):
        shape=X.shape; mask=np.isfinite(X).astype('float32')
        v=np.where(np.isfinite(X),X,self.median).reshape(-1,shape[-1])
        v=np.clip(self.scale.transform(v),-12,12).reshape(shape).astype('float32')
        return np.concatenate([v,mask],axis=-1)

def tabular(X):
    return np.concatenate([X[:,-1],X.mean(1),X.std(1),X[:,-1]-X[:,0]],axis=1)

def torch_model(features,width=16):
    import torch
    from torch import nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.input=nn.Conv1d(features*2,width,1)
            self.branches=nn.ModuleList([nn.Conv1d(width,width,3,dilation=d) for d in (1,2,4)])
            self.gate=nn.Linear(width,3)
            self.head=nn.Sequential(nn.Linear(width+features*2,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,1))
        def forward(self,x):
            h=torch.nn.functional.gelu(self.input(x.transpose(1,2)))
            branches=[torch.nn.functional.gelu(conv(torch.nn.functional.pad(h,(2*d,0))))[:,:,-1] for conv,d in zip(self.branches,(1,2,4))]
            weights=torch.softmax(self.gate(h[:,:,-1]),-1)
            fused=(torch.stack(branches,dim=-1)*weights[:,None,:]).sum(-1)
            return self.head(torch.cat([fused,x[:,-1]],-1)).squeeze(-1)
    return Net()

class TemporalEstimator:
    def __init__(self,features,width=16,seed=42,epochs=20,batch_size=256):
        self.features=features;self.width=width;self.seed=seed;self.epochs=epochs;self.batch_size=batch_size
    def fit(self,X,y,Xv,yv):
        import torch,copy
        torch.set_num_threads(2);random.seed(self.seed);np.random.seed(self.seed);torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.net=torch_model(self.features,self.width)
        self.parameters=sum(p.numel() for p in self.net.parameters())
        if self.parameters>50000: raise ValueError('Strict 50k parameter budget exceeded')
        opt=torch.optim.AdamW(self.net.parameters(),lr=.002,weight_decay=.001)
        xt=torch.as_tensor(X);yt=torch.as_tensor(y.astype('float32'))
        best=float('inf');state=None;wait=0;self.learning_curve=[]
        gen=torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            for ids in torch.randperm(len(yt),generator=gen).split(self.batch_size):
                opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(self.net(xt[ids]),yt[ids]);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0);opt.step()
            p=self.predict_proba(Xv);score=log_loss(yv,p,labels=[0,1])
            self.learning_curve.append({'epoch':epoch+1,'validation_log_loss':float(score)})
            if score<best-1e-5: best=score;state=copy.deepcopy(self.net.state_dict());self.best_epoch=epoch+1;wait=0
            else: wait+=1
            if wait>=4: break
        self.net.load_state_dict(state);return self
    def predict_proba(self,X):
        import torch
        self.net.eval();p=[]
        with torch.no_grad():
            for start in range(0,len(X),512): p.append(torch.sigmoid(self.net(torch.as_tensor(X[start:start+512]))).numpy())
        return np.concatenate(p) if p else np.array([])

def classical(name,seed):
    if name=='Logistic': return LogisticRegression(C=.2,max_iter=1000,random_state=seed)
    if name=='RandomForest': return RandomForestClassifier(n_estimators=150,min_samples_leaf=10,max_depth=12,n_jobs=2,random_state=seed)
    if name=='HistGradientBoosting': return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,learning_rate=.05,early_stopping=False,random_state=seed)
    raise ValueError(name)
