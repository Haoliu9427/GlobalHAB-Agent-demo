"""Inference from this project's trusted saved artifacts, without retraining."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .models import TemporalEstimator,torch_model,tabular

class Predictor:
    def __init__(self,folder,seed=42):
        self.folder=Path(folder);self.seed=seed
        self.manifest=json.loads((self.folder/'split_manifest.json').read_text())
        self.prep=joblib.load(self.folder/'preprocessor.joblib')
        self.calibrators=joblib.load(self.folder/f'calibrators_{seed}.joblib')
    def raw(self,X,model='EcoFusion'):
        if X.ndim!=3 or X.shape[1:]!=(self.manifest['sequence_length'],len(self.manifest['features'])):
            raise ValueError('History length and feature order must match split_manifest.json')
        z=self.prep.transform(X)
        if model in ['Logistic','RandomForest','HistGradientBoosting']:
            return joblib.load(self.folder/f'{model}_{self.seed}.joblib').predict_proba(tabular(z))[:,1]
        import torch
        cfg=json.loads((self.folder/f'EcoTemporalNet_{self.seed}.json').read_text())
        net=TemporalEstimator(cfg['features'],cfg['width'],self.seed)
        net.net=torch_model(cfg['features'],cfg['width'])
        net.net.load_state_dict(torch.load(self.folder/f'EcoTemporalNet_{self.seed}.pt',map_location='cpu',weights_only=True))
        torch.set_num_threads(2)
        p=net.predict_proba(z)
        if model=='EcoTemporalNet':return p
        if model!='EcoFusion':raise ValueError('Unsupported persisted model')
        selection=pd.read_csv(self.folder/'training_selection.csv').set_index('seed')
        w=float(selection.loc[self.seed,'temporal_weight'])
        return w*p+(1-w)*self.raw(X,'HistGradientBoosting')
    def predict(self,X,model='EcoFusion'):
        return self.calibrators[model].predict(self.raw(X,model))
