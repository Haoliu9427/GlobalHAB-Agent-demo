"""Adapters for the shared blocked scientific task; no unsupported-model fallback."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .real_training.model_registry import MODELS, TABULAR, missing, estimator

MODEL_NAMES={"logistic":"Logistic","random_forest":"Random Forest"}
MODEL_NAMES.update({"tab_"+str(i):n for i,n in enumerate(TABULAR) if n not in MODEL_NAMES.values()})
MODEL_NAMES["sts_interaction"]="STS-Interaction GLM"

def inventory():
    ids={v:k for k,v in MODEL_NAMES.items()}
    rows=[]
    for name in MODELS:
        model_id=ids.get(name)
        deps=missing(name) if model_id else []
        if model_id and not deps: reason="可用于当前阻断验证任务"
        elif deps: reason="缺少依赖："+", ".join(deps)
        elif name in ("Seasonal Climatology","Event Persistence"): reason="参照方法；季节参照已单独计算"
        elif name.startswith("STS-"): reason="需要完整序列与门控训练适配；本轮未接入"
        else: reason="保留数据分析工作区入口；本轮尚无同任务执行适配"
        rows.append({"model_id":model_id,"name":name,"available":bool(model_id and not deps),"reason":reason})
    return rows

FEATURES=["candidate_signal","nitrate_mmol_m3","phosphate_mmol_m3","silicate_mmol_m3","season_sin","season_cos"]

def fit_predict(name,train,target,seed):
    if name=="sts_interaction":
        # Reuse the existing STS interaction feature construction. The upstream
        # signal here is the selected route/lag, not an inferred causal network.
        from .real_training.user_engine import sts_arrays
        columns=["local_raw_signal","upstream_raw_signal","circulation_residence_proxy","nitrate","phosphate","silicate","season_sin","season_cos"]
        def raw(frame):
            signal=frame.candidate_signal.to_numpy()/(.55+.90*frame.circulation_residence_proxy.to_numpy())
            return np.column_stack([signal,signal,frame.circulation_residence_proxy,frame.nitrate_mmol_m3,frame.phosphate_mmol_m3,frame.silicate_mmol_m3,frame.season_sin,frame.season_cos])
        scale=make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
        x=scale.fit_transform(raw(train));z=scale.transform(raw(target))
        x=sts_arrays(x[:,None,:],columns)[3];z=sts_arrays(z[:,None,:],columns)[3]
        model=estimator("Logistic",seed,len(train)).fit(x,train.hab_event)
        return model.predict_proba(z)[:,1]
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),estimator(MODEL_NAMES[name],seed,len(train)))
    model.fit(train[FEATURES],train.hab_event)
    return model.predict_proba(target[FEATURES])[:,1]
