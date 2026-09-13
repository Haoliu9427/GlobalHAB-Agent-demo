"""Synthetic software tests only, never published scientific scores."""
import io,runpy,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from globalhab_demo.real_training.user_engine import run,build
from globalhab_demo.real_training.model_registry import MODELS,TABULAR,BASELINES,FOUNDATIONS,expand
ROOT=Path(__file__).resolve().parents[1]
fixture=runpy.run_path(str(ROOT/'tests/test_own_observations.py'))['fixture']

def test_catalogue_covers_original():
    from globalhab_demo.broad_benchmark import benchmark_catalogue
    assert set(benchmark_catalogue()['模型'])<=set(MODELS)
    assert len(FOUNDATIONS)==8

def test_all_classical_future_common_partition():
    df,f,text,m,b=run(fixture(),TABULAR+BASELINES,description='Synthetic software fixture only',future=True,epochs=1)
    assert len(df)==len(TABULAR+BASELINES)*3
    assert f.y.isna().all() and f.target_value.isna().all()
    assert (f.label_date>f.origin_date).all()
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        p=pd.read_csv(z.open('predictions.csv.gz'),compression='gzip')
        groups=p.groupby(['model','seed']).sample_id.apply(set)
        assert all(v==groups.iloc[0] for v in groups)

def test_future_does_not_change_test_or_training():
    a,ia,ma=build(fixture(),7,False);b,ib,mb=build(fixture(),7,True)
    assert ma['sha256']==mb['sha256']
    np.testing.assert_array_equal(a.X,b.X[:len(a.X)])
    for k in ia:np.testing.assert_array_equal(ia[k],ib[k])

def test_deep_and_science_and_fusion():
    d=pd.read_csv(io.BytesIO(fixture()));d['local_raw_signal']=d.value;d['upstream_raw_signal']=d.value.shift(1).fillna(0)
    d['circulation_residence_proxy']=.3;d['nitrate']=1.;d['phosphate']=.1;d['silicate']=2.;d['upstream_available_at']=d.date
    names=['Lightweight TCN','STS-Gated TCN','STS-Interaction GLM','EcoFusion']
    table,forecast,_,m,_=run(d.to_csv(index=False).encode(),names,description='Synthetic test only',epochs=1,future=True)
    assert set(expand(['Seasonal Climatology']+names))==set(table.model)
    assert forecast.probability.between(0,1).all()

def test_data_changes_are_recomputed():
    a=fixture();d=pd.read_csv(io.BytesIO(a));d.observed_event=1-d.observed_event;b=d.to_csv(index=False).encode()
    _,fa,_,ma,_=run(a,['Logistic'],description='Synthetic fixture',epochs=1,future=True)
    _,fb,_,mb,_=run(b,['Logistic'],description='Synthetic fixture',epochs=1,future=True)
    assert ma['input_sha256']!=mb['input_sha256']
    assert not np.allclose(fa[fa.model=='Logistic'].probability,fb[fb.model=='Logistic'].probability)

def test_foundation_fusion_same_rows_with_contract_stub(monkeypatch):
    import globalhab_demo.real_training.user_engine as e
    monkeypatch.setattr(e,'missing',lambda n:[])
    def stub(name,ds,ids,horizon,description,notify):
        return np.linspace(.2,.8,len(ids)),{'parameters':1,'purpose':'synthetic contract stub'}
    monkeypatch.setattr(e,'foundation_score',stub)
    df,f,_,m,b=e.run(fixture(),['SmolLM2-360M-Instruct + EcoFusion'],epochs=1,description='Synthetic contract only',future=True)
    assert set(df.model)==set(expand(['Seasonal Climatology','SmolLM2-360M-Instruct + EcoFusion']))
    assert f.y.isna().all()
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        p=pd.read_csv(z.open('predictions.csv.gz'),compression='gzip')
        assert p.groupby('model').sample_id.nunique().nunique()==1
