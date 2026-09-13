"""Synthetic integration fixtures; these are NOT scientific validation evidence."""
import io,json,zipfile
import numpy as np
import pandas as pd
import pytest
from globalhab_demo.real_training.own_observations import prepare,execute

def fixture():
    rng=np.random.default_rng(21);dates=pd.date_range('2016-01-01',periods=1100)
    v=rng.normal(size=len(dates));y=(v>0).astype(int)
    return pd.DataFrame(dict(station_id='001',date=dates,available_at=dates,latitude=22.,longitude=114.,observed_event=y,value=v,source='SYNTHETIC_SOFTWARE_TEST_ONLY')).to_csv(index=False).encode()

def test_own_partition_and_no_future_input():
    b=fixture();ds,ix,m=prepare(b);assert m['chronos_eligible']
    assert ds.rows.site.iloc[0]=='station:001'
    f=pd.read_csv(io.BytesIO(b));cut='2018-01-01';f.loc[f.date>=cut,'observed_event']=1-f.loc[f.date>=cut,'observed_event']
    ds2,_,_=prepare(f.to_csv(index=False).encode());old=ds.rows.origin_date<pd.Timestamp(cut)
    np.testing.assert_array_equal(ds.X[old],ds2.X[old])
    for a,c in [('train','validation'),('validation','calibration'),('calibration','test')]:
        assert ds.rows.iloc[ix[a]].label_date.max()<ds.rows.iloc[ix[c]].origin_date.min()

def test_irregular_chronos_not_silently_resampled():
    f=pd.read_csv(io.BytesIO(fixture())).drop(index=30)
    assert not prepare(f.to_csv(index=False).encode())[2]['chronos_eligible']

def test_classical_full_run_and_download():
    df,m,b=execute(fixture(),['Logistic','RandomForest','HistGradientBoosting'],description='Synthetic unit test label, no research claim')
    assert len(df)==9
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        assert z.testzip() is None
        p=pd.read_csv(z.open('predictions.csv.gz'),compression='gzip')
        groups=p.groupby(['model','seed']).sample_id.apply(set)
        assert all(x==groups.iloc[0] for x in groups)
        assert len(groups.iloc[0])==m['counts']['test']
        assert json.loads(z.read('protocol.json'))['input_sha256']==m['input_sha256']
        assert p.probability.between(0,1).all()

def test_tcn_full_run():
    pytest.importorskip('torch')
    df,m,b=execute(fixture(),['EcoTemporalNet'],epochs=1,description='Synthetic software integration test')
    assert len(df)==3 and df.parameters.max()<50000

def test_foundation_common_samples_with_stub(monkeypatch):
    import globalhab_demo.real_training.own_observations as o
    monkeypatch.setattr(o,'dependencies',lambda name:[])
    def stub(name,ds,ids,horizon,description,notify):
        return np.full(len(ids),.3),{'parameters':1,'test_fixture':True}
    monkeypatch.setattr(o,'foundation_score',stub)
    df,_,b=execute(fixture(),['Logistic','Chronos-Bolt-small','Qwen2.5-0.5B-Instruct'],description='Synthetic adapter contract test only')
    assert len(df)==5
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        p=pd.read_csv(z.open('predictions.csv.gz'),compression='gzip')
        assert p.groupby('model').sample_id.nunique().nunique()==1

def test_failed_model_cannot_be_success(monkeypatch):
    import globalhab_demo.real_training.own_observations as o
    monkeypatch.setattr(o,'dependencies',lambda name:[])
    monkeypatch.setattr(o,'foundation_score',lambda *args:(np.array([np.nan]),{}))
    with pytest.raises((ValueError,KeyError)):execute(fixture(),['Qwen2.5-0.5B-Instruct'],description='Failure contract test')
