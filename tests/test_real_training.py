"""Synthetic test fixtures only; never used as scientific experiment results."""
import io,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from globalhab_demo.real_training.data import _sequences,split,field,hong_kong
from globalhab_demo.real_training.models import Preprocessor,TemporalEstimator
from globalhab_demo.real_training.evaluation import metrics,block_delta

ROOT=Path(__file__).resolve().parents[1]

def test_controller_reacts_to_feedback_and_rejects_test_labels():
    from globalhab_demo.real_training.agent import choose_capacity
    assert choose_capacity()[0]==16
    assert choose_capacity({'train_loss':.1,'val_loss':.3})[0]==8
    assert choose_capacity({'train_loss':.3,'val_loss':.31})[0]==24
    with pytest.raises(ValueError):choose_capacity({'train_loss':.1,'val_loss':.3,'test_AP':.9})

def fixture_frame():
    dates=pd.date_range('2010-01-01',periods=1000)
    return pd.concat([pd.DataFrame(dict(site=f's{i}',date=dates,event=np.arange(1000)%2,value=np.arange(1000)%9,
        latitude=22.,longitude=114.,source='SYNTHETIC_UNIT_TEST')) for i in range(15)],ignore_index=True)

def test_future_edits_do_not_enter_past_inputs():
    f=fixture_frame();cut=pd.Timestamp('2012-01-01')
    a=_sequences(f,['value'],12,7,3,'fixture','test only')
    f.loc[f.date>=cut,'value']=999999
    b=_sequences(f,['value'],12,7,3,'fixture','test only')
    np.testing.assert_array_equal(a.X[a.rows.origin_date<cut],b.X[b.rows.origin_date<cut])
    assert (a.rows.label_date>a.rows.origin_date).all()
    assert 'event' not in a.features

def test_label_purge_and_held_sites():
    ds=_sequences(fixture_frame(),['value'],12,7,3,'fixture','test only')
    ix,m=split(ds,['2011-01-01','2011-07-01','2012-01-01'])
    assert ds.rows.iloc[ix['train']].label_date.max()<pd.Timestamp(m['cutoffs'][0])
    assert ds.rows.iloc[ix['validation']].label_date.max()<pd.Timestamp(m['cutoffs'][1])
    assert ds.rows.iloc[ix['calibration']].label_date.max()<pd.Timestamp(m['cutoffs'][2])
    train_sites=set(ds.rows.iloc[ix['train']].site)
    assert not train_sites.intersection(ds.rows.iloc[ix['spatial_test']].site)
    parts=list(ix.values())
    for i,a in enumerate(parts):
        for b in parts[i+1:]:assert len(np.intersect1d(a,b))==0

def test_preprocessor_has_no_test_fit_and_handles_inf():
    train=np.array([[[1.,np.nan],[2.,np.inf]],[[3.,np.nan],[4.,np.nan]]])
    p=Preprocessor().fit(train);old=p.median.copy()
    out=p.transform(np.full_like(train,100000))
    np.testing.assert_array_equal(old,p.median)
    assert np.isfinite(out).all() and out[:,:,:2].max()<=12
    assert p.median[1]==0

def test_field_delayed_availability_is_rejected():
    text='station_id,date,available_at,latitude,longitude,observed_event,value,source\na,2020-01-01,2020-01-02,22,114,1,2,test\n'
    with pytest.raises(ValueError,match='available'):field(io.StringIO(text))

def test_unknown_not_silently_negative():
    text='station_id,date,available_at,latitude,longitude,observed_event,value,source\na,2020-01-01,2020-01-01,22,114,,2,test\n'
    with pytest.raises(ValueError,match='No eligible'):field(io.StringIO(text))

def test_tiny_network_seed_and_parameter_cap():
    pytest.importorskip('torch')
    rng=np.random.default_rng(1);raw=rng.normal(size=(64,12,2)).astype('float32');y=(raw[:,-1,0]>0).astype(int)
    prep=Preprocessor().fit(raw[:40]);x=prep.transform(raw)
    a=TemporalEstimator(2,8,17,epochs=3).fit(x[:40],y[:40],x[40:],y[40:])
    b=TemporalEstimator(2,8,17,epochs=3).fit(x[:40],y[:40],x[40:],y[40:])
    np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x));assert a.parameters<50000

def test_metric_semantics():
    m=metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
    assert m['AP']==1 and m['Brier']<.1 and m['prevalence']==.5
    c=block_delta(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]),np.array([.1,.9,.2,.8]),np.array(['a','a','b','b']),20)
    assert c['delta_ap_low']==c['delta_ap_high']==0

def test_public_snapshot_integrity():
    raw=ROOT/'data/training_real/raw';m=json.loads((raw/'snapshot_manifest.json').read_text())
    assert m['habsos']['complete_snapshot']
    for name,digest in m['sha256'].items():assert hashlib.sha256((raw/name).read_bytes()).hexdigest()==digest

def test_hk_snapshot_split_and_inference_reproduces_saved_predictions():
    pytest.importorskip('torch')
    from globalhab_demo.real_training.inference import Predictor
    ds=hong_kong(ROOT/'data/training_real/raw/hong_kong_red_tide.csv')
    folder=ROOT/'outputs/real_training/china_hk_7d';m=json.loads((folder/'split_manifest.json').read_text())
    ix,actual=split(ds,m['cutoffs']);assert actual['sha256']==m['sha256']
    ids=ix['test'][:10];p=Predictor(folder).predict(ds.X[ids])
    stored=pd.read_csv(folder/'predictions.csv.gz')
    stored=stored[(stored.seed==42)&(stored.model=='EcoFusion')].set_index('sample_id')
    np.testing.assert_allclose(p,stored.loc[ds.rows.iloc[ids].sample_id].probability,atol=1e-6)
