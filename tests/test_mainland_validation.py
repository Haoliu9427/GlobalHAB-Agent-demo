from pathlib import Path
import json
import numpy as np
import pandas as pd
from globalhab_demo.real_training.mainland import sea,read,FILES,ece
ROOT=Path(__file__).resolve().parents[1]
def test_geographic_buffers():
    assert sea(119,39)=='渤海'
    assert sea(123,36)=='黄海'
    assert sea(122,28)=='东海'
    assert sea(110,20)=='南海'
    assert sea(121,38)=='边界过渡区'
    assert sea(float('nan'),20)=='坐标缺失'
def test_raw_data_integrity_and_unknown_values():
    raw=ROOT/'data/china_mainland/raw'
    for name in FILES:assert len(read(raw,name))>0
    x=read(raw,'water_toxin')
    assert pd.to_numeric(x['STX（µg/L）'],errors='coerce').isna().any()
def test_chronological_results():
    for method in ['ITS1','18S_V4']:
        out=ROOT/'outputs/china_mainland'/method
        m=json.loads((out/'manifest.json').read_text())
        assert m['train_years']==[2019,2020] and m['test_years']==[2021]
        p=pd.read_csv(out/'predictions.csv.gz');assert set(pd.to_datetime(p.date).dt.year)=={2021}
        groups=p.groupby(['model','seed']).record_id.apply(lambda x:set(x))
        assert all(ids==groups.iloc[0] for ids in groups)
        assert p.probability.between(0,1).all()
        assert not (p.sea=='渤海').any()
def test_ece_edges():
    assert np.isclose(ece(np.array([0,1]),np.array([0.,1.])),0)
