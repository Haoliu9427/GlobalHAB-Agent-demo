"""Real observations only. Targets, availability and geographic units are explicit."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

@dataclass
class Dataset:
    X: np.ndarray
    rows: pd.DataFrame
    features: list
    task: str
    note: str

def _sequences(frame, features, length, horizon_days, tolerance_days, task, note):
    if length<2 or horizon_days<1 or tolerance_days<0:raise ValueError('Invalid sequence or forecast horizon')
    if frame.duplicated(['site','date']).any():raise ValueError('Duplicate site/date in sequence source')
    X, rows = [], []
    for site,g in frame.groupby('site',sort=True):
        g=g.sort_values('date').reset_index(drop=True)
        times=g.date.values.astype('datetime64[D]')
        vals=g[features].to_numpy(dtype=np.float32)
        # Labels are read only into rows; no target-date fields enter X.
        for i in range(length-1,len(g)):
            j=np.searchsorted(times,times[i]+np.timedelta64(horizon_days,'D'))
            if j>=len(g) or (times[j]-times[i]).astype(int)>horizon_days+tolerance_days: continue
            if (times[i]-times[i-length+1]).astype(int)>730: continue
            r=g.iloc[i]; target=g.iloc[j]
            X.append(vals[i-length+1:i+1])
            rows.append(dict(sample_id=f'{task}:{site}:{str(times[i])}',site=site,
                origin_date=r.date,label_date=target.date,y=int(target.event),
                latitude=r.latitude,longitude=r.longitude,source=r.source,
                last_observed_event=int(r.event),target_value=float(target.value)))
    if not X: raise ValueError('No eligible future targets; need repeated observations and sufficient history.')
    rows=pd.DataFrame(rows)
    order=rows.sort_values(['origin_date','site']).index.to_numpy()
    return Dataset(np.asarray(X)[order],rows.iloc[order].reset_index(drop=True),features,task,note)

def habsos(path, horizon=7, length=12, threshold=100000.):
    d=pd.read_csv(path,low_memory=False)
    d['date']=pd.to_datetime(d.SAMPLE_DATE,unit='ms',errors='coerce').dt.floor('D')
    # Exclude malformed and non-comparable units, preserve missing environmental values.
    keep=(d.GENUS.str.lower().eq('karenia') & d.SPECIES.str.lower().eq('brevis')
          & d.CELLCOUNT_UNIT.str.lower().eq('cells/l') & d.CELLCOUNT.ge(0)
          & d.CELLCOUNT_QA.eq(1) & d.LATITUDE.between(-90,90) & d.LONGITUDE.between(-180,180))
    d=d[keep & d.date.notna()].copy()
    for col,qa in [('WATER_TEMP','WATER_TEMP_QA'),('SALINITY','SALINITY_QA'),('WIND_SPEED','WIND_SPEED_QA')]:
        d.loc[d[qa].ne(1),col]=np.nan
    d.loc[~d.WATER_TEMP.between(-2,40),'WATER_TEMP']=np.nan
    d.loc[~d.SALINITY.between(0,50),'SALINITY']=np.nan
    d.loc[~d.WIND_SPEED.between(0,100),'WIND_SPEED']=np.nan
    # 0.1-degree cells are analysis units, not fabricated fixed sampling stations.
    d['site']=((d.LATITUDE/.1).apply(np.floor).astype(int).astype(str)+':'+
               (d.LONGITUDE/.1).apply(np.floor).astype(int).astype(str))
    d=d.groupby(['site','date'],as_index=False).agg(value=('CELLCOUNT','max'),
        temperature=('WATER_TEMP','median'),salinity=('SALINITY','median'),
        wind_speed=('WIND_SPEED','median'),latitude=('LATITUDE','mean'),longitude=('LONGITUDE','mean'))
    d['event']=(d.value>=threshold).astype(int)
    d['log_cells']=np.log1p(d.value)
    d['gap_days']=d.groupby('site').date.diff().dt.days.clip(upper=730)
    d['season_sin']=np.sin(2*np.pi*d.date.dt.dayofyear/365.25)
    d['season_cos']=np.cos(2*np.pi*d.date.dt.dayofyear/365.25)
    d['source']='NOAA_HABSOS'
    features=['log_cells','temperature','salinity','wind_speed','gap_days','season_sin','season_cos','latitude','longitude']
    return _sequences(d,features,length,horizon,3,f'habsos_{horizon}d',
        f'First observed same-cell sample {horizon}–{horizon+3} days later; Karenia brevis >= {threshold:g} cells/L. '
        '0.1 degree cells; sampled-target prediction, not continuous daily bloom forecast. Sample availability assumed at collection; no publication-latency archive.')

def hong_kong(path,length=26):
    d=pd.read_csv(path)
    d['date']=pd.to_datetime(d['Date of Report'],errors='coerce')
    d=d.dropna(subset=['date']).drop_duplicates('Red Tide Sighting No.')
    # Restrict to complete historical years. No future/current incomplete week.
    # Use archive dates, not wall-clock time: a later rerun must not fabricate
    # additional future zero-report weeks from the same frozen source file.
    end=pd.Timestamp(f'{d.date.max().year-1}-12-31')
    start=pd.Timestamp('1976-01-01')
    dates=pd.date_range(start,end,freq='W-SUN')
    counts=d.set_index('date').resample('W-SUN').size().reindex(dates,fill_value=0)
    f=pd.DataFrame({'date':dates,'value':counts.values})
    f['event']=(f.value>0).astype(int);f['log_reports']=np.log1p(f.value)
    f['season_sin']=np.sin(2*np.pi*f.date.dt.dayofyear/365.25)
    f['season_cos']=np.cos(2*np.pi*f.date.dt.dayofyear/365.25)
    f['site']='CN_HongKong';f['source']='AFCD_HongKong';f['latitude']=22.35;f['longitude']=114.2
    return _sequences(f,['log_reports','season_sin','season_cos'],length,7,0,'china_hk_7d',
        'Next calendar week contains >=1 AFCD red-tide report. A zero means no report in this archive, '
        'not monitored biological absence. Retrospective as-of report date; historical publication revisions unknown. '
        'China Hong Kong waters only; not a validation for all Chinese seas.')

def field(path,horizon=7,length=12):
    d=pd.read_csv(path)
    required={'station_id','date','available_at','latitude','longitude','observed_event','value','source'}
    if required-set(d): raise ValueError(f'Missing fields: {sorted(required-set(d))}')
    if not d.observed_event.dropna().isin([0,1]).all(): raise ValueError('observed_event must be verified 0/1; unknown must be empty.')
    d['date']=pd.to_datetime(d.date);d['available_at']=pd.to_datetime(d.available_at)
    if d.date.isna().any() or d['station_id'].isna().any() or d.source.isna().any():
        raise ValueError('Date, station and source must be present')
    if d.available_at.isna().any() or (d.available_at>d.date).any():
        raise ValueError('This adapter requires observations available at origin date. Align delayed records to actual issue times before use.')
    if not d.latitude.between(-90,90).all() or not d.longitude.between(-180,180).all(): raise ValueError('Invalid coordinates')
    d=d.dropna(subset=['observed_event']).rename(columns={'station_id':'site','observed_event':'event'})
    if d.duplicated(['site','date']).any(): raise ValueError('Duplicate station-date; aggregate with documented policy first.')
    feats=['value']+[c for c in ['temperature','salinity','dissolved_oxygen','nitrate','phosphate','silicate','u_current','v_current'] if c in d]
    d['season_sin']=np.sin(2*np.pi*d.date.dt.dayofyear/365.25); d['season_cos']=np.cos(2*np.pi*d.date.dt.dayofyear/365.25)
    return _sequences(d,feats+['season_sin','season_cos'],length,horizon,3,'china_field',
        'User supplied monitored presence/absence; future and site holdouts. Geographic origin is user-declared, not independently verified. Threshold and species must be documented in source.')

def split(ds,dates=None,holdout=True):
    r=ds.rows
    dates=dates or tuple(str(r.origin_date.quantile(q).date()) for q in [.6,.72,.82])
    a,b,c=map(pd.Timestamp,dates)
    if not a<b<c: raise ValueError('Need three ordered split boundaries')
    sites=sorted(r.site.unique())
    if holdout and len(sites)>=10:
        held={s for s in sites if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5==0}
    else: held=set()
    spatial=r.site.isin(held)
    masks={'train':(r.label_date<a)&~spatial,
        'validation':(r.origin_date>=a)&(r.label_date<b)&~spatial,
        'calibration':(r.origin_date>=b)&(r.label_date<c)&~spatial,
        'test':(r.origin_date>=c)&~spatial}
    if held: masks['spatial_test']=(r.origin_date>=c)&spatial
    idx={k:np.flatnonzero(v) for k,v in masks.items()}
    for k in ['train','validation','calibration','test']:
        if len(idx[k])<20 or r.iloc[idx[k]].y.nunique()<2: raise ValueError(f'{k}: insufficient rows or only one class; do not report AP as validated.')
    manifest={'cutoffs':list(map(str,dates)),'held_out_sites':sorted(held),'counts':{k:len(v) for k,v in idx.items()},
        'purge_rule':'label_date strictly before next partition start','task':ds.task,'note':ds.note,
        'sha256':{k:hashlib.sha256('\n'.join(r.iloc[v].sample_id).encode()).hexdigest() for k,v in idx.items()}}
    return idx,manifest
