"""Download public archives once, preserving source bytes and frozen membership."""
import concurrent.futures as cf
import gzip, hashlib, json, time, urllib.request, urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'training_real'/'raw'; OUT.mkdir(parents=True,exist_ok=True)
BASE='https://gis.ncdc.noaa.gov/arcgis/rest/services/ms/HABSOS_CellCounts/MapServer/0/query'
def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(url,timeout=90) as r: return r.read()
        except Exception:
            if k==3: raise
            time.sleep(1+k)
def query(params):
    j=json.loads(get(BASE+'?'+urllib.parse.urlencode(params)))
    if 'error' in j: raise RuntimeError(j['error'])
    return j
def main():
    hk='https://redtide.afcd.gov.hk/data/RTMS_ob_RTLE.csv'
    dst=OUT/'hong_kong_red_tide.csv'
    if not dst.exists(): dst.write_bytes(get(hk))
    idfile=OUT/'habsos_object_ids.json'
    if not idfile.exists(): idfile.write_text(json.dumps(query(dict(where='1=1',returnIdsOnly='true',f='json'))))
    ids=sorted(json.loads(idfile.read_text())['objectIds']); print('Frozen HABSOS records:',len(ids),flush=True)
    pages=OUT/'habsos_pages'; pages.mkdir(exist_ok=True)
    def chunk(pair):
        i,batch=pair; path=pages/f'{i:05}.json.gz'
        if path.exists(): return path
        j=query(dict(where=f'OBJECTID >= {min(batch)} AND OBJECTID <= {max(batch)}',outFields='*',returnGeometry='false',resultRecordCount=2000,f='json'))
        received={f['attributes']['OBJECTID'] for f in j.get('features',[])}
        if received!=set(batch): raise RuntimeError(f'Incomplete page {i}: {len(received)}/{len(batch)}')
        path.write_bytes(gzip.compress(json.dumps(j).encode(),mtime=0)); return path
    groups=list(enumerate([ids[i:i+1000] for i in range(0,len(ids),1000)]))
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,path in enumerate(pool.map(chunk,groups)):
            if i%20==0: print('Downloaded',i+1,'/',len(groups),flush=True)
    rows=[]
    for i,b in groups:
        j=json.loads(gzip.decompress((pages/f'{i:05}.json.gz').read_bytes()))
        rows.extend(f['attributes'] for f in j['features'])
    import pandas as pd
    df=pd.DataFrame(rows).sort_values('OBJECTID')
    df.to_csv(OUT/'habsos.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    manifest={'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),'habsos':{'source':BASE,'records':len(df),'frozen_ids':len(ids),'complete_snapshot':len(df)==len(ids),'date_min':str(pd.to_datetime(df.SAMPLE_DATE,unit='ms').min()),'date_max':str(pd.to_datetime(df.SAMPLE_DATE,unit='ms').max()),'license':'US NOAA public data; original contributors retained'},'hong_kong':{'source':hk,'records':len(pd.read_csv(dst)),'license':'Hong Kong Government open data terms; attribution required'},'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'habsos.csv.gz',dst,idfile]}}
    (OUT/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()
