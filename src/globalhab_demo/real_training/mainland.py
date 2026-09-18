"""China coastal survey audit; molecular detection is not a HAB forecast label."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE='https://www.scidb.cn/en/detail?dataSetId=a52529b9f4d64617a3511759ec28d643'
FILES={'microalgae':('869d74ad1f5103d3657d1aaceac949f5','ba3cc617576be0c1e7cc96aafe2ecde2'),
'molecular':('5a0a190664d895672181ebdc1537978a','9a6f224cbecae2668a2e2b35d73b50de'),
'animal_toxin':('feb008bdb5c09d59b788d455cedc9ff8','a81ab0c8a4c43d11599a76d5ceebfede'),
'water_toxin':('6427122eeb4f5a428b7610fdf2719fc7','87adce59647bc319ef0cf782a9be5379'),
'plankton_toxin':('a8991bc56cb2aa89b07eb8aa806e5dbc','6784a240202e8bb7c0ecc8e3a9dfb712')}
SEAS=['渤海','黄海','东海','南海','边界过渡区']

def sea(lon,lat):
    # Conservative study boxes, NOT official boundaries. Strait points remain unassigned.
    if not(np.isfinite(lon) and np.isfinite(lat)):return '坐标缺失'
    if 117<=lon<=121.5 and 37<=lat<=41.5:
        if lat>=39 or lon<=120.5:return '渤海'
        return '边界过渡区'
    if 120<=lon<=127 and 32<=lat<=40:return '黄海'
    if 119<=lon<=130 and 25<=lat<32:return '东海'
    if 105<=lon<=122 and 3<=lat<23.5:return '南海'
    return '边界过渡区'

def read(raw,name):
    p=Path(raw)/(name+'.xlsx')
    if hashlib.md5(p.read_bytes()).hexdigest()!=FILES[name][1]:raise ValueError('原始文件校验失败：'+name)
    x=pd.read_excel(p).rename(columns={'经度（东经°）':'longitude','纬度（北纬°）':'latitude','采样日期':'date','数据id':'record_id','藻类物种':'species'})
    for c in ['longitude','latitude']:x[c]=pd.to_numeric(x[c],errors='coerce')
    x['date']=pd.to_datetime(x.date,errors='coerce');x['sea']=[sea(a,b) for a,b in zip(x.longitude,x.latitude)]
    x['station']=x.longitude.round(4).astype(str)+'_'+x.latitude.round(4).astype(str)
    return x

def ece(y,p):
    total=0.
    for lo in np.linspace(0,0.9,10):
        mask=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1)
        if mask.any():total+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(total)

def run(root, marker="ITS1"):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score,brier_score_loss
    root=Path(root);raw=root/'data/china_mainland/raw';out=root/'outputs/china_mainland'/marker.replace(' ','_');out.mkdir(parents=True,exist_ok=True)
    coverage=[];points=[];sources=[]
    for name in FILES:
        x=read(raw,name);sources.append({'file':name+'.xlsx','md5':FILES[name][1],'rows':len(x),'url':'https://china.scidb.cn/download?fileId='+FILES[name][0]})
        for region,g in x.groupby('sea'):
            coverage.append(dict(dataset=name,sea=region,records=len(g),stations=g.station.nunique(),station_dates=len(g[['station','date']].drop_duplicates()),dates=g.date.nunique(),first=str(g.date.min().date()),last=str(g.date.max().date())))
        p=x[['longitude','latitude','date','sea','station']].dropna().drop_duplicates();p['dataset']=name;points.append(p)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    pd.concat(points).to_csv(out/'observation_points.csv',index=False)
    x=read(raw,'molecular');x=x[x['基因扩增靶区'].str.strip()==marker].copy()
    x['abundance']=pd.to_numeric(x['相对丰度'],errors='coerce')
    x=x.dropna(subset=['date','abundance','longitude','latitude','species'])
    x=x[x.abundance.between(0,1)].copy();x['species']=x.species.str.strip()
    # All depth/size fractions from a station-day stay in the same chronological partition.
    x=x.drop_duplicates('record_id');x['y']=(x.abundance>0).astype(int)
    x['month_sin']=np.sin(x.date.dt.month*2*np.pi/12);x['month_cos']=np.cos(x.date.dt.month*2*np.pi/12)
    x['split']=np.where(x.date.dt.year<=2020,'train','test')
    train=x[x.split=='train'].copy();test=x[x.split=='test'].copy()
    if len(train)==0 or len(test)==0:raise ValueError('无跨年留出样本')
    assert train.date.max()<test.date.min()
    nums=['longitude','latitude','month_sin','month_cos'];cols=nums+['species']
    prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['species'])])
    t0=time.perf_counter();a=prep.fit_transform(train[cols]);b=prep.transform(test[cols]);preprocess_seconds=time.perf_counter()-t0
    prior=train.groupby('species').y.mean();p0=test.species.map(prior).fillna(train.y.mean()).to_numpy()
    metrics=[];predictions=[]
    for seed in [17,42,73]:
        for name,model in [('物种检出率基线',None),('Logistic',LogisticRegression(C=1,max_iter=1000,random_state=seed)),('HistGradientBoosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1,early_stopping=False,random_state=seed))]:
            start=time.perf_counter()
            if model is None:p=p0
            else:model.fit(a,train.y);p=model.predict_proba(b)[:,1]
            seconds=time.perf_counter()-start
            pred=test[['record_id','station','date','sea','species','y']].copy();pred['probability']=p;pred['seed']=seed;pred['model']=name;predictions.append(pred)
            for region in ['全部']+SEAS:
                mask=np.ones(len(test),dtype=bool) if region=='全部' else test.sea.eq(region).to_numpy()
                if not mask.any():continue
                y=test.y.to_numpy()[mask];q=p[mask]
                metrics.append(dict(sea=region,model=name,seed=seed,n=len(y),station_dates=len(test.loc[mask,['station','date']].drop_duplicates()),positive_rate=float(y.mean()),AP=float(average_precision_score(y,q)) if len(np.unique(y))==2 else None,Brier=float(brier_score_loss(y,q)),ECE=ece(y,q),seconds=seconds))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    # Cluster bootstrap acknowledges multiple species/fractions per sampling occasion.
    intervals=[];pred=pd.concat(predictions);rng=np.random.default_rng(2026)
    for region in ['全部']+SEAS:
        d=pred[(pred.seed==17)&(pred.model=='HistGradientBoosting')].copy()
        if region!='全部':d=d[d.sea==region]
        d['cluster']=d.station+'_'+d.date.astype(str)
        groups=[g.index.to_numpy() for _,g in d.reset_index(drop=True).groupby('cluster')]
        if len(groups)<2:continue
        yy=d.y.to_numpy();pp=d.probability.to_numpy();samples=[]
        for _ in range(200):
            ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            if len(np.unique(yy[ix]))==2:samples.append(average_precision_score(yy[ix],pp[ix]))
        if samples:intervals.append(dict(sea=region,model='HistGradientBoosting',seed=17,clusters=len(groups),AP_low=float(np.quantile(samples,.025)),AP_high=float(np.quantile(samples,.975)),valid_replicates=len(samples)))
    pd.DataFrame(intervals).to_csv(out/'cluster_intervals.csv',index=False)
    manifest={'source':SOURCE,'doi':'10.57760/sciencedb.j00001.00831','license':'CC BY-NC-ND 4.0','files':sources,
      'marker':marker,'task':marker+'分子相对丰度大于0的跨年检出识别；不是藻华爆发或提前预警',
      'train_years':[2019,2020],'test_years':[2021],'tuning':'固定参数，无测试调参，无事后选最佳种子','seeds':[17,42,73],
      'inputs':cols,'excluded_inputs':['相对丰度','其他物种当期丰度','毒素浓度','未来观测'],
      'counts':{'train':len(train),'test':len(test),'train_station_dates':len(train[['station','date']].drop_duplicates()),'test_station_dates':len(test[['station','date']].drop_duplicates())},
      'test_ids_sha256':hashlib.sha256('\n'.join(sorted(test.record_id)).encode()).hexdigest(),
      'preprocessing_seconds':preprocess_seconds,'geography':'保守矩形研究分区，非官方海界；海峡及边界样本单列；不是行政归属判定',
      'limits':['未获得逐日连续事件标签，未验证中国内地7天藻华预警','渤海若无2021样本则不输出测试指标','未检出不代表无藻华；相对丰度不等于细胞浓度','样本可能存在同航次空间相关；重复种子不能替代独立航次','当前地理与季节特征模型未接入实测流场、温度或营养盐','未提供独立采样可用时间，属于回顾性检出识别，非实时运营验证']}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest['counts']));print(pd.DataFrame(metrics).groupby(['sea','model'])[['AP','Brier']].mean().to_string())

def render(root):
    from globalhab_demo.display_locale import st
    root=Path(root)
    marker=st.selectbox('检测方法',['ITS1','18S V4'],key='mainland_marker')
    out=root/'outputs/china_mainland'/marker.replace(' ','_')
    st.subheader('中国近海观测与跨年检验')
    if not (out/'manifest.json').exists():st.info('尚未生成中国近海结果，请运行 scripts/validate_mainland.py。');return
    m=json.loads((out/'manifest.json').read_text());coverage=pd.read_csv(out/'coverage.csv');points=pd.read_csv(out/'observation_points.csv');metrics=pd.read_csv(out/'metrics.csv')
    st.caption('2019—2021年实测调查。分子检出验证与藻华提前预警为不同任务。')
    region=st.selectbox('海域',['全部']+SEAS,key='mainland_sea')
    c=coverage if region=='全部' else coverage[coverage.sea==region]
    p=points if region=='全部' else points[points.sea==region]
    cols=st.columns(3);cols[0].metric('检测记录',int(c.records.sum()));cols[1].metric('采样坐标',p.station.nunique());cols[2].metric('独立采样日数',p.date.nunique())
    dates=pd.to_datetime(p['date'],errors='coerce').dropna()
    date_note=(f'当前海域实际采样日期范围：{dates.min().date()}—{dates.max().date()}；上方数值表示去重后的独立采样日数，并非日期编号。' if not dates.empty else '当前海域没有可用的采样日期。')
    st.caption(date_note)
    st.caption('记录数包含物种、扩增靶区和样品类别，不是独立事件数。海域为保守研究分区，边界点单列。')
    from globalhab_demo.map_style import point_map,draw_map
    draw_map(point_map(p[['latitude','longitude']].drop_duplicates()))
    st.dataframe(c,hide_index=True,use_container_width=True)
    st.write(f'跨年检验：以2019—2020年训练，2021年留出；统一{marker}检测方法，使用坐标、月份和物种预测分子检出。')
    s=metrics[metrics.sea==region]
    if s.empty:st.info('该海域没有符合当前协议的2021年测试样本，仅展示真实观测覆盖。')
    else:
        st.dataframe(s.groupby('model',sort=False).agg(AP=('AP','mean'),AP_std=('AP','std'),Brier=('Brier','mean'),ECE=('ECE','mean'),test_records=('n','first'),sampling_occasions=('station_dates','first'),seconds=('seconds','mean')).reset_index(),hide_index=True,use_container_width=True)
        st.caption('固定参数、相同留出记录、三个种子；确定性模型可产生相同结果。seconds是全测试任务耗时，非单个海域耗时。')
        ci=pd.read_csv(out/'cluster_intervals.csv');st.dataframe(ci[ci.sea==region],hide_index=True,use_container_width=True)
        st.caption('置信区间按采样坐标与日期整组重采样，不能据此声称模型显著优于基线。')
    with st.expander('来源、协议与适用范围'):
        st.write('来源：[Science Data Bank]('+SOURCE+')；原始数据许可CC BY-NC-ND 4.0，未改写为工程代码许可。')
        st.json(m)
    st.download_button('下载海域验证结果',data=s.to_csv(index=False).encode('utf-8-sig'),file_name='china_coastal_validation.csv')
