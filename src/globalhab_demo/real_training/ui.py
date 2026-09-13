"""Read-only experiment viewer. Heavy training libraries are imported on demand."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def render(root):
    import streamlit as st
    import plotly.express as px
    root=Path(root);base=root/'outputs/real_training'
    mode=st.radio('验证数据', ['连续观测预测','中国近海调查'],horizontal=True,key='real_data_mode')
    if mode=='中国近海调查':
        from .mainland import render as render_mainland
        render_mainland(root)
        return
    st.subheader('真实观测上的预测验证')
    st.caption('这里展示已完成的固定时间留出实验。上方合成情景设置不会改变这些实验结果。')
    folders=[p for p in sorted(base.glob('*')) if (p/'split_manifest.json').exists() and (p/'metrics.csv').exists()
             and p.name not in ['china_hk_foundation']]
    if (base/'china_hk_foundation_v2/metrics.csv').exists():
        folders=[p for p in folders if p.name!='china_hk_7d']
    if not folders:
        st.info('尚无已完成的真实数据训练结果。请运行 scripts/train_real.py 后再查看。');return
    labels={'habsos_7d':'墨西哥湾 · 7–10天后采样浓度', 'habsos_14d':'墨西哥湾 · 14–17天后采样浓度',
        'habsos_30d':'墨西哥湾 · 30–33天后采样浓度', 'china_hk_7d':'中国香港 · 下一周赤潮报告',
        'china_hk_foundation_v2':'中国香港 · 时序基础模型对照'}
    folder=st.selectbox('验证任务',folders,format_func=lambda p:labels.get(p.name,p.name),key='real_eval_task')
    manifest=json.loads((folder/'split_manifest.json').read_text())
    df=pd.read_csv(folder/'metrics.csv')
    st.write('香港任务预测的是报告发生；HABSOS任务预测的是有复测记录位置的浓度阈值事件。两个任务的AP不能直接比较。')
    counts=manifest['counts'];columns=st.columns(4)
    for col,key,title in zip(columns,['train','validation','calibration','test'],['训练样本','选模样本','校准样本','时间留出样本']):
        col.metric(title,f'{counts[key]:,}')
    splitname=st.radio('测试范围',df['split'].unique(),format_func=lambda x:'未见网格的未来样本' if x=='spatial_test' else '已见区域的未来样本',horizontal=True,key='real_eval_split')
    subset=df[df['split']==splitname]
    summary=subset.groupby('model',sort=False).agg(AP=('AP','mean'),AP_std=('AP','std'),Brier=('Brier','mean'),ECE=('ECE','mean'),seconds=('seconds','mean'),parameters=('parameters','max')).reset_index()
    figure=px.bar(summary,x='model',y='AP',error_y='AP_std',color='model',title='同一留出集：Average Precision（越高越好）')
    figure.update_layout(showlegend=False,font=dict(family='Microsoft YaHei, sans-serif'),height=390)
    st.plotly_chart(figure,use_container_width=True)
    st.dataframe(summary,use_container_width=True,hide_index=True)
    st.caption('误差线为三个种子的标准差；不等于统计显著性。Brier、ECE越低越好。seconds包含该模型选型与推理；未含数据下载。')
    warning=folder/'warning_threshold_results.csv'
    if warning.exists():
        with st.expander('固定预警阈值下的命中、误报与漏报'):
            w=pd.read_csv(warning);st.dataframe(w[w['split']==splitname],hide_index=True,use_container_width=True)
            st.caption('阈值仅由校准期10%报警预算确定，测试期不调整。未来报警比例可以变化；这不是实际养殖损失或自动运营指令。')
    audit=folder/'comparative_audit.csv'
    if audit.exists():
        with st.expander('配对置信区间与模型比较'):
            st.dataframe(pd.read_csv(audit),hide_index=True,use_container_width=True)
            st.caption('按季度整块重采样；区间跨0时不能宣称稳定胜出。不是因果效应。')
    views=st.tabs(['时间与区域','稳定性','输入贡献','实验选择','语言模型','自有观测'])
    with views[0]:
        st.write('三个分界日期：'+' / '.join(manifest['cutoffs']))
        rows=pd.read_csv(folder/f'{splitname}_rows.csv')
        st.write(f"目标阳性比例：{rows.y.mean():.2%}；{rows.site.nunique()}个分析单元。")
        if manifest['task'].startswith('china_hk'):
            st.caption('地理覆盖为香港水域；以下单点为区域示意，非采样点坐标。')
        st.map(rows[['latitude','longitude']].dropna().drop_duplicates())
        st.caption('HABSOS的空间留出以0.1°网格为单位；邻近网格仍可能相关，不等同于跨海域外推验证。')
        with st.expander('完整任务定义与样本校验值'):st.json(manifest)
        if (folder/'input_availability.csv').exists():
            with st.expander('实测变量覆盖与缺测比例'):
                st.dataframe(pd.read_csv(folder/'input_availability.csv'),hide_index=True,use_container_width=True)
    with views[1]:
        values=pd.read_csv(folder/'stability.csv');model=st.selectbox('查看模型',sorted(values.model.unique()),key='real_stability_model')
        st.dataframe(values[values.model==model],hide_index=True,use_container_width=True)
        st.caption('missing20与noise01为人为输入扰动；observed_high_temperature为训练温度90分位以上的实测子集，并非所有极端天气。')
    with views[2]:
        imp=pd.read_csv(folder/'explainability.csv').groupby('feature',as_index=False).AP_drop.mean()
        st.plotly_chart(px.bar(imp,x='AP_drop',y='feature',orientation='h',title='遮蔽输入后的AP变化'),use_container_width=True)
        st.caption('正值表示遮蔽后排名能力下降。相关特征、分布变化会影响该量，不能解释为生物因果作用。物理方向与时滞证据仍在“科学解释”中单独呈现。')
    with views[3]:
        st.dataframe(pd.read_json(folder/'real_agent_log.jsonl',lines=True),hide_index=True,use_container_width=True)
        st.caption('受约束实验控制器根据验证反馈调整网络容量；测试标签不进入选择规则。原24候选、8步合成探索保留独立。')
    with views[4]:
        llm=base/'china_hk_llm';status=llm/'status.json'
        if status.exists():
            status=json.loads(status.read_text());st.write('语言模型：'+status.get('model','')+'；状态：'+status['status'])
            if status['status']=='completed':
                st.dataframe(pd.read_csv(llm/'metrics.csv'),hide_index=True,use_container_width=True)
                st.caption('这是香港固定留出集的对照，与当前选择的其他海域结果不混合。小型语言模型不代表所有LLM；公开预训练数据的重叠无法独立排除。')
            with st.expander('模型及提示词记录'):st.json(status)
        else:st.info('尚未运行语言模型对照。Chronos为时序基础模型，不是聊天LLM。')
    with views[5]:
        st.write('上传连续现场观测CSV，先检查是否具备时间留出验证条件。缺失事件标签不作为阴性。')
        uploaded=st.file_uploader('现场观测CSV',type=['csv'],key='real_training_upload')
        if uploaded is not None:
            try:
                from .data import field,split
                ds=field(uploaded);parts,info=split(ds)
                st.success(f'形成{len(ds.rows):,}条未来目标样本，可按时间与站点分割。')
                st.json(info['counts']);st.caption('上传仅在当前会话解析，未训练、未写入仓库。用scripts/train_real.py --dataset china_field --input 文件路径执行完整实验。')
            except Exception as exc:st.error('数据检查未通过：'+str(exc))
        st.download_button('下载字段模板',data='station_id,date,available_at,latitude,longitude,observed_event,value,source,temperature,salinity,dissolved_oxygen\n',file_name='field_observations_template.csv')
    st.download_button('下载当前实验指标',data=(folder/'metrics.csv').read_bytes(),file_name=folder.name+'_metrics.csv')
