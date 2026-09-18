"""Read-only experiment viewer. Heavy training libraries are imported on demand."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def render(root):
    from globalhab_demo.display_locale import st
    import plotly.express as px
    from globalhab_demo.research_panels import research_plot, research_table
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
    df['model']=df['model'].replace({'Chronos':'Chronos-Bolt-small','Chronos+EcoFusion':'Chronos-Bolt-small + EcoFusion'})
    st.write('香港任务预测的是报告发生；HABSOS任务预测的是有复测记录位置的浓度阈值事件。两个任务的AP不能直接比较。')
    counts=manifest['counts'];columns=st.columns(4)
    for col,key,title in zip(columns,['train','validation','calibration','test'],['训练样本','选模样本','校准样本','时间留出样本']):
        col.metric(title,f'{counts[key]:,}')
    splitname=st.radio('测试范围',df['split'].unique(),format_func=lambda x:'未见网格的未来样本' if x=='spatial_test' else '已见区域的未来样本',horizontal=True,key='real_eval_split')
    subset=df[df['split']==splitname]
    summary=subset.groupby('model',sort=False).agg(AP=('AP','mean'),AP_std=('AP','std'),Brier=('Brier','mean'),ECE=('ECE','mean'),seconds=('seconds','mean'),parameters=('parameters','max')).reset_index()
    figure=px.scatter(summary,x='AP',y='model',error_x='AP_std',color='AP',color_continuous_scale=['#9ecfd9','#087d95'],title='同一留出集：模型排名与种子波动',labels={'AP':'AP · 越高越好','model':''})
    figure.update_traces(marker=dict(size=14))
    figure.update_layout(coloraxis_showscale=False)
    figure.update_layout(showlegend=False,font=dict(family='Microsoft YaHei, sans-serif'),height=390)
    research_plot(figure,use_container_width=True)
    research_table(summary,use_container_width=True,hide_index=True)
    st.caption('误差线为三个种子的标准差；不等于统计显著性。Brier、ECE越低越好。seconds包含该模型选型与推理；未含数据下载。')
    warning=folder/'warning_threshold_results.csv'
    if warning.exists():
        with st.expander('固定预警阈值下的命中、误报与漏报'):
            w=pd.read_csv(warning);research_table(w[w['split']==splitname],hide_index=True,use_container_width=True)
            st.caption('阈值仅由校准期10%报警预算确定，测试期不调整。未来报警比例可以变化；这不是实际养殖损失或自动运营指令。')
    audit=folder/'comparative_audit.csv'
    if audit.exists():
        with st.expander('配对置信区间与模型比较'):
            research_table(pd.read_csv(audit),hide_index=True,use_container_width=True)
            st.caption('按季度整块重采样；区间跨0时不能宣称稳定胜出。不是因果效应。')
    views=st.tabs(['时间与区域','稳定性','输入贡献','实验选择','基础模型对照'])
    with views[0]:
        st.write('三个分界日期：'+' / '.join(manifest['cutoffs']))
        rows=pd.read_csv(folder/f'{splitname}_rows.csv')
        st.write(f"目标阳性比例：{rows.y.mean():.2%}；{rows.site.nunique()}个分析单元。")
        if manifest['task'].startswith('china_hk'):
            st.caption('地理覆盖为香港水域；以下单点为区域示意，非采样点坐标。')
        from globalhab_demo.map_style import point_map,draw_map
        draw_map(point_map(rows[['latitude','longitude']].dropna().drop_duplicates()))
        st.caption('HABSOS的空间留出以0.1°网格为单位；邻近网格仍可能相关，不等同于跨海域外推验证。')
        with st.expander('完整任务定义与样本校验值'):st.json(manifest)
        if (folder/'input_availability.csv').exists():
            with st.expander('实测变量覆盖与缺测比例'):
                research_table(pd.read_csv(folder/'input_availability.csv'),hide_index=True,use_container_width=True)
    with views[1]:
        values=pd.read_csv(folder/'stability.csv');model=st.selectbox('查看模型',sorted(values.model.unique()),format_func=lambda n:{'Chronos':'Chronos-Bolt-small','Chronos+EcoFusion':'Chronos-Bolt-small + EcoFusion'}.get(n,n),key='real_stability_model')
        shown=values[values.model==model].copy();shown['model']=shown['model'].replace({'Chronos':'Chronos-Bolt-small','Chronos+EcoFusion':'Chronos-Bolt-small + EcoFusion'})
        research_table(shown,hide_index=True,use_container_width=True)
        st.caption('missing20与noise01为人为输入扰动；observed_high_temperature为训练温度90分位以上的实测子集，并非所有极端天气。')
    with views[2]:
        imp=pd.read_csv(folder/'explainability.csv').groupby('feature',as_index=False).AP_drop.mean()
        research_plot(px.bar(imp,x='AP_drop',y='feature',orientation='h',title='遮蔽输入后的AP变化'),use_container_width=True)
        st.caption('正值表示遮蔽后排名能力下降。相关特征、分布变化会影响该量，不能解释为生物因果作用。物理方向与时滞证据仍在“科学解释”中单独呈现。')
    with views[3]:
        research_table(pd.read_json(folder/'real_agent_log.jsonl',lines=True),hide_index=True,use_container_width=True)
        st.caption('受约束实验控制器根据验证反馈调整网络容量；测试标签不进入选择规则。原24候选、8步合成探索保留独立。')
    with views[4]:
        chosen=st.selectbox('基础模型', ['Qwen2.5-0.5B-Instruct','Chronos-Bolt-small'],key='published_foundation_model')
        if chosen=='Qwen2.5-0.5B-Instruct':
            exp=base/'china_hk_llm';statusfile=exp/'status.json'
        else:
            exp=base/'china_hk_foundation_v2';statusfile=exp/'foundation_status.json'
        if statusfile.exists():
            status=json.loads(statusfile.read_text())
            st.write(chosen+' · '+('已完成' if status.get('status')=='completed' else '尚未完成'))
            if status.get('status')=='completed' and (exp/'metrics.csv').exists():
                scores=pd.read_csv(exp/'metrics.csv')
                if chosen.startswith('Chronos'):
                    scores=scores[scores.model.isin(['Chronos','Chronos+EcoFusion'])]
                scores['model']=scores['model'].replace({'LLM':'Qwen2.5-0.5B-Instruct','LLM+EcoFusion':'Qwen2.5-0.5B-Instruct + EcoFusion',
                    'Chronos':'Chronos-Bolt-small','Chronos+EcoFusion':'Chronos-Bolt-small + EcoFusion'})
                research_table(scores,hide_index=True,use_container_width=True)
                st.download_button('下载模型对照指标',data=scores.to_csv(index=False).encode('utf-8-sig'),file_name=chosen+'_metrics.csv')
            st.caption('这里是香港固定留出实验，与上方其他海域任务分开。Qwen是语言模型，Chronos是时序基础模型。预训练语料重叠不能独立排除。')
            with st.expander('模型及运行记录'):st.json(status)
        else:st.info('当前工程尚无该模型的已完成实验记录。')
        st.caption('分析自己的观测数据，请在左侧工作区选择“数据分析”。')
    st.download_button('下载当前实验指标',data=df.to_csv(index=False).encode('utf-8-sig'),file_name=folder.name+'_metrics.csv')
