"""Shared multi-object scenario assessment with explicit, editable thresholds.

No species tolerances are invented: equal demo thresholds yield equal exposure
results. Cage-fish physiology remains an optional object-specific drill-down.
"""
from __future__ import annotations

from itertools import product
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

OBJECTS = ['网箱鱼', '捕捞资源', '贝类养殖', '海藻养殖']
FACTORS = ['藻华压力', '水温', '溶解氧']
COLORS = ['#167f99', '#bd8550', '#539bc4', '#43a894']
FONT = 'Microsoft YaHei, PingFang SC, Arial, sans-serif'


def default_thresholds():
    return pd.DataFrame({'评估对象': OBJECTS, '藻华上限': [60.] * 4,
                         '水温上限℃': [30.] * 4, '溶解氧下限mg/L': [4.] * 4})


def evaluate_joint(hab, temperature, oxygen, thresholds):
    """Compare the same environment with each object's user-set thresholds."""
    values = np.asarray([hab, temperature, oxygen], dtype=float)
    if not np.isfinite(values).all() or not (0 <= hab <= 100 and 0 <= temperature <= 45 and 0 < oxygen <= 14):
        raise ValueError('环境输入必须是有效数值，溶解氧必须大于0。')
    rows = []
    for row in thresholds.to_dict('records'):
        limits = np.asarray([row['藻华上限'], row['水温上限℃'], row['溶解氧下限mg/L']], dtype=float)
        if not np.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError('每个对象的阈值都必须是大于0的有效数值。')
        # Ratio >1 always means the adverse side of the configured threshold.
        ratios = [hab / limits[0], temperature / limits[1], limits[2] / oxygen]
        for factor, value, limit, ratio in zip(FACTORS, values, limits, ratios):
            rows.append({'对象': row['评估对象'], '因子': factor, '当前输入': float(value),
                         '自定阈值': float(limit), '阈值比': float(ratio),
                         '超过自定阈值': bool(ratio > 1.0 + 1e-10)})
    return pd.DataFrame(rows)


def perturb_joint(hab, temperature, oxygen, thresholds):
    frames = []
    for number, (hm, td, od) in enumerate(product([.9, 1., 1.1], [-.4, 0., .4], [-.5, 0., .5]), 1):
        frame = evaluate_joint(float(np.clip(hab * hm, 0, 100)),
                               float(np.clip(temperature + td, 0, 45)),
                               float(np.clip(oxygen + od, .1, 14)), thresholds)
        frame['情景编号'] = number
        frame['藻华倍率'] = hm
        frame['水温扰动℃'] = td
        frame['溶解氧扰动mg/L'] = od
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def matrix_figure(frame, *, stability=False):
    objects = list(dict.fromkeys(frame['对象']))
    if stability:
        matrix = frame.groupby(['对象', '因子'])['超过自定阈值'].mean().unstack().reindex(index=objects, columns=FACTORS) * 100
        text = [[f'{v:.0f}%' for v in row] for row in matrix.to_numpy()]
        z = matrix.to_numpy() / 100
        hover = '%{y} · %{x}<br>27种扰动组合中越阈占比 %{text}<extra></extra>'
        scale = [[0, '#eaf4f4'], [.5, '#8dbec4'], [1, '#157b90']]
    else:
        matrix = frame.pivot(index='对象', columns='因子', values='阈值比').reindex(index=objects, columns=FACTORS)
        text = [[f'{v:.2f}×' for v in row] for row in matrix.to_numpy()]
        z = matrix.to_numpy()
        hover = '%{y} · %{x}<br>阈值比 %{text}<br>大于1表示越过自定阈值<extra></extra>'
        scale = [[0, '#f0f7f8'], [.49, '#b7d8df'], [.5, '#f5dfb5'], [.75, '#dda068'], [1, '#b86643']]
    fig = go.Figure(go.Heatmap(x=FACTORS, y=objects, z=z, text=text, texttemplate='%{text}',
                              textfont=dict(size=16, color='#173f52'), zmin=0, zmax=1 if stability else 2,
                              colorscale=scale, showscale=False, xgap=8, ygap=8, hovertemplate=hover))
    fig.update_layout(height=300, margin=dict(l=85, r=5, t=35, b=15), font=dict(family=FONT, size=13, color='#173f52'),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(side='top', showgrid=False, fixedrange=True)
    fig.update_yaxes(autorange='reversed', showgrid=False, fixedrange=True)
    return fig


def region_map(selected):
    """One bounded map of all production settings, without repeating worlds."""
    from .bio_response import production_region_frame
    data = production_region_frame()
    fig = go.Figure()
    for color, (kind, group) in zip(COLORS, data.groupby('production_type', sort=False)):
        fig.add_trace(go.Scattergeo(lon=group.longitude, lat=group.latitude, text=group.region,
                      customdata=group.representative_stock, mode='markers', name=kind.replace('背景', ''),
                      marker=dict(size=[17 if n == selected else 9 for n in group.region], color=color,
                                  line=dict(width=1.5, color='white')),
                      hovertemplate='%{text}<br>%{customdata}<extra></extra>'))
    fig.update_geos(projection_type='equirectangular', showframe=False, showland=True, landcolor='#edf2ef',
                    showocean=True, oceancolor='#e3f2f6', showcountries=True, countrycolor='white',
                    showcoastlines=True, coastlinecolor='#b2c6ce', lonaxis_range=[-180, 180], lataxis_range=[-60, 85])
    fig.update_layout(height=270, margin=dict(l=0, r=0, t=5, b=30), geo=dict(bgcolor='rgba(0,0,0,0)'),
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(family=FONT, size=12, color='#173f52'),
                      legend=dict(orientation='h', y=-.02, x=.5, xanchor='center'))
    from .map_style import style_map
    return style_map(fig, global_view=True)


def render(root):
    from .display_locale import st
    from .bio_response import BIO_PRODUCTION_REGIONS, BIO_SCENARIO_PRESETS, compare_interventions, evaluate_intervention_robustness
    from .research_figures import pressure_timeline, pressure_comparison, robustness_distribution
    from .result_pool import register

    with st.container(border=True, key='research_section_joint_bio'):
        st.markdown('### 多对象联合响应沙盘')
        st.caption('同一海区、同一环境输入，同时对照网箱鱼、捕捞资源、贝类与海藻；默认阈值为可修改的演示设定。')
        selected = st.selectbox('联合评估情景海区', list(BIO_PRODUCTION_REGIONS),
                                index=list(BIO_PRODUCTION_REGIONS).index('智利巴塔哥尼亚峡湾'), key='joint_region')
        objects = st.multiselect('一起评估的对象', OBJECTS, default=OBJECTS, key='joint_objects',
                                help='评估对象代表本次假设情景，不表示这些对象都已在该海区取得实测数据。')
        st.plotly_chart(region_map(selected), width='stretch', config={'displayModeBar': False}, key='joint_region_map')
        preset_name = st.selectbox('共同环境情景', ['区域背景情景'] + list(BIO_SCENARIO_PRESETS), key='joint_preset')
        region = BIO_PRODUCTION_REGIONS[selected]
        defaults = BIO_SCENARIO_PRESETS['复合高压科研情景']
        preset = {**defaults, **region} if preset_name == '区域背景情景' else BIO_SCENARIO_PRESETS[preset_name]
        key = f'{selected}_{preset_name}'
        a, b, c, d = st.columns(4)
        hab = a.slider('藻华压力（0–100）', 0., 100., float(preset['hab_pressure']), 1., key='joint_hab_'+key)
        background = b.slider('背景水温（℃）', 0., 35., 24., .5, key='joint_temp_'+key)
        mhw = c.slider('热浪增温（℃）', 0., 5., float(preset['mhw_intensity_c']), .1, key='joint_mhw_'+key)
        oxygen = d.slider('溶解氧（mg/L）', .5, 14., float(preset['dissolved_oxygen_mg_l']), .1, key='joint_do_'+key)
        temperature = background + mhw
        st.caption(f'情景水温 {temperature:.1f}℃ = 背景水温 + 热浪增温；所有对象共用以上输入。海区参数为情景初值。')
        if not objects:
            st.info('至少选择一个对象后显示联合评估。')
            return None
        with st.expander('各对象的评估阈值 · 可分别修改', expanded=True):
            st.caption('各对象初始阈值相同，避免无依据地假定物种耐受差异；可按物种或场站资料分别修改。')
            thresholds = st.data_editor(default_thresholds(), hide_index=True, disabled=['评估对象'], key='joint_thresholds',
                column_config={
                    '藻华上限': st.column_config.NumberColumn(min_value=1., max_value=100., step=1.),
                    '水温上限℃': st.column_config.NumberColumn(min_value=1., max_value=45., step=.5),
                    '溶解氧下限mg/L': st.column_config.NumberColumn(min_value=.1, max_value=14., step=.1)})
        thresholds = thresholds[thresholds['评估对象'].isin(objects)].copy()
        try:
            result = evaluate_joint(hab, temperature, oxygen, thresholds)
            perturbations = perturb_joint(hab, temperature, oxygen, thresholds)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            return None
        if len(thresholds) > 1 and len(thresholds[['藻华上限','水温上限℃','溶解氧下限mg/L']].drop_duplicates()) == 1:
            st.info('当前所选对象的三项阈值相同，因此两张图的对应行相同；这是共同环境的阈值对照，尚不能区分对象的生物敏感性。可在上表按物种或场站资料分别设置。')
        left, right = st.columns(2, gap='large')
        with left:
            st.markdown('#### 当前环境 · 多对象并列对照')
            st.plotly_chart(matrix_figure(result), width='stretch', key='joint_current')
            st.caption('阈值比 > 1 为越阈；溶解氧按“下限 ÷ 当前值”计算。')
        with right:
            st.markdown('#### 参数扰动 · 各对象越阈稳定性')
            st.plotly_chart(matrix_figure(perturbations, stability=True), width='stretch', key='joint_stability')
            st.caption('27种组合：藻华±10%、水温±0.4℃、溶解氧±0.5 mg/L；格内为越阈组合占比。')
        st.caption('以上为环境阈值对照，不是物种损伤概率；相同阈值显示相同结果。')
        payload = {'region': selected, 'objects': objects, 'inputs': {'hab': hab, 'temperature': temperature, 'mhw': mhw, 'oxygen': oxygen},
                   'thresholds': thresholds.to_dict('records'), 'assessment': result.to_dict('records'),
                   'perturbation_exceedance': perturbations.groupby(['对象', '因子'])['超过自定阈值'].mean().reset_index().to_dict('records'),
                   'scope': '共同环境情景与用户阈值对照；非物种生理预测；非发生概率'}
        register('多对象联合沙盘', '未标定的共同环境情景', payload)
        d1, d2 = st.columns(2)
        d1.download_button('下载联合评估', result.to_csv(index=False).encode('utf-8-sig'), 'joint_bio_assessment.csv', 'text/csv')
        d2.download_button('下载完整情景与阈值', json.dumps(payload, ensure_ascii=False, indent=2), 'joint_bio_scenario.json', 'application/json')
        with st.expander('查看27情景明细与下载'):
            st.dataframe(perturbations, hide_index=True)
            st.download_button('下载27情景结果', perturbations.to_csv(index=False).encode('utf-8-sig'), 'joint_bio_perturbations.csv', 'text/csv')

        # One shared workflow; the existing uncalibrated fish process model is
        # available as a drill-down. Never apply feeding or density to shells.
        card = None
        if '网箱鱼' in objects:
            with st.expander('网箱鱼干预细节 · 沿用上方共同环境', expanded=False):
                st.caption('网箱鱼专属过程模型；贝类和捕捞资源的环境结果已在上方并列展示。')
                p1, p2, p3, p4 = st.columns(4)
                density = p1.slider('网箱密度（kg/m³）', 2., 45., float(preset['stocking_density_kg_m3']), 1., key='joint_density_'+key)
                feed = p2.slider('计划投喂（%）', 0., 120., float(preset['planned_feeding_pct']), 5., key='joint_feed_'+key)
                duration = p3.slider('压力持续（小时）', 12, 72, int(preset['hab_duration_hours']), 6, key='joint_duration_'+key)
                horizon = p4.selectbox('模拟时长（小时）', [48, 72, 96], index=1, key='joint_horizon')
                inputs = dict(hab_pressure=hab, mhw_intensity_c=mhw, dissolved_oxygen_mg_l=oxygen,
                              stocking_density_kg_m3=density, planned_feeding_pct=feed,
                              hab_duration_hours=min(duration, horizon), horizon_hours=horizon)
                simulation = st.cache_data(show_spinner=False)(compare_interventions)(**inputs)
                robust = st.cache_data(show_spinner=False)(evaluate_intervention_robustness)(**inputs)
                card = robust['card']
                st.plotly_chart(pressure_timeline(simulation['trajectories']), width='stretch', key='joint_fish_timeline')
                st.plotly_chart(pressure_comparison(simulation['summary']), width='stretch', key='joint_fish_comparison')
                st.markdown('#### 参数扰动下的干预稳定性')
                st.plotly_chart(robustness_distribution(robust['detail'], robust['summary']), width='stretch', key='joint_fish_robustness')
                st.caption('每行包含81种扰动组合；箱体为四分位范围，横线为全部范围，点为中位数。零出现率的方案也完整显示。')
                st.caption('非劣出现率：在降低压力与保留摄食机会两项上不被其他方案同时超越的组合占比，并非现场有效率。')
                with st.expander('干预对照、模型参数与下载'):
                    st.dataframe(simulation['summary'], hide_index=True)
                    st.dataframe(robust['summary'], hide_index=True)
                    st.dataframe(simulation['parameters'], hide_index=True)
                    for label, frame, name in [('响应轨迹', simulation['trajectories'], 'trajectories'),
                                                ('干预对照', simulation['summary'], 'comparison'),
                                                ('模型参数', simulation['parameters'], 'parameters'),
                                                ('81情景结果', robust['detail'], 'robustness')]:
                        st.download_button('下载'+label, frame.to_csv(index=False).encode('utf-8-sig'), 'cage_fish_'+name+'.csv', 'text/csv')
                    st.download_button('下载网箱鱼沙盘卡', json.dumps(simulation['scenario_card'], ensure_ascii=False, indent=2), 'cage_fish_sandbox_card.json')
                register('网箱鱼干预细节', '同一环境下的未标定过程模型', {'region': selected, 'summary': simulation['summary'].to_dict('records')})
        return card
