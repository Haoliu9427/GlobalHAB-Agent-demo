"""Readable figures from computed results; hide raw audit prose by default."""
from globalhab_demo.display_locale import st

def render_metrics(table,interpretation=''):
    import plotly.express as px
    if table.empty:return
    split=st.selectbox('评估范围',list(table.split.unique()),key='summary_metric_split') if table.split.nunique()>1 else table.split.iloc[0]
    data=table[table.split==split].groupby('model',as_index=False).agg(AP=('AP','mean'),Brier=('Brier','mean'),ECE=('ECE','mean'),AP_sd=('AP','std'))
    cols=st.columns(3)
    for col,metric,title in zip(cols,['AP','Brier','ECE'],['风险排序 · 越高越好','概率误差 · 越低越好','校准偏差 · 越低越好']):
        with col,st.container(border=True):
            st.markdown('#### '+title)
            fig=px.bar(data,x=metric,y='model',orientation='h',text=metric,color_discrete_sequence=['#168b9d'],labels={'model':'',metric:metric})
            fig.update_traces(texttemplate='%{x:.3f}',textposition='outside',cliponaxis=False)
            fig.update_layout(height=230,margin=dict(l=5,r=45,t=12,b=10),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
    with st.expander('指标含义与完整结果'):
        st.write('AP衡量风险排序，不是准确率；Brier衡量概率预测误差；ECE衡量概率与实际频率的偏离。图中为随机种子均值。')
        st.dataframe(table,hide_index=True)
    if interpretation:
        with st.expander('运行与审计说明'):st.text(interpretation)
