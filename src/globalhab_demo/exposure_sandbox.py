"""Object-specific environmental threshold comparison, separate from fish physiology."""
from globalhab_demo.display_locale import st

def render(kind,root):
    import pandas as pd,plotly.graph_objects as go
    from globalhab_demo.result_pool import register
    st.markdown('### '+kind+' · 环境压力对照')
    profiles={'贝类养殖':'重点复核藻种与毒素、缺氧和温度暴露。','海藻养殖':'重点复核温度、光照、水质及营养条件。','捕捞资源':'重点复核水团变化、缺氧与栖息地暴露。'}
    st.caption(profiles[kind]+'本页比较输入与自定阈值，不使用网箱鱼生理模型。')
    with st.container(border=True):
        a,b=st.columns(2)
        with a:
            temp=st.slider('情景水温（℃）',0.,40.,28.,.5,key='ex_temp_'+kind)
            oxygen=st.slider('情景溶解氧（mg/L）',0.,15.,5.,.1,key='ex_do_'+kind)
            hab=st.slider('情景藻华压力指数',0.,100.,60.,1.,key='ex_hab_'+kind)
        with b:
            tlimit=st.number_input('自定水温上限（℃）',min_value=1.,max_value=40.,value=30.,key='ex_tlimit_'+kind)
            dlimit=st.number_input('自定溶解氧下限（mg/L）',min_value=.1,max_value=15.,value=4.,key='ex_dlimit_'+kind)
            hlimit=st.number_input('自定藻华压力上限',min_value=1.,max_value=100.,value=60.,key='ex_hlimit_'+kind)
    st.caption('默认阈值仅用于界面演示，不是上述对象的耐受阈值或监管标准；请依据具体物种与场站校准。')
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=1,cols=3,subplot_titles=['水温（℃）','溶解氧（mg/L）','藻华压力（0–100）'])
    for i,(v,limit,maxv) in enumerate([(temp,tlimit,40),(oxygen,dlimit,15),(hab,hlimit,100)],1):
        fig.add_trace(go.Bar(x=['当前输入'],y=[v],marker_color='#168d9e',text=[f'{v:g}'],textposition='outside'),row=1,col=i)
        fig.add_hline(y=limit,line_dash='dot',line_color='#d08a59',row=1,col=i)
        fig.update_yaxes(range=[0,maxv*1.1],row=1,col=i)
    fig.update_layout(height=300,showlegend=False,margin=dict(t=35,b=20,l=15,r=15))
    st.plotly_chart(fig,use_container_width=True)
    warnings=[label for label,flag in [('水温超过自定上限',temp>tlimit),('溶解氧低于自定下限',oxygen<dlimit),('藻华压力超过自定上限',hab>hlimit)] if flag]
    st.write('；'.join(warnings) if warnings else '当前输入均未越过自定阈值；不代表无生态或食品安全风险。')
    if st.button('记录本情景供模型解读',key='ex_save_'+kind):
        register('生物响应 · '+kind,'未标定的环境情景对照',{'temperature':temp,'oxygen':oxygen,'hab_pressure':hab,'user_thresholds':[tlimit,dlimit,hlimit],'flags':warnings,'scope':'非生理预测，不输出死亡率或毒素结论'});st.success('已加入本会话结果汇总。')
