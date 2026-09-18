"""Research figures share axes, compact labels and evidence-preserving hover details."""
import plotly.graph_objects as go
import numpy as np

INK='#173f52';TEAL='#168c96';PALE='#bed5de';CORAL='#bc7d58'
def finish(fig,height=360):
    fig.update_layout(height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Microsoft YaHei, Arial',size=12,color=INK),
        margin=dict(l=15,r=45,t=55,b=45),legend=dict(orientation='h',y=1.16,x=0),
        hoverlabel=dict(bgcolor='white'))
    fig.update_xaxes(zeroline=False,gridcolor='#eaf0f2')
    fig.update_yaxes(zeroline=False,gridcolor='#eaf0f2')
    return fig

def pressure_timeline(data):
    matrix=data.pivot_table(index='intervention',columns='hour',values='relative_physiological_pressure',aggfunc='mean')
    fig=go.Figure(go.Heatmap(x=matrix.columns,y=matrix.index,z=matrix.values,zmin=0,zmax=100,
        colorscale=[[0,'#eff6f5'],[.2,'#bddbd9'],[.4,'#398e98'],[.6,'#146375'],[.8,'#dfb078'],[1,'#b75b47']],
        xgap=0,ygap=7,colorbar=dict(title='压力',thickness=9,len=.8,tickvals=[0,50,100]),
        hovertemplate='%{y}<br>模拟 %{x} 小时<br>相对生理压力 %{z:.1f}/100<extra></extra>'))
    fig.update_layout(title='压力随时间的变化',xaxis_title='模拟时间（小时）',yaxis_autorange='reversed')
    return finish(fig,340)

def pressure_comparison(data):
    d=data.sort_values('peak_pressure_index',ascending=False)
    fig=go.Figure(go.Bar(x=d.peak_pressure_index,y=d.intervention,orientation='h',
        marker_color=[PALE if x=='维持监测' else TEAL for x in d.intervention],
        
        error_x=dict(type='data',array=d.peak_pressure_upper-d.peak_pressure_index,arrayminus=d.peak_pressure_index-d.peak_pressure_lower,color=INK,thickness=1),
        hovertemplate='%{y}<br>峰值压力 %{x:.1f}<br>误差线：±15%参数敏感性包络，非置信区间<extra></extra>'))
    for row in d.itertuples():
        fig.add_annotation(x=float(row.peak_pressure_upper)+2,y=row.intervention,text=f'{row.peak_pressure_index:.1f}',showarrow=False,xanchor='left',font_size=12)
    fig.update_layout(title='哪种情景的峰值压力更低',xaxis=dict(range=[0,110],title='相对压力 ↓'),yaxis_autorange='reversed',bargap=.48,showlegend=False)
    return finish(fig,350)

def baseline_comparison(data):
    d=data.sort_values('pr_auc')
    fig=go.Figure(go.Bar(x=d.pr_auc,y=d['方法'].str.replace('GlobalHAB-Agent最佳候选','Agent候选'),orientation='h',
        marker_color=[TEAL if 'Agent' in x else PALE for x in d['方法']],
        text=d.pr_auc,texttemplate='%{x:.3f}',textposition='outside',cliponaxis=False,
        hovertemplate='%{y}<br>同一留出集 AP %{x:.3f}<extra></extra>'))
    fig.update_layout(title='同一留出集 · 风险排序对比',xaxis=dict(title='AP ↑',range=[0,max(.1,d.pr_auc.max()*1.3)]),showlegend=False,bargap=.46)
    return finish(fig,350)

def exploration_trace(log):
    d=log.sort_values('step')
    y=d.pr_auc.astype(float);c=y.cummax()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.step,y=c,mode='lines',name='截至该步的最佳结果',
        line=dict(color=TEAL,width=3,shape='hv'),fill='tozeroy',fillcolor='rgba(22,140,150,.08)',
        hovertemplate='截至第%{x}步<br>最佳AP %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=d.step,y=y,mode='markers+text',name='本步试验',
        text=[f'{v:.3f}' for v in y],textposition='bottom center',marker=dict(size=12,color=[TEAL if v==m else CORAL for v,m in zip(y,c)],line=dict(color='white',width=2)),
        hovertemplate='第%{x}步<br>本步AP %{y:.3f}<extra></extra>'))
    fig.update_layout(xaxis=dict(title='实验步',dtick=1),yaxis=dict(title='AP ↑',range=[0,max(.1,y.max()*1.25)]))
    return finish(fig,330)


def risk_series(predictions):
    from plotly.subplots import make_subplots
    d=predictions.sort_values('date')
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.75,.25],vertical_spacing=.12)
    fig.add_trace(go.Scatter(x=d.date,y=d.risk_probability,mode='lines',line=dict(color=TEAL,width=2),fill='tozeroy',fillcolor='rgba(22,140,150,.1)',name='预测风险',
        hovertemplate='%{x}<br>预测概率 %{y:.3f}<extra></extra>'),row=1,col=1)
    fig.add_trace(go.Heatmap(x=d.date,y=['实际事件','排名报警'],z=[d.hab_event,d.top20_alert],zmin=0,zmax=1,
        colorscale=[[0,'#edf3f4'],[1,TEAL]],showscale=False,ygap=4,
        hovertemplate='%{x}<br>%{y}：%{z}<extra></extra>'),row=2,col=1)
    fig.update_yaxes(range=[0,1],title='预测概率',row=1,col=1)
    fig.update_layout(showlegend=False)
    return finish(fig,340)
