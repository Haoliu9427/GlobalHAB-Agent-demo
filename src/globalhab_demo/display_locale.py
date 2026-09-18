"""Chinese display adapter. Internal model IDs and exported evidence stay intact."""
import re
import numpy as np
import streamlit as _st
import pandas as pd

def chinese(value):
    if not isinstance(value, str):
        return value
    value = re.sub(r'\b(downstream|upstream|reversed|local)\b', lambda m: {'downstream':'顺流传播','upstream':'上游信号','reversed':'逆流对照','local':'局地变化'}[m[0]], value)
    value = re.sub(r'(?<![A-Za-z0-9])(\d+)d\b', r'\1天', value)
    value = re.sub(r'Agent\s+探索', 'Agent探索', value)
    return value.replace('Route / Lag', '传播方向 / 时间差').replace('Route/Lag', '传播方向 / 时间差')

def translated(value):
    if isinstance(value,np.ndarray) and value.dtype.kind in {"U","S","O"}:return translated(value.tolist())
    if isinstance(value,str):return chinese(value)
    if isinstance(value,dict):return {k:translated(v) for k,v in value.items()}
    if isinstance(value,list):return [translated(v) for v in value]
    if isinstance(value,tuple):return tuple(translated(v) for v in value)
    if isinstance(value,pd.DataFrame):
        out=value.copy()
        for col in out.select_dtypes(include=['object','string']).columns:out[col]=out[col].map(chinese)
        out.columns=[chinese(c) for c in out.columns]
        return out
    return value

class Display:
    def __getattr__(self,name):
        original=getattr(_st,name)
        if name in {'markdown','caption','write','text','info','warning','success','error','header','subheader','title','metric','dataframe','table','json'}:
            def output(*args,**kwargs):
                return original(*(translated(a) for a in args),**kwargs)
            return output
        if name=='plotly_chart':
            def chart(fig,*args,**kwargs):
                import plotly.graph_objects as go
                if hasattr(fig,'to_plotly_json'):
                    def labels(obj):
                        if isinstance(obj,dict):
                            return {k:(translated(v) if k in {'text','name','hovertext','hovertemplate','ticktext','labels','x','y'} else labels(v)) for k,v in obj.items()}
                        if isinstance(obj,list):return [labels(v) for v in obj]
                        return obj
                    fig=go.Figure(labels(fig.to_plotly_json()))
                return original(fig,*args,**kwargs)
            return chart
        return original
st=Display()
