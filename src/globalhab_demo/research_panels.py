"""Presentation-only chart and detail cards for research workspaces."""
from globalhab_demo.display_locale import st

def research_plot(fig,*args,**kwargs):
    with st.container(border=True):
        return st.plotly_chart(fig,*args,**kwargs)

def research_table(*args,**kwargs):
    with st.expander('查看数据明细',expanded=False):
        return st.dataframe(*args,**kwargs)
