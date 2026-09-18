"""Shared geographic presentation. Does not change scientific values."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os


def _map_backend():
    """Return the presentation backend without changing scientific coordinates."""
    env = os.getenv("GLOBALHAB_MAP_BACKEND", "").strip().lower()
    if env in {"geo", "osm"}:
        return env
    try:
        from globalhab_demo.display_locale import st
        value = str(st.session_state.get("map_backend", "osm")).strip().lower()
        return value if value in {"geo", "osm"} else "osm"
    except Exception:
        return "osm"

def point_map(frame):
    d=frame[['latitude','longitude']].apply(pd.to_numeric,errors='coerce').dropna().drop_duplicates()
    d=d[d.latitude.between(-90,90)&d.longitude.between(-180,180)]
    fig=go.Figure(go.Scattergeo(lat=d.latitude,lon=d.longitude,mode='markers',
        marker=dict(size=8,color='#1c78c0',opacity=.8,line=dict(color='white',width=.7)),
        hovertemplate='纬度 %{lat:.3f}°<br>经度 %{lon:.3f}°<extra></extra>'))
    if d.empty:return style_map(fig,True)
    latpad=max(1.,(d.latitude.max()-d.latitude.min())*.12)
    lonpad=max(1.,(d.longitude.max()-d.longitude.min())*.12)
    fig.update_geos(projection_type='equirectangular',
        lataxis_range=[max(-89,d.latitude.min()-latpad),min(89,d.latitude.max()+latpad)],
        lonaxis_range=[max(-180,d.longitude.min()-lonpad),min(180,d.longitude.max()+lonpad)])
    return style_map(fig)

def draw_map(fig):
    from globalhab_demo.display_locale import st
    with st.container(border=True):
        st.plotly_chart(fig,width='stretch',config={'displayModeBar':False,'responsive':True})

# Keep a single tile-map renderer for every map in the application.

def style_map(fig, global_view=False):
    # ``geo`` uses Plotly's built-in vector geography and therefore does not
    # request external map tiles. It is the safe fallback for networks where
    # OpenStreetMap tiles are slow or unavailable. Scientific values and source
    # coordinates remain unchanged in both modes.
    if _map_backend() == "geo":
        result = go.Figure(fig)
        result.update_geos(
            showland=True, landcolor="#edf3f7", showocean=True, oceancolor="#e8f5fd",
            showcountries=True, countrycolor="#ffffff", showcoastlines=True, coastlinecolor="#6f93aa",
            showframe=False, bgcolor="#f9fcff",
        )
        result.update_layout(
            height=480, margin=dict(l=0,r=0,t=50,b=25), paper_bgcolor="#f9fcff",
            font=dict(family="Microsoft YaHei, PingFang SC, sans-serif",size=12,color="#173f52"),
            legend=dict(orientation="h",x=.5,xanchor="center",y=1.12),
            hoverlabel=dict(bgcolor="#f9fcff",font_size=13),
        )
        return result

    import math
    traces=[];lats=[];lons=[]
    for tr in fig.data:
        if tr.type not in ('scattergeo','scattermap'):continue
        raw=tr.to_plotly_json()
        marker=raw.get('marker',{}).copy()
        for k in ['line','symbol','sizemin']:marker.pop(k,None)
        marker['symbol']='circle'
        valid={k:raw[k] for k in ['lat','lon','text','customdata','hovertemplate','name','showlegend','legendgroup','opacity','ids'] if k in raw}
        valid['mode']='markers';valid['marker']=marker
        traces.append(go.Scattermap(**valid))
        for lat,lon in zip(tr.lat,tr.lon):
            try:
                if np.isfinite(float(lat)) and np.isfinite(float(lon)):
                    lats.append(float(lat));lons.append(float(lon))
            except (ValueError,TypeError):pass
    center={'lat':15,'lon':0};zoom=0.5
    if lats and not global_view:
        center={'lat':float((min(lats)+max(lats))/2),'lon':float((min(lons)+max(lons))/2)}
        span=max(max(lons)-min(lons),(max(lats)-min(lats))*2,0.2)
        zoom=max(.5,min(9.,math.log2(360/span)-1.1))
    result=go.Figure(traces)
    result.update_layout(map=dict(style='open-street-map',center=center,zoom=zoom),height=480,
        margin=dict(l=0,r=0,t=50,b=25),paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Microsoft YaHei, PingFang SC, sans-serif',size=12,color='#173f52'),
        legend=dict(orientation='h',x=.5,xanchor='center',y=1.12),
        hoverlabel=dict(bgcolor='white',font_size=13))
    if fig.layout.coloraxis.to_plotly_json():result.update_layout(coloraxis=fig.layout.coloraxis.to_plotly_json())
    for tr in result.data:
        if tr.marker.colorscale is not None or tr.marker.showscale:
            tr.marker.colorbar.update(orientation='h',x=.5,xanchor='center',y=1.01,yanchor='bottom',len=.45,thickness=10)
    result.update_layout(coloraxis_colorbar=dict(orientation='h',x=.5,xanchor='center',y=1.01,yanchor='bottom',len=.45,thickness=10))
    return result
