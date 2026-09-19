"""Responsive research-mode cards using native accessible Streamlit buttons."""
from .display_locale import st

MODES = ["模型驱动 · 智能研究", "自主接入 · 智能研究", "规则驱动 · 科学检验"]
ICONS = ["account_tree", "link", "fact_check"]
DESCRIPTIONS = ["选择已配置的大模型，由模型规划、工具检验。", "连接你自己的模型服务，沿用同一套科学工具。", "按明确规则组织实验，对照与验证过程可复核。"]
PATHS = [
    '<rect x="8" y="2" width="8" height="6" rx="2"/><path d="M12 8v5M5 13h14M5 13v3M19 13v3"/><rect x="2" y="16" width="6" height="6" rx="2"/><rect x="16" y="16" width="6" height="6" rx="2"/>',
    '<path d="m10 14 4-4M9 7l2-2a5 5 0 0 1 7 7l-2 2M15 17l-2 2a5 5 0 0 1-7-7l2-2"/>',
    '<rect x="4" y="3" width="16" height="19" rx="3"/><path d="m7 9 2 2 3-4M14 9h3m-10 7 2 2 3-4m2 2h3"/>'
]

def select_mode():
    prior = st.session_state.get("research_controller_mode", MODES[0])
    if prior not in MODES:
        prior = MODES[2] if "规则" in prior else MODES[1] if "API" in prior or "自选" in prior else MODES[0]
    st.session_state["research_controller_mode"] = prior
    st.markdown("""<style>
    .st-key-research_modes [data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important;}
    .research-mode-icon{width:42px;height:42px;border-radius:12px;background:#e7f4f8;color:#0d8398;display:flex;align-items:center;justify-content:center;margin-bottom:10px}
    .research-mode-icon svg{width:24px;height:24px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
    .st-key-research_modes [data-testid="stCaptionContainer"]{min-height:42px;color:#53738a;}
    .st-key-research_modes button{min-height:44px!important;}
    .stApp div[data-testid="stMainBlockContainer"] .st-key-research_modes button[kind="primary"] [data-testid="stMarkdownContainer"] p{color:#fff!important;}
    .research-note{padding:16px 20px;border-left:3px solid #1699a5;background:#f0f8fb;border-radius:0 12px 12px 0;color:#42647a;font-size:14px;line-height:1.8;margin:12px 0 24px}
    @media(max-width:700px){.st-key-research_modes [data-testid="stHorizontalBlock"]{flex-direction:column!important;gap:8px!important}.st-key-research_modes [data-testid="stColumn"]{width:100%!important;flex:1 1 100%!important;min-width:0!important}.st-key-research_modes [data-testid="stCaptionContainer"]{min-height:0}.research-mode-icon{float:left;margin:0 12px 8px 0;width:36px;height:36px}}
    </style>""", unsafe_allow_html=True)
    with st.container(key="research_modes"):
        for i, col in enumerate(st.columns(3, gap="small")):
            with col, st.container(border=True):
                st.markdown('<div class="research-mode-icon" aria-hidden="true"><svg viewBox="0 0 24 24">'+PATHS[i]+'</svg></div>', unsafe_allow_html=True)
                st.caption(DESCRIPTIONS[i])
                if st.button(MODES[i], icon=":material/"+ICONS[i]+":", key="research_mode_"+str(i), type="primary" if prior == MODES[i] else "secondary", width="stretch"):
                    st.session_state["research_controller_mode"] = MODES[i]
                    st.rerun()
    return st.session_state["research_controller_mode"]

UI_REVISION = 'HF3.9.11-CAMERA-INPUT-20260920'
