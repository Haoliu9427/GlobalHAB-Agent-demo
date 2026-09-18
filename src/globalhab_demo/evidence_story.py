"""Presentation of saved, auditable experiment results; no model execution."""
import html
import json
from pathlib import Path
import pandas as pd
from .display_locale import st


def hero(root):
    root = Path(root)
    try:
        n = json.loads((root/'outputs/norway_forward_benchmark_card.json').read_text(encoding='utf-8'))
        d = json.loads((root/'outputs/discovery_card.json').read_text(encoding='utf-8'))
        best = d['best_candidate']
        model, ref = n['model_average_precision'], n['reference_average_precision']
        return f'''<section class="es-hero"><div class="es-eyebrow">GLOBALHAB-AGENT / 科研证据工作台</div>
<h1>从异常线索，到可检验的海洋风险证据</h1>
<p class="es-lead">筛选传播假设，比较候选模型，用独立观测检验风险排序。</p>
<div class="es-results"><article class="es-primary"><span class="es-tag">真实观测 · 挪威前向验证</span><h2>把有限监测资源，优先放到高风险样本</h2>
<div class="es-number">{n['top10_recall']:.0%}<small>事件覆盖 / 风险排序前10%样本</small></div>
<div class="es-track"><i style="width:{n['top10_recall']*100:.2f}%"></i></div>
<p>选出 {n['top10_selected']:,} / {n['samples']:,} 条样本，覆盖 {n['top10_true_positives']} / {n['events']} 个事件。</p>
<div class="es-foot">其中 {n['top10_false_positives']} 条为非事件样本；参考模型同预算覆盖 {n['reference_top10_recall']:.0%} 事件。</div></article>
<article><span class="es-tag">真实观测 · 排序表现</span><h2>整体 AP 高于参考模型</h2><div class="es-number">{model:.3f}<small>参考 {ref:.3f} → 当前 {model:.3f}</small></div>
<p>绝对增加 {model-ref:.3f} · 相对增加 {(model/ref-1):.1%}</p><div class="es-foot">四个前向检验窗口；AP不是准确率。<br>回顾性预测1–14天内的下一次观测，不是连续14天预报。</div></article>
<article><span class="es-tag es-synthetic">合成实验 · 方法验证</span><h2>恢复预设的顺流传播线索</h2><div class="es-number">{int(best['lag_days'])}<small>天 / 候选传播时滞</small></div><p>候选 AP {best['pr_auc']:.3f} · 简单基线 {d['minimum_references']['strongest_simple_baseline_pr_auc']:.3f}</p><div class="es-foot">通过合成真值恢复测试。<br>不代表已经发现真实海域传播因果。</div></article></div>
<div class="es-hero-bottom"><span>证据分层：真实观测 / 合成实验 / 演示沙盘</span><a href="?workspace=研究验证&section=真实数据训练与验证" target="_self">打开验证工作区 ↗</a></div></section>'''
    except (OSError, KeyError, ValueError):
        return '<section class="es-hero"><h1>科研证据工作台</h1><p>结果文件尚不完整，请先生成验证结果。</p></section>'


def render_trace(root):
    root = Path(root)
    with st.container(border=True, key='evidence_replay'):
        st.markdown('### Agent 做了什么 · 一次探索的证据链')
        st.caption('合成实验 / 已保存运行回放。展示可审计的假设、工具与反馈，不是实时推理或内部思维记录。')
        try:
            log = pd.read_csv(root/'outputs/agent_log.csv')
            router = pd.read_csv(root/'outputs/adaptive_router_trace.csv')
        except (OSError, ValueError):
            st.info('尚无探索日志。')
            return
        labels = {'local':'局地', 'downstream':'顺流', 'upstream':'逆流'}
        models = {'logistic':'逻辑回归', 'random_forest':'随机森林'}
        st.markdown('<div class="es-chain"><span>01 提出传播假设</span><b>→</b><span>02 路由分析工具</span><b>→</b><span>03 比较实验反馈</span><b>→</b><span>04 验证候选与边界</span></div>', unsafe_allow_html=True)
        step = st.select_slider('选择一次试验，查看它的依据与结果', options=log.step.astype(int).tolist(), value=int(log.loc[log.utility.idxmax(),'step']), key='evidence_trace_step')
        r = log[log.step.eq(step)].iloc[0]
        a,b,c = st.columns([1.25,1,1])
        with a:
            st.markdown('**待检验假设**')
            st.write(f"{labels.get(r.route,r.route)}路径的 {int(r.lag_days)} 天热异常信号与 HAB 风险相关。")
            st.caption(f"检验工具：{models.get(r.model,r.model)}；记录状态：{'候选' if r.status=='candidate' else '未保留'}。")
        with b:
            st.metric('本次 AP', f'{r.pr_auc:.3f}')
            st.caption(f'训练 {int(r.train_rows):,} 条 / 检验 {int(r.test_rows)} 条 / 事件 {int(r.test_events)} 个')
        with c:
            st.metric('候选效用分数', f'{r.utility:.3f}')
            st.caption(f'剩余试验预算 {int(r.budget_remaining)}；切分日期 {r.cut_date}')
        earlier = log[log.step.lt(step)]
        if len(earlier):
            prev = earlier.iloc[-1]
            changes = []
            for field, label in [('route','路径'),('lag_days','时滞'),('model','模型')]:
                if r[field] != prev[field]:
                    changes.append(f'{label}：{prev[field]} → {r[field]}')
            st.info(f"相邻试验记录：{'；'.join(changes) or '配置不变'}。AP {prev.pr_auc:.3f} → {r.pr_auc:.3f}。这是日志中的配置与反馈变化，不据此推断未记录的决策理由。")
        st.caption('验证边界：结果卡记录了时间留出与空间阻断；不同试验的有效检验行数存在变化，不能将相邻AP差值直接解释为单个因素的因果贡献。')
        with st.expander('工具选择依据与原始证据'):
            st.dataframe(router[['branch','selected','reason']].rename(columns={'branch':'分析工具','selected':'是否选用','reason':'记录中的选择依据'}), hide_index=True, width='stretch')
            st.dataframe(log[['step','route','lag_days','model','pr_auc','utility','status']], hide_index=True, width='stretch')
            st.download_button('下载本次探索日志 CSV', log.to_csv(index=False).encode('utf-8-sig'), 'agent_log.csv', 'text/csv', key='evidence_log_download')
            st.caption('来源：outputs/agent_log.csv、adaptive_router_trace.csv、discovery_card.json；首屏真实观测结果来自 norway_forward_benchmark_card.json。')
