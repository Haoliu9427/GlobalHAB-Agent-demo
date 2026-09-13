"""Generate a concise report from completed artifacts; no fitted-model selection."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'outputs/real_training'
    lines=['# 本次真实数据实验结果','',
        '以下数值读取自完整实验输出，AP不是分类准确率。不同任务的事件定义、阳性比例和留出区间不同，不能用香港AP与原挪威0.079直接比较改进幅度。','',
        '## 数据与评估','',
        'NOAA原始快照223,394条；香港原始档案1,931行、1,757个去重报告编号。原始记录数不等于训练样本数。三个训练随机种子为17、42、73，全部保留；没有择优挑选种子。','']
    for name,title in [('habsos_7d','HABSOS：7–10天后的同网格复测'),('habsos_14d','HABSOS：14–17天后的同网格复测'),
        ('habsos_30d','HABSOS：30–33天后的同网格复测'),('china_hk_foundation_v2','中国香港：下一周赤潮报告')]:
        folder=base/name
        if not (folder/'metrics.csv').exists():continue
        m=json.loads((folder/'split_manifest.json').read_text());d=pd.read_csv(folder/'metrics.csv')
        lines+=['## '+title,'',f"训练/选模/校准/时间测试样本：{m['counts']['train']:,} / {m['counts']['validation']:,} / {m['counts']['calibration']:,} / {m['counts']['test']:,}。分界日期：{'、'.join(m['cutoffs'])}。",'']
        for part,g in d.groupby('split'):
            lines+=[('时间留出' if part=='test' else '未见0.1°网格的未来样本')+f"：{int(g.iloc[0]['n']):,}条，阳性{int(g.iloc[0].positives):,}条（{g.iloc[0].prevalence:.2%}）。",'',
                '| 模型 | AP均值±种子标准差 | Brier | ECE | 平均秒数 |','|---|---:|---:|---:|---:|']
            for model,v in g.groupby('model'):
                lines.append(f'| {model} | {v.AP.mean():.3f} ± {v.AP.std():.3f} | {v.Brier.mean():.4f} | {v.ECE.mean():.4f} | {v.seconds.mean():.2f} |')
            lines+=['']
        audit=folder/'comparative_audit.csv'
        if audit.exists():
            a=pd.read_csv(audit);lines+=['三种子季度块置信区间检查（时间测试，EcoTemporalNet）：','']
            for ref,g in a[(a.model=='EcoTemporalNet')&(a.split=='test')].groupby('reference'):
                conclusion='三个种子下限均大于0' if (g.delta_ap_low>0).all() else '至少一个种子的区间跨0或不支持优越'
                lines.append(f'* 相对{ref}：{conclusion}。')
            lines+=['']
        if name=='china_hk_foundation_v2':
            weights=pd.read_csv(folder/'training_selection.csv')
            lines+=['Chronos融合权重（按种子顺序）：'+', '.join(map(str,weights.chronos_weight.tolist()))+'。验证期选到零权重时，说明该配方下未采用基础模型，不能宣称基础模型带来了增益。','']
        warning=folder/'warning_threshold_results.csv'
        if warning.exists():
            w=pd.read_csv(warning);lines+=['固定校准阈值的预警结果（三种子均值；测试期不调阈值）：','',
                '| 模型 | 报警中真实事件比例 | 事件召回率 | 测试期报警比例 |','|---|---:|---:|---:|']
            for model,g in w[w['split']=='test'].groupby('model'):
                lines.append(f'| {model} | {g.precision.mean():.1%} | {g.recall.mean():.1%} | {g.alert_fraction.mean():.1%} |')
            lines+=['','报警预算在校准期设为10%；未来比例会变化。这些是观测事件指标，不是降低经济损失的实测比例。','']
    llm=base/'china_hk_llm';status=llm/'status.json'
    if status.exists():
        meta=json.loads(status.read_text());lines+=['## 语言模型对照','',f"模型：{meta.get('model','')}；状态：{meta['status']}。",'']
        if meta['status']=='completed':
            d=pd.read_csv(llm/'metrics.csv');lines+=['| 模型 | AP | Brier | ECE |','|---|---:|---:|---:|']
            for model,g in d.groupby('model'):lines.append(f'| {model} | {g.AP.mean():.3f} | {g.Brier.mean():.4f} | {g.ECE.mean():.4f} |')
            lines+=['',f"完整评分耗时{meta['seconds']:.1f}秒，含初始化/可能的权重下载；固定语言模型没有三个独立训练种子，重复项对应三个下游融合模型。",'',
                '公开语料重叠无法排除；一个小型Qwen固定提示词实验不代表所有大模型。','']
    lines+=['## 如何使用这些结果','',
        '短窗口的平均AP提升值得继续验证，但不能据此宣传所有区域、所有窗口均已稳定胜出。较长窗口、校准误差、季度块置信区间和逐年结果一起保留。传统模型未做穷尽调参，此比较针对工程中公开的固定配方。', '',
        '中国内地海域仍需取得连续、带可靠阴性标签与真实可用时间的监测数据后验证。香港报告任务提供中国水域的实测证据，但不替代内地各海域、不同物种和不同预警阈值验证。', '',
        '原24候选、8步机制约束合成探索保持独立；新增真实实验没有改变原合成科学结论。']
    (ROOT/'REAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
