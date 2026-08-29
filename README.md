# GlobalHAB-Agent · GOAI复赛可运行环境

GlobalHAB-Agent 将复杂海洋场景转化为可计算、可验证、可复现的开放探索任务：Agent 在固定预算内比较局地/沿流路径、3–45天响应时滞与轻量模型，使用前向时间+完全留一海区验证，根据反馈继续、修正或停止假设，并保留正结果、负结果、参照与发现卡。

四个竞赛等价研究模块对应“风险研判—真实事件回放—生物响应沙盘—科学解释—探索验证—成果证据”六个工作区。真实模块包含南澳大利亚2025复杂Karenia事件的115条qPCR样本，以及挪威沿岸2006–2019年的5,919条有毒藻与环境监测记录。v3.5进一步加入网箱鱼生物响应沙盘，将HAB、高温、溶解氧、养殖密度与投喂情景连接到相对生理压力，并比较降低投喂、增氧和转移准备等干预路径。

> 重要边界：PR-AUC等预测性能只来自合成基准。南澳大利亚和挪威开放观测用于真实事件回放，不与合成数据混合训练。生物响应沙盘使用公开但未经鱼种/场站标定的原型参数，不输出死亡率、生物量损失、真实毒素或自动运营指令。

## 在线页面能展示什么

- 什么条件：MHW强度、Nitrate、Phosphate、Silicate、输运/停留/汇聚代理；
- 什么时间：7、14、30天候选窗口；
- 什么地方：7个演示性候选海区；
- 多强：0–100无量纲HAB风险与相对强度；
- 哪类养殖先核查：根据危害、暴露、脆弱性和证据置信度形成响应优先级；
- 为什么相信：季节气候态、事件持续性、随机探索、反向路径和时间置换参照；
- 机制信号：多尺度事件、TE/CTE方向性、14天平均峰值与FDR结果；
- 影响如何传播：Durbin直接、间接和总影响及90%块Bootstrap区间；
- 如何复核：完整探索日志、发现卡、运行清单、固定随机种子和自动测试。
- 真实观测：南澳qPCR空间回放，以及挪威14年有毒藻、SST、盐度、混合层深度和光照监测回放；
- 真实事件风险转译：两个回放均逐项显示已观测证据、情景假设、参数设定和待补数据，并输出现场复核/加密监测优先级；
- 网箱鱼生物响应：模拟HAB、高温、DO、密度和投喂的复合压力，比较五项干预的48/72/96小时轨迹；
- 干预权衡：同时展示压力缓解、摄食机会、有效DO和准备响应时间，不用单一“最优方案”替代现场判断；
- 全球证据：南澳、挪威、美国Salish Sea和全球HAEDAT/OBIS研究资源的统一证据地图。

## 30秒命令行试跑

    python -m pip install -r requirements.txt
    python run_demo.py --config config/demo.json

生成文件：

- 'outputs/agent_log.csv'：逐步假设、动作、反馈、成本与预算；
- 'outputs/baseline_results.csv'：季节气候态和事件持续性基线；
- 'outputs/negative_controls.csv'：反向路径和时间置换负对照；
- 'outputs/risk_predictions.csv'：留出海区风险与Top20%固定容量报警；
- 'outputs/discovery_card.json'：最佳信号、参照、验证与适用边界；
- 'outputs/run_manifest.json'：版本、随机种子及配置/数据SHA-256；
- 'outputs/run_summary.md'：试跑摘要。
- 'outputs/multiscale_anomaly_daily.csv'：逐日、逐尺度稳健异常分数；
- 'outputs/multiscale_event_catalog.csv'：合并后的异常事件目录；
- 'outputs/adaptive_router_trace.csv'：门控诊断、兼容度、决策与原因；
- 'outputs/te_cte_network.csv'：边级TE/CTE、反向路径、置换p值与FDR；
- 'outputs/te_cte_lag_summary.csv'：跨边时滞摘要；
- 'outputs/spatial_durbin_effects.csv'：直接、间接和总影响；
- 'outputs/spatial_weight_matrix.csv'：匿名有向空间权重矩阵；
- 'outputs/method_diagnostics.json'：路由与Durbin诊断。
- 'outputs/sa_real_replay_timeline.csv'：真实qPCR采样时间线；
- 'outputs/sa_real_site_summary.csv'：真实地点峰值与检出证据；
- 'outputs/sa_real_species_summary.csv'：7种Karenia采样集组成；
- 'outputs/sa_real_router_trace.csv'：真实数据条件下run/defer及原因；
- 'outputs/sa_real_aquaculture_priority.csv'：基于观测丰度的养殖复核优先级；
- 'outputs/sa_real_risk_evidence_matrix.csv'：南澳回放的风险研判证据属性与数据缺口；
- 'outputs/sa_real_replay_card.json'：机器可检查的真实事件卡。
- 'outputs/norway_real_replay_timeline.csv'：挪威沿岸真实监测时间线；
- 'outputs/norway_real_station_summary.csv'：35个沿岸区域的观测与事件摘要；
- 'outputs/norway_real_taxa_summary.csv'：A. tamarense complex与D. acuta摘要；
- 'outputs/norway_real_aquaculture_priority.csv'：挪威监测区域的加密监测优先级；
- 'outputs/norway_real_risk_evidence_matrix.csv'：挪威回放的风险研判证据属性与数据缺口；
- 'outputs/cage_fish_response_trajectories.csv'：五项干预的逐小时相对生理压力轨迹；
- 'outputs/cage_fish_intervention_comparison.csv'：峰值压力、敏感性包络、摄食机会和准备时间；
- 'outputs/cage_fish_sandbox_parameters.csv'：全部公开原型参数及解释边界；
- 'outputs/cage_fish_sandbox_card.json'：沙盘输入、最低压力情景和排除性声明；
- 'outputs/norway_real_replay_card.json'：挪威真实观测回放卡；
- 'outputs/global_nature_evidence_cases.csv'：Nature Portfolio全球证据接口清单。

默认合成验证结果：

| 证据 | 结果 |
|---|---:|
| 候选实验数 / Agent预算 | 24 / 8 |
| 恢复的隐藏信号 | 沿流路径，14天 |
| PR-AUC | 0.624 |
| Brier Skill | 0.383 |
| ECE | 0.039 |
| Top20%风险召回 | 76.9% |
| Top20%虚警率（FP/全部负例） | 10.6% |
| 相同预算随机搜索恢复率（200次） | 55.0% |
| 多尺度合并事件数 | 41 |
| 跨边平均CTE峰值 | 14天 |
| 自适应路由运行分支 | 4 / 4 |

这些数字验证环境能否恢复已知合成真值，不代表现实海区性能。

## 启动网页

    streamlit run app.py

网页包含六个工作区：

1. 风险研判：情景地图、养殖对象、危害机制、证据等级和响应优先级；
2. 真实事件回放：全球证据地图、南澳qPCR回放和挪威14年监测回放；
3. 生物响应沙盘：网箱鱼复合压力、五项干预轨迹、敏感性和运营权衡；
4. 科学解释：持续异常、方法选择、跨区域传播和邻区溢出；
5. 探索与验证：基线、随机参照、负对照、完整探索轨迹和风险序列；
6. 成果与证据：开放来源、成果转化信息体系、发现卡与证据包下载。

Streamlit Community Cloud 部署见 'DEPLOY_STREAMLIT.md'。

## 关键科学约束

- MHW日：'SST > climatological p90'；
- MHW强度：仅在MHW日计算 'SST - climatological mean'；
- 营养盐：Nitrate、Phosphate、Silicate分项保留；
- 微塑料：仅经有界变换代理输运/停留/汇聚状态，不是流速、流向或HAB直接生物驱动；
- 验证：完整留出一个区域，并只用时间截止点之前的数据训练；
- 报警：使用Top20%固定容量排名，不用留出标签反向调阈值；
- 发现：必须优于平凡解，并接受随机搜索和负对照检查；
- 养殖风险：只输出响应优先级；真实决策必须接入物种、毒素、溶解氧、现场生物反应和当地规则。
- 生物响应：0–100压力状态、有效DO和摄食机会均为透明过程代理；展示分档不是死亡、福利或监管阈值。
- 干预对照：转移准备在未执行前不得降低生理压力；降低投喂和增氧均保留机会成本或设备能力边界。
- 多尺度异常：所有滚动参考只用当日以前数据，以MAD稳健标准化并要求至少两个尺度一致；
- TE/CTE：离散条件互信息以bit计量，圆周移位置换保留源序列自相关，边级检验执行BH-FDR；
- Durbin：W为匿名行标准化有向图，14天空间暴露尺度由Agent/TE-CTE结果固定，输出为关联尺度而非因果效应。

## 目录

    globalhab_agent_v35/
    ├── app.py
    ├── run_demo.py
    ├── config/demo.json
    ├── data/
    │   ├── real_case/             # 南澳qPCR原始与派生数据
    │   └── real_case_norway/      # 挪威监测原始与派生数据
    ├── outputs/
    ├── src/globalhab_demo/
    │   ├── agent.py
    │   ├── data.py
    │   ├── experiment.py
    │   ├── workflow.py
    │   ├── scenario.py
    │   ├── aquaculture.py
    │   ├── bio_response.py
    │   ├── evidence.py
    │   ├── multiscale.py
    │   ├── router.py
    │   ├── transfer_entropy.py
    │   ├── spatial_durbin.py
    │   ├── real_replay.py
    │   └── global_cases.py
    ├── scripts/prepare_sa_real_replay.py
    ├── scripts/prepare_norway_replay.py
    ├── tests/test_smoke.py
    ├── docs/
    │   ├── GOAI_SEMIFINAL_CHECKLIST.md
    │   └── TECHNICAL_NOTE.md
    ├── MODEL_CARD.md
    ├── DATA_DICTIONARY.md
    ├── OPEN_SOURCE_BOUNDARY.md
    ├── THIRD_PARTY_DATA.md
    ├── DEPLOY_STREAMLIT.md
    └── LICENSE

## Docker

    docker build -t globalhab-agent-semifinal .
    docker run --rm -p 8501:8501 globalhab-agent-semifinal

打开 'http://localhost:8501'。

## 外部事件依据

- Murray, S. A. et al. Nature Ecology & Evolution (2026). https://doi.org/10.1038/s41559-026-03115-0
- 配套开放数据：https://doi.org/10.5281/zenodo.20227730
- Silva, E. et al. Communications Earth & Environment (2025). https://doi.org/10.1038/s43247-025-02421-y
- 挪威开放数据与模型：https://doi.org/10.5281/zenodo.10958487
- Føre et al. Digital Twins in intensive aquaculture — Challenges, opportunities and future prospects. Computers and Electronics in Agriculture (2024). https://doi.org/10.1016/j.compag.2024.108676
- Lima et al. Digital twins for land-based aquaculture: A case study for rainbow trout. Open Research Europe (2023). https://open-research-europe.ec.europa.eu/articles/2-16
- Ruvindy, R. et al. Environmental Science & Technology (2024). https://doi.org/10.1021/acs.est.3c10502

## 开源

本竞赛环境采用 MIT License。四个模块的竞赛等价实现已公开。

南澳qPCR工作簿、挪威监测表及其派生文件沿用各自Zenodo记录的CC BY 4.0许可；MIT许可不覆盖第三方数据。完整归属和转换说明见 'THIRD_PARTY_DATA.md'。
