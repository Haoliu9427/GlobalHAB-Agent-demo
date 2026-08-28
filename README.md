# GlobalHAB-Agent · GOAI复赛可运行环境

GlobalHAB-Agent 将复杂海洋场景转化为可计算、可验证、可复现的开放探索任务：Agent 在固定预算内比较局地/沿流路径、3–45天响应时滞与轻量模型，使用前向时间+完全留一海区验证，根据反馈继续、修正或停止假设，并保留正结果、负结果、参照与发现卡。

使用南澳大利亚2025复杂Karenia藻华的真实事件回放。其直接读取论文配套Zenodo qPCR数据，展示115条样本、22个采样日期、22个地点、7种Karenia丰度、空间热点和养殖复核优先级，并由数据路由器明确区分可运行分析与暂缓分析。

> 重要边界：PR-AUC等预测性能来自匿名合成数据。南澳大利亚qPCR数据用于真实事件回放，不用于监督训练；“Not detected”不被解释为完整生态负样本。

## 在线页面能展示什么

- 什么条件：MHW强度、Nitrate、Phosphate、Silicate、输运/停留/汇聚代理；
- 什么时间：7、14、30天候选窗口；
- 什么地方：7个演示性候选海区；
- 多强：0–100无量纲HAB风险与相对强度；
- 哪类养殖先核查：根据危害、暴露、脆弱性和证据置信度形成响应优先级；
- 为什么相信：季节气候态、事件持续性、随机探索、反向路径和时间置换参照；
- 机制信号：多尺度事件、信息流的方向性、14天平均峰值与FDR结果；
- 影响如何传播：Durbin直接、间接和总影响及90%块Bootstrap区间；
- 如何复核：完整探索日志、发现卡、运行清单、固定随机种子和自动测试。
- 真实事件：南澳qPCR采样地图、时间线、物种组成、真实数据路由与养殖复核顺序。

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
- 'outputs/sa_real_replay_card.json'：机器可检查的真实事件卡。

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

网页包含五个工作区：

1. 藻华与养殖风险：情景地图、养殖对象、危害机制、证据等级和响应优先级；
2. 南澳真实事件回放：真实qPCR地图、时间线、物种组成、数据充分性和养殖复核；
3. 机制模块与影响分解：多尺度异常、自适应路由、TE/CTE网络和Durbin乘数；
4. Agent探索与研究信号：基线、随机参照、负对照、完整探索轨迹和风险序列；
5. 证据链与复现：来源、许可证、变量边界、发现卡与证据包下载。

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
- 多尺度异常：所有滚动参考只用当日以前数据，以MAD稳健标准化并要求至少两个尺度一致；
- TE/CTE：离散条件互信息以bit计量，圆周移位置换保留源序列自相关，边级检验执行BH-FDR；
- Durbin：W为匿名行标准化有向图，14天空间暴露尺度由Agent/TE-CTE结果固定，输出为关联尺度而非因果效应。

## 目录

    globalhab_agent_v32/
    ├── app.py
    ├── run_demo.py
    ├── config/demo.json
    ├── data/
    │   └── real_case/
    │       ├── raw/              # Zenodo原始qPCR工作簿与说明
    │       └── derived/          # 清洗后CSV和来源清单
    ├── outputs/
    ├── src/globalhab_demo/
    │   ├── agent.py
    │   ├── data.py
    │   ├── experiment.py
    │   ├── workflow.py
    │   ├── scenario.py
    │   ├── aquaculture.py
    │   ├── evidence.py
    │   ├── multiscale.py
    │   ├── router.py
    │   ├── transfer_entropy.py
    │   ├── spatial_durbin.py
    │   └── real_replay.py
    ├── scripts/prepare_sa_real_replay.py
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
- Ruvindy, R. et al. Environmental Science & Technology (2024). https://doi.org/10.1021/acs.est.3c10502

## 开源

本竞赛环境采用 MIT License。四个模块的实现已公开。

真实qPCR工作簿及其派生表沿用Zenodo记录的CC BY 4.0许可，引用Murray等人（2026）；MIT许可不覆盖第三方数据。完整归属和转换说明见 'THIRD_PARTY_DATA.md'。
