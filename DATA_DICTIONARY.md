# 合成示例数据字典

本项目同时包含合成基准与两套真实观测回放。下表第一部分为固定随机种子生成的合成数据，仅用于验证程序能否恢复预设的“沿抽象传导路径、滞后14天”信号；南澳与挪威真实数据在后续部分单独说明，三者不得混为同一训练集。

| 字段 | 单位/类型 | 合成定义及用途 |
|---|---|---|
| `date` | 日 | 连续日尺度时间索引 |
| `region` | 类别 | `Synthetic_Region_A–D`，不对应真实海区 |
| `sst_c` | °C | 合成海表温度 |
| `climatological_mean_sst_c` | °C | 季节性SST气候平均值 |
| `climatological_p90_sst_c` | °C | 合成日历日90百分位阈值 |
| `is_mhw` | 0/1 | `sst_c > climatological_p90_sst_c` |
| `mhw_intensity_c` | °C | MHW日为`sst_c - climatological_mean_sst_c`，非MHW日为0 |
| `nitrate_mmol_m3` | mmol m⁻³ | 硝酸盐浓度 |
| `phosphate_mmol_m3` | mmol m⁻³ | 磷酸盐浓度 |
| `silicate_mmol_m3` | mmol m⁻³ | 硅酸盐浓度 |
| `microplastic_concentration_items_m3` | items m⁻³ | 合成微塑料浓度 |
| `circulation_residence_proxy` | 0–1 | `log(1+MPs)`的有界单调变换，仅代理输运、停留和汇聚背景 |
| `hab_event` | 0/1 | 合成事件标签 |
| `hidden_probability` | 0–1 | 仅用于检查合成器，不参与模型输入 |

## 微塑料变量的解释边界

微塑料浓度不被解释为流速、流向或HAB的直接驱动因素。本示例仅将其作为表层水体输运、停留和汇聚状态的可观测代理，并用于调节候选传导路径的权重。真实研究中仍需通过漂流浮标、再分析流场或独立环流资料验证该代理关系。

## 情景地图输出

网页地图使用北阿拉斯加湾、加州沿岸、墨西哥湾、北大西洋中部、日本黑潮延伸区、中国南部近岸和南澳大利亚近岸作为粗略演示锚点。`综合风险指数`与`预计藻华强度指数`均为0–100无量纲界面展示量，仅用于比较同一次假设情景下的空间相对差异。二者不等同于校准后的事件概率、藻细胞密度、叶绿素浓度或毒素浓度，也不能用于现场决策。

## 输出文件

- `baseline_results.csv`：季节气候态与事件持续性参照；
- `negative_controls.csv`：反向路径与区内时间置换；
- `agent_log.csv`：假设、动作、指标、状态、成本和剩余预算；
- `run_manifest.json`：版本、随机种子与配置/数据SHA-256；
- `aquaculture_response_priority.csv`：网页按用户选择生成的养殖响应优先结果。
- `multiscale_anomaly_daily.csv`：四个尺度的稳健z分数、综合分数、尺度一致数和事件标记；
- `multiscale_event_catalog.csv`：起止/峰值日期、持续时间和峰值分数；
- `adaptive_router_trace.csv`：六个门控值、兼容度、路由概率、run/defer与原因；
- `te_cte_network.csv`：边级TE/CTE bit、反向CTE、净方向性、置换p和FDR q；
- `spatial_durbin_effects.csv`：每1 SD的直接、间接和总关联影响及90%区间；
- `spatial_weight_matrix.csv`：目标行、来源列的匿名行标准化W矩阵；
- `method_diagnostics.json`：路由诊断、Durbin拟合和峰值时滞摘要。

## 养殖决策支持字段

| 字段 | 定义 |
|---|---|
| `藻华危害指数` | 0–100合成情景指数 |
| `养殖暴露度` | 0–1透明演示系数 |
| `脆弱性系数` | 养殖对象×危害机制演示系数 |
| `养殖响应优先指数` | 危害×暴露×脆弱性，0–100 |
| `证据置信度` | A/B/C证据只用于调整不确定性宽度 |
| `不确定性下限/上限` | 界面演示范围，不是统计预测区间 |

上述字段不估算死亡率、毒素浓度、停采决定或经济损失。

## 南澳大利亚真实qPCR数据

来源为Murray等人（2026）论文配套Zenodo记录20227730中的`Figure2_Final_qPCR_data_integrated.xlsx`，许可为CC BY 4.0。项目保留原始工作簿和数据说明，并生成`sa_qpcr_observations.csv`供离线网页运行。

| 字段 | 定义 |
|---|---|
| `sample_date` | 真实采样日期，2025-03-18至2025-09-13 |
| `location` | 原工作簿采样地点 |
| `latitude/longitude` | 原工作簿经纬度；南纬转换为负值 |
| `depth` | Surface、1m、DCM或4m integrated tube |
| `k_*_cells_l` | 7种Karenia的qPCR细胞丰度；`Not detected`数值列置0但另设状态字段 |
| `k_*_reported_not_detected` | 是否由原工作簿明确报告`Not detected` |
| `karenia_total_cells_l` | 同一样本7种Karenia数值和，仅用于描述 |
| `k_cristata_share` | K. cristata占同一样本Karenia数值和的比例 |
| `source_record/source_license` | Zenodo DOI与CC BY 4.0许可 |

真实数据共115条样本、22个采样日期和22个地点。`Not detected`只表示该次样本/检测结果，不能解释为该海区当天不存在藻华，也不能直接作为监督训练负标签。

## 挪威沿岸真实监测数据

| 字段 | 含义 |
|---|---|
| `sample_date` | 周尺度监测日期 |
| `region` | 挪威沿岸监测区域名称 |
| `a_tamarense_cells_l` | A. tamarense complex细胞丰度（cells L⁻¹） |
| `d_acuta_cells_l` | D. acuta细胞丰度（cells L⁻¹） |
| `sst_c` | 海表温度（°C） |
| `sea_surface_salinity_psu` | 海表盐度（PSU） |
| `mixed_layer_depth_m` | 混合层深度（m） |
| `par_e_m2_d` | 光合有效辐射（E m⁻² d⁻¹） |
| `a_tamarense_hab_event` | 原论文研究定义：A. tamarense complex >200 cells L⁻¹ |
| `d_acuta_hab_event` | 原论文研究定义：D. acuta >200 cells L⁻¹ |
| `target_hab_event` | 上述两类事件标识至少一个为真 |

完整数据包含5,919条观测、868个采样日期和35个沿岸区域，覆盖2006-04-03至2019-10-28。事件标识用于复现Silva等人（2025）的研究设置，不代表全球统一的养殖停采或人体健康阈值。

## 真实回放输出

- `sa_real_replay_timeline.csv`：按采样日汇总的样本数、地点数和丰度峰值；
- `sa_real_site_summary.csv`：按地点汇总的采样次数、K. cristata峰值和检出证据；
- `sa_real_species_summary.csv`：7种Karenia的采样集描述；
- `sa_real_router_trace.csv`：真实数据支持或暂缓各分析分支的原因；
- `sa_real_aquaculture_priority.csv`：观测丰度×情景暴露×对象脆弱性的现场复核顺序；
- `sa_real_replay_card.json`：来源、范围、峰值和解释边界。

## 网箱鱼生物响应沙盘输出

| 字段 | 含义 |
|---|---|
| `hour` | 沙盘模拟小时，默认0–72 h |
| `intervention` | 维持监测、降低投喂、启动增氧、转移准备或联合干预 |
| `relative_physiological_pressure` | 0–100相对生理压力状态；无量纲且未经物种标定 |
| `compound_challenge` | HAB、热异常、低氧、密度和交互项形成的0–100综合挑战 |
| `effective_do_mg_l` | 基线DO加情景增氧、减密度/摄食耗氧代理后的演示值 |
| `feeding_opportunity_pct` | 相对计划投喂的摄食机会代理，不是实际摄食量或生长率 |
| `response_readiness_hours` | 情景设定的响应准备时间，不代表行动已经执行 |
| `peak_pressure_lower/upper` | 综合挑战参数±15%时的峰值压力敏感性包络 |
| `pressure_load_reduction_vs_baseline_pct` | 相对“维持监测”的累计压力变化 |

- `cage_fish_response_trajectories.csv`：五项干预的逐小时状态；
- `cage_fish_intervention_comparison.csv`：峰值、累计压力、摄食机会、DO和准备时间对照；
- `cage_fish_sandbox_parameters.csv`：全部公开原型参数与解释边界；
- `cage_fish_sandbox_card.json`：输入、最低压力情景和明确排除的结论类型。

## Florida/Gulf真实流场回顾验证数据

该模块不把数据随包静态写成一个固定成绩，而是在运行时读取公开数据或接受用户上传。

### Karenia/HABSOS观测标准字段

| 统一字段 | 含义 |
|---|---|
| `date` | 采样日期，日尺度对齐 |
| `latitude/longitude` | 真实采样经纬度 |
| `cell_count` | Karenia/K. brevis细胞计数（cells L⁻¹） |
| `category` | HABSOS观测类别/描述，若有 |
| `salinity` | 现场盐度，若有 |
| `water_temp_c` | 现场水温，若有 |
| `sample_depth` | 采样深度，若有 |
| `genus/species` | 分类信息，用于识别Karenia brevis |

### 连续流场标准字段

| 统一字段 | 含义 |
|---|---|
| `date` | 流场日期，与观测日对齐 |
| `latitude/longitude` | 流场网格点 |
| `u_ms` | 东向流速（m s⁻¹） |
| `v_ms` | 北向流速（m s⁻¹） |

Florida/Gulf回顾分析默认在线读取HYCOM-TSIS GOMb0.04表层u/v流场；NOAA CoastWatch地转流保留为备选在线源。上传模式可识别`u_current/v_current`、`u/v`、`uo/vo`等常见字段。绝对流速超过5 m s⁻¹的记录会被质量门控剔除，以减少cm/s误作m/s等单位错误。

回顾验证按候选lag构建“先前源观测→后续目标观测”配对，并分别计算真实流向约束、反向流负对照和仅空间邻近对照。该结果是流场约束的关联证据，不等于完整Lagrangian粒子轨迹或因果证明。

## 未来出海/场站前向验证数据协议

最低现场观测字段见`data/field_validation/field_observations_template.csv`：

`date, station_id, latitude, longitude, cell_count`

最低流场字段见`data/field_validation/field_currents_template.csv`：

`date, latitude, longitude, u_ms, v_ms`

建议同步保留但当前不强制参与事件标签的字段：

- `toxin_value/toxin_unit`：毒素检测；
- `water_temp_c/salinity`：水温与盐度；
- `dissolved_oxygen_mg_l`：溶解氧；
- `nitrate_mmol_m3/phosphate_mmol_m3/silicate_mmol_m3`：营养盐；
- `chlorophyll`：叶绿素/荧光代理；
- 后续可扩展鱼体、鳃部、网箱密度与设备运行信息。

严格前向模式先执行质量门控；较早时间块只用于选择lag，较晚时间块只评估一次。输出包括训练期lag表、前向测试流向/反向/无流向指标以及下一批采样候选位置。数据不足时返回`defer`而不是填补一个结果。

## Agent实验设计策略输出

- `agent_policy_benchmark_default.csv`：同一候选景观、同一预算下各实验选择策略的恢复率、首次触达隐藏模式步数和最佳效用；
- `agent_policy_trajectories_default.csv`：策略代表性逐步动作轨迹。

贝叶斯策略只观察已执行动作的反馈；合成隐藏真值只在轨迹完成后评分，不进入获取函数。
