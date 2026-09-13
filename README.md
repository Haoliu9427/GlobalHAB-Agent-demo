# GlobalHAB-Agent

本工程新增真实观测训练与验证模块，保留原有六个工作区和24候选、8步合成实验。新增内容、实测结果、命令和适用范围见 **[真实数据训练与验证说明](REAL_DATA_GUIDE.md)**；一览表见 **[实验结果汇总](REAL_RESULTS.md)**。已包含下载的数据快照、训练权重、逐样本预测和测试记录，无需重新训练即可查看结果。请作为本地独立工程使用。

GlobalHAB-Agent 是一个面向跨区域有害藻华（Harmful Algal Bloom, HAB）研究的可运行原型。系统将环境条件、输运信息、生物观测和水产养殖响应放在同一分析流程中，用于情景分析、传播时滞检验、模型比较和严格前向验证。

## 1. 主要功能

Streamlit 界面包含七个工作区：

1. **风险研判**：对代表性海区进行 7 / 14 / 30 天情景推演和相对风险排序。
2. **真实事件回放**：包括南澳 qPCR 事件回放、挪威长期监测前向验证、Florida/Gulf 真实流场约束回顾分析，以及现场数据前向验证。
3. **生物响应沙盘**：比较 HAB、高温、低溶解氧、养殖密度和投喂情景下的网箱鱼相对压力变化。
4. **科学分析**：包括多尺度异常检测、传播路径诊断、TE/CTE 时滞分析和空间 Durbin 效应分解。
5. **探索与验证**：包括有限预算实验选择、Bayesian 策略比较、负对照、模型 Benchmark 和严格留出验证。
6. **数据与溯源**：展示数据质量、数据来源、结果边界和可下载证据文件。
7. **真实数据训练与验证**：NOAA长期观测、中国香港报告序列、轻量时序网络、基础模型及语言模型对照、时间/网格留出、逐年稳定性和输入敏感性。

## 2. 数据结构

### 2.1 机制约束型合成基准

默认合成基准包含 4 个匿名区域和日尺度序列，主要变量包括：

- 海表温度（SST）及季节气候态；
- 海洋热浪状态与强度；
- 硝酸盐、磷酸盐和硅酸盐；
- 有界的输运 / 停留 / 汇聚代理；
- HAB 事件标签。

默认生成器预先设置一个 **14 天的上游—下游统计信号**。该真值在实验选择过程中不可见，只在完整探索轨迹结束后用于评价是否恢复预设结构。

### 2.2 真实观测数据

真实数据模块与合成基准分开使用：

- **South Australia**：115 条 qPCR 观测，用于真实事件回放；
- **Norway**：5,919 条 2006–2019 年有害藻及环境监测记录，用于严格前向排序验证；
- **Florida/Gulf**：NOAA HABSOS `Karenia brevis` 观测，默认结合 HYCOM GOMb0.04 Gulf reanalysis 流场；同时保留 NOAA CoastWatch 和上传 Copernicus / HF-radar 流场的入口；
- **现场前向验证**：接受用户上传的连续站点观测和流场数据。

第三方数据来源、许可和引用方式见 `THIRD_PARTY_DATA.md`。


## 2.3 结果状态分层

系统中的数字按科学角色分为四类，解释时不能混用：

| 状态 | 典型页面/结果 | 变化方式 | 正确解释 |
|---|---|---|---|
| 即时情景结果 | 风险研判、生物响应 | 改变控件即重算 | 当前情景，不是实时海洋观测 |
| 当前合成运行 | 探索与验证、科学解释 | 修改序列长度、seed、留出区、预算或前向比例后显式重算 | 当前有效配置下的运行结果 |
| 当前真实回放 | South Australia / Norway 回放 KPI | 随日期、区域、深度等筛选变化 | 当前回放窗口的观测摘要 |
| 注册固定审计 | Norway 严格前向基准、TCN 5-seed 容量审计 | 不随交互筛选变化 | 固定协议用于跨时间、跨设置审计 |

固定审计并不是“写死结果”。它保持固定，是为了避免当前界面筛选反向改变测试窗、模型选择和评价口径。

## 2.4 Global 与对象适用边界

- 风险研判使用 **12 个代表性风险情景锚点**；
- 生物响应页使用 **13 个全球生产背景区**；
- 其中只有 **7 个具有海水网箱养殖背景** 的区域可以进入网箱鱼沙盘；
- 其余区域保留为捕捞、贝类或其他生产背景，不强行套用网箱鱼生理模型。

这里的 `Global` 表示跨海区问题框架、代表性生产背景和统一数据接口，不表示已经完成全球真实业务验证。


## 3. 核心实验空间

核心实验空间定义为：

```text
route ∈ {local, downstream}
lag   ∈ {3, 7, 14, 21, 30, 45} days
model ∈ {logistic, random_forest}
```

共 24 个候选实验，默认实验预算为 8 步。

实验选择策略可以比较：

- 当前受约束策略；
- Bayesian Expected Improvement；
- Bayesian Information Gain 代理；
- Thompson Sampling；
- Random。

所有策略使用同一候选表、同一预算和同一已观察反馈。合成基准中的 14 天真值不进入 Bayesian 采集函数或其他动作选择规则。

## 4. 模型 Benchmark

完整模型 Benchmark 在**相同的外层留出样本**上比较统计方法、经典机器学习、Boosting 和轻量深度模型，包括：

- Seasonal / Persistence baselines；
- Logistic Regression；
- GAM；
- Gaussian Naive Bayes；
- kNN；
- RBF-SVM；
- Decision Tree；
- Random Forest；
- Extra Trees；
- AdaBoost；
- Gradient Boosting；
- HistGradientBoosting；
- XGBoost；
- LightGBM；
- MLP；
- STS-Interaction GLM；
- Lightweight TCN；
- STS-Gated TCN。

需要模型选择的方法只允许在外层训练时段内部调参，最终留出集不参与参数选择。


## 4.1 三组时间尺度

系统中存在三组用途不同的时间尺度：

- **科学候选 lag**：3 / 7 / 14 / 21 / 30 / 45 天，用于 local / downstream 竞争假设；
- **风险情景窗口**：7 / 14 / 30 天，用于相对情景研判；
- **多尺度异常窗口**：7 / 14 / 30 / 60 天，用于异常持续性判断。

三组时间尺度不能互相替代。尤其是默认合成真值中的 14 天，不等同于真实海洋已经证明存在 14 天传播规律。


## 5. 验证设计

合成基准的核心验证包括：

- 完全留出区域；
- 前向时间测试块；
- Seasonal Climatology 和 Event Persistence 基线；
- 同预算 Random 搜索；
- Reverse-path 和时间置换负对照；
- Average Precision、Brier Skill、ECE 和固定容量 Top-k 指标。

挪威数据采用扩展式前向窗口。Florida/Gulf 模块比较：

```text
真实流场约束
vs.
无流向空间匹配
vs.
反向流对照
```

并在候选 lag 上进行回顾性比较。

## 6. 现场前向验证

数据模板位于：

```text
data/field_validation/
```

观测数据最低字段：

```text
date, station_id, latitude, longitude, cell_count
```

流场数据最低字段：

```text
date, latitude, longitude, u_ms, v_ms
```

可选字段包括：

```text
toxin_value
water_temp_c
salinity
dissolved_oxygen_mg_l
nitrate_mmol_m3
phosphate_mmol_m3
silicate_mmol_m3
chlorophyll
```

系统首先检查样本量、时间连续性、空间支持、事件数和流场日期覆盖。满足条件后，较早时间块用于选择传播 lag，后续时间块仅用于一次独立前向评价；数据不足时返回 `DEFER`。

## 7. 生物响应模型

网箱鱼沙盘使用有界的相对压力状态：

```text
P(t+1) = clip[P(t) + 1.45*C(t)*(1-P(t)/100)
              - 0.55*(1-C(t))*P(t)/100, 0, 100]
```

其中：

- `C(t)`：0–1 的综合环境挑战项；
- `P(t)`：0–100 的相对生理压力状态。

参数公开记录在：

```text
outputs/cage_fish_sandbox_parameters.csv
```

该模块用于情景相对比较，不代表特定物种、生命阶段或养殖场的校准死亡率模型。

## 8. 快速开始

### 8.1 安装依赖

推荐 Python 3.12。

```bash
python -m pip install -r requirements.txt
```

### 8.2 启动网页

```bash
streamlit run app.py
```

### 8.3 默认命令行流程

```bash
python run_demo.py --config config/demo.json
```

### 8.4 最小离线复现

```bash
python scripts/run_minimal_reproduction.py
```

该命令用于快速检查：

- 24 个候选实验；
- 8 步预算；
- 完全留出区域与前向时间阻断；
- 默认配置下对预注册 `downstream / 14-day` 合成结构的恢复；
- 同预算 Random、平凡基线和负对照。

### 8.5 Agent 策略比较

```bash
python scripts/run_agent_policy_benchmark.py
```

### 8.6 完整模型 Benchmark

```bash
python scripts/run_broad_benchmark_audit.py
```

### 8.7 Florida/Gulf 回顾分析

在线模式：

```bash
python scripts/run_florida_sts_validation.py --online
```

在线模块依赖 NOAA / HYCOM 服务状态。公开服务不可用时，可改用导出的 HABSOS 和 HYCOM / Copernicus / HF-radar CSV。

### 8.8 现场前向验证

```bash
python scripts/run_field_forward_validation.py \
  --observations <field_observations.csv> \
  --currents <field_currents.csv>
```


### 8.9 一键 smoke test

```bash
python scripts/smoke_test.py
```

该命令依次执行 Python 编译检查、发布文件核验和核心快速测试；GitHub Actions 使用同一组核心检查。

## 9. 发布检查

快速发布检查：

```bash
python scripts/verify_release.py
python -m pytest -q \
  tests/test_release_smoke_fast.py \
  tests/test_bayesian_design.py \
  tests/test_florida_sts.py \
  tests/test_broad_benchmark.py
```

完整离线测试：

```bash
python -m pytest -q
```

## 10. 工程目录

```text
app.py
run_demo.py
config/
data/
docs/
outputs/
prompts/
scripts/
src/globalhab_demo/
tests/
```

最短复现说明见：

```text
docs/MINIMAL_REPRODUCTION.md
```

方法细节见：

```text
docs/TECHNICAL_NOTE.md
```

## 11. 结果边界

- 合成基准结果用于检验方法和探索环境，不代表真实海洋预报性能。
- 南澳模块是事件回放，不用于训练合成模型。
- Florida/Gulf 当前采用一阶表层流场位移约束，不等同于完整三维 Lagrangian 粒子追踪。
- 生物响应参数未针对特定物种、生命阶段或养殖场进行校准。
- 当前输出不构成死亡率估计、毒素监管阈值、业务预警或自动控制指令。

## 12. 许可证与第三方数据

代码采用 MIT License。

第三方数据仍遵循其原始许可证和引用要求，详见：

```text
THIRD_PARTY_DATA.md
```

英文版说明保留在 `README_EN.md`。


## 13. 结果解释纪律

- 默认合成试跑的 `AP≈0.624`、Random 恢复率等数字引用时应注明“默认注册试跑”；
- South Australia 回放 KPI 随筛选变化，但稀疏采样不足以支持连续传播机制分析；
- Norway 回放 KPI 可以变化，但严格前向 AP≈0.102 属于注册审计结果，不随回放筛选变化；
- 当前 TE/CTE 峰值由当前运行重新估计；空间 Durbin 的数值会重算，但当前竞赛等价实现仍使用预注册的 14 天滞后异常暴露变量；
- TCN 5-seed 审计对应默认 720 天 / seed42 / budget8 / Region D / 25% 前向测试，非默认设置下不能与当前 Logistic / RF 结果直接横比；
- Florida/Gulf 在线回顾验证不预填固定高分，外部服务、日期覆盖或样本支持不足时允许 `DEFER`。

## 中国近海补充验证

在“真实数据训练与验证”工作区选择“中国近海调查”，查看渤海、黄海、东海、南海的实测覆盖、跨年分子检出检验与来源。此页不是中国内地提前7天藻华预警成绩。详情见 `CHINA_MAINLAND_VALIDATION.md`。运行 `python scripts/validate_mainland.py` 重算。

## 上传自有观测运行模型

在“左侧工作区 → 自有数据分析”上传CSV，使用“训练与验证”或“基础模型对照”。支持传统模型、轻量TCN，以及Chronos-Bolt-small和Qwen2.5-0.5B-Instruct本地推理。安装、输入要求和下载结果说明见 `OWN_OBSERVATIONS_GUIDE.md`。这两个入口使用本次上传数据计算，不调用已封存的香港实验分数。

## 完整自有数据工作台

自有数据分析现支持35个模型/融合方案条目，历史验证与最新时点未来预测分开，可下载本次结果和指标解读。科学解释/生理过程等非分类模型在目录说明其原模块入口。数据条件、基础模型运行资源和本轮测试范围详见 OWN_OBSERVATIONS_GUIDE.md。
