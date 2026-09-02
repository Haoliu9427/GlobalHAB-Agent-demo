# GlobalHAB-Agent

GlobalHAB-Agent 是一个面向跨区域有害藻华（Harmful Algal Bloom, HAB）研究的可运行原型。系统将环境条件、输运信息、生物观测和水产养殖响应放在同一分析流程中，用于情景分析、传播时滞检验、模型比较和严格前向验证。

## 1. 主要功能

Streamlit 界面包含六个工作区：

1. **风险研判**：对代表性海区进行 7 / 14 / 30 天情景推演和相对风险排序。
2. **真实事件回放**：包括南澳 qPCR 事件回放、挪威长期监测前向验证、Florida/Gulf 真实流场约束回顾分析，以及现场数据前向验证。
3. **生物响应沙盘**：比较 HAB、高温、低溶解氧、养殖密度和投喂情景下的网箱鱼相对压力变化。
4. **科学分析**：包括多尺度异常检测、传播路径诊断、TE/CTE 时滞分析和空间 Durbin 效应分解。
5. **探索与验证**：包括有限预算实验选择、Bayesian 策略比较、负对照、模型 Benchmark 和严格留出验证。
6. **数据与溯源**：展示数据质量、数据来源、结果边界和可下载证据文件。

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

