# GlobalHAB-Agent

本工程新增真实观测训练与验证模块，并在侧栏形成四个一级工作区：**研究与验证**、**自有数据分析**、**现场影像甄别**、**大模型结果解读**。原有24候选、8步合成实验和七个研究页签全部保留。新增内容、实测结果、命令和适用范围见 **[真实数据训练与验证说明](REAL_DATA_GUIDE.md)**；一览表见 **[实验结果汇总](REAL_RESULTS.md)**。已包含下载的数据快照、训练权重、逐样本预测和测试记录，无需重新训练即可查看结果。请作为本地独立工程使用。

GlobalHAB-Agent 是一个面向跨区域有害藻华（Harmful Algal Bloom, HAB）研究的可运行原型。系统将环境条件、输运信息、生物观测和水产养殖响应放在同一分析流程中，用于情景分析、传播时滞检验、模型比较和严格前向验证。

## 1. 主要功能

Streamlit 侧栏包含四个一级工作区：

- **研究与验证**：项目主体科研流程，包含下列七个研究页签；
- **自有数据分析**：上传现场CSV，完成历史验证、多模型比较和可选未来预测；
- **现场影像甄别**：手机拍摄或上传海面照片，先做图像质量门控，再按场景自适应路由到 EfficientNet / ConvNeXt / DINOv2，融合现场元数据并执行不确定性/OOD检查；EfficientNet 与 ConvNeXt 即使在完全离线环境也可执行确定性的深度原型编码路径，公共预训练权重可在首次运行时自动缓存；DINOv2在依赖与权重可用时进入路由。若自适应模式没有任何深度分支真正执行，则直接 DEFER，而不是把规则基线伪装成深度结果；
- **大模型结果解读**：调用用户配置的DeepSeek、Qwen或其他Chat Completions兼容服务，对已经计算完成的项目结果、自有数据结果或用户上传结果文件进行结构化解释。该工作区不重新训练模型、不修改指标。

“研究与验证”工作区包含七个页签：

1. **风险研判**：对代表性海区进行 7 / 14 / 30 天情景推演和相对风险排序。
2. **真实事件回放**：包括南澳 qPCR 事件回放、挪威长期监测前向验证、Florida/Gulf 真实流场约束回顾分析，以及现场数据前向验证。
3. **生物响应沙盘**：比较 HAB、高温、低溶解氧、养殖密度和投喂情景下的网箱鱼相对压力变化。
4. **科学分析**：包括多尺度异常检测、传播路径诊断、TE/CTE 时滞分析和空间 Durbin 效应分解。
5. **探索与验证**：包括有限预算实验选择、Bayesian 策略比较、负对照、模型 Benchmark 和严格留出验证。
6. **数据与溯源**：展示数据质量、数据来源、结果边界和可下载证据文件。
7. **真实数据训练与验证**：NOAA长期观测、中国香港报告序列、轻量时序网络、基础模型及语言模型对照、时间/网格留出、逐年稳定性和输入敏感性。


## 1.1 现场影像甄别工作区

该工作区支持手机直接拍摄或上传JPG/PNG海面照片，先检查分辨率、亮度、过曝、反光和纹理质量，再输出“正常/未见明显异常、绿色水体异常、红棕色水体异常、高浑浊/泥沙样、表层浮沫/漂浮物样或不确定”等**视觉现象**。用户还可以填写水色、异味、泡沫/浮膜、近期高温以及温盐、DO、Chl-a等现场信息。

决赛工程进一步加入 **Adaptive Visual Screening Router (AVSR)**：质量合格后根据颜色异常、浮沫/浑浊纹理和场景不确定性，优先选择 EfficientNet-B0、ConvNeXt-Tiny 或 DINOv2。编码器之后先查找项目训练得到的轻量融合头；若不存在，则使用内置“视觉现象原型头”，将冻结视觉 embedding 与透明颜色/纹理线索、12维现场元数据共同融合。系统再用预测熵、Top1/Top2 margin、多分支 disagreement 与 OOD 检查决定是否 `DEFER`。内置原型头让深度路由在没有项目照片训练头时仍可真正执行，但它只是视觉现象筛查头，并不等于经过真实HAB现场照片校准的藻华分类器。页面输出的是 `低 / 中 / 高 / DEFER` **复核优先级**，不是HAB发生概率；普通照片不能确诊具体藻种或毒素。完成甄别后可一键跳转到“大模型结果解读”，但只传递结构化文字摘要，照片本身不会自动发送给远程模型。完整说明见 `FIELD_VISUAL_SCREENING_GUIDE.md` 与 `VISION_ROUTER_GUIDE.md`。

## 1.2 大模型结果解读工作区

该工作区可以直接读取：项目核心合成发现、模型Benchmark、挪威长期前向验证、南澳事件回放、真实观测训练、中国近海跨年检验、生物响应沙盘，以及当前会话最近一次自有数据分析结果、最近一次现场影像甄别结果；也可上传CSV、JSON、TXT或Markdown结果文件。

可选择“科研结果解读”“答辩讲解”“论文结果段”“管理与应用摘要”四种输出方式，并填写一个额外问题。发送前会在页面上显示结构化结果摘要，只有用户明确勾选授权后才会调用远程模型。API Key只保存在当前Streamlit会话内存，不写入工程、结果包或下载文件。

大模型只负责解释上游已经计算的结果，不重新计算AP、Brier、ECE、概率、阈值或因果关系。系统Prompt固定要求区分合成验证、真实观测、事件回放和未来无标签预测，并禁止把结果改写为未经验证的死亡率、经济损失、监管阈值或自动运营指令。远程服务配置见 `QWEN_ONLINE_SETUP.md`；完整使用说明见 `LLM_INTERPRETATION_GUIDE.md`。

DeepSeek V4 兼容说明：当前 DeepSeek Chat Completions 默认开启思考模式。结果解读页对 DeepSeek 默认使用“稳定解读（推荐）”，显式关闭思考并只读取最终可见 `content`；用户仍可切换低/高强度思考。若思考模式只返回推理内容、最终 `content` 为空或达到输出长度上限，程序会自动以稳定模式重试一次，而不会把 `reasoning_content` 当成解读结果。

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

基础部署不需要深度视觉依赖。若要启用 DINOv2 / ConvNeXt / EfficientNet 路由，再额外安装：

```bash
python -m pip install -r requirements-vision.txt
```

主 `requirements.txt` 已包含深度视觉运行依赖。EfficientNet / ConvNeXt 优先使用公共预训练或本地权重，完全离线时仍可使用确定性原型编码初始化保证真实深度前向；DINOv2在Transformers与模型缓存/下载可用时启用。自适应模式若没有任何深度分支成功执行，则返回DEFER。

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


## Qwen远程API入口
新增无需本地torch的Qwen远程预测及独立结果解读。服务器配置见 `QWEN_ONLINE_SETUP.md`。未配置真实服务，本次仅通过模拟响应集成测试；不代表远程模型真实预测性能。原有实验不变。

## 用户自填远程API
在自有数据分析中选择自行填写，输入公网HTTPS API地址、模型ID和密钥；支持Chat Completions兼容大模型，不限Qwen。详见QWEN_ONLINE_SETUP.md。

## 1.2 持续学习型现场视觉 Agent

“现场影像甄别”现在包含三个子页：**现场影像甄别 / 我的影像数据 / 模型训练与版本**。用户可以把人工或实验室确认后的照片保存在本地影像库中，选择哪些照片进入后续训练；系统优先把 entropy 高、margin 小、多模型不一致、OOD 或 DEFER 的照片加入主动学习待确认队列。重新训练时冻结 DINOv2 / ConvNeXt / EfficientNet 视觉编码器，只更新轻量分类/现场元数据融合头，并在固定留出集上与公共视觉基线比较。训练完成不会自动覆盖当前模型，只有通过门槛且由用户显式注册的候选版本才写入 `vision_models/active_model.json` 并进入后续推理。详见 `CONTINUOUS_VISUAL_AGENT_GUIDE.md`。

公共第三方原始照片不会直接打包进普通Git工程。项目提供 `PUBLIC_VISUAL_DATA_SOURCES.md`、`scripts/fetch_public_visual_data.py` 和 `scripts/build_public_visual_adapter.py`，用于按需获取公开表层藻华影像并只保存聚合embedding adapter/轻量模型资产。

## 1.3 Case驱动的跨工作区证据闭环

研究与验证、现场影像甄别、视觉持续学习和大模型结果解读现在由统一 `Case / Evidence Ledger` 串联。风险研判页可把当前高风险候选、Route、Lag、Top-k容量与事件覆盖生成现场复核任务；影像页自动读取该Case，并把视觉筛查登记为独立的现场证据层。专业人员、显微镜、qPCR或毒素确认可继续写入同一证据链，并可把确认后的照片加入视觉训练库。大模型工作区新增“当前完整Case（推荐）”，综合解释风险、视觉、现场环境和实验室证据，并可把解释文本保存为Case记录，但不会改变上游指标或证据等级。

详细结构、证据等级和复现命令见 `CASE_EVIDENCE_LOOP_GUIDE.md`。

## 1.4 Case闭环界面一致性修复

研究与验证、现场影像甄别、自有数据分析和大模型结果解读现使用统一的工作区Hero与科研KPI卡片视觉语言。风险研判生成现场复核任务、现场影像读取当前Case时均使用可自动换行的四列KPI卡，长海区名和Case ID不会再被原生metric截断。修复了生物响应说明表中局部变量与`case_manager.evidence_rows()`重名导致的跨tab `TypeError`；Case证据链现在可在数据来源与复核页正常显示，并继续供其他研究页签整页执行。

## 1.5 多Case现场复核任务队列

风险研判现在不仅能把最高候选生成单个Case，还可以在候选表中勾选多个海区，批量生成独立现场复核任务。每个Case仍分别保存位置、风险、Route/Lag、现场照片、元数据、实验室证据和大模型解释，不会用一张照片同时验证多个海区。

Case状态采用显式生命周期：`待现场复核 → 现场复核中 → 已完成视觉筛查 / 视觉DEFER → 已有专业/实验室确认`，并提供`已取消复核`和`已归档`旁路。左侧“Case / 现场任务”可开始/继续、取消、归档、恢复或永久删除；永久删除前会检查是否已有影像样本或历史模型训练引用。删除Case不会改写已经训练完成的模型，用户可选择同时把关联样本从未来训练集中移除。

现场影像甄别页新增待复核任务队列，可切换到下一个任务，也可在完成当前视觉登记后直接点击“登记并处理下一个”，用于连续处理批量现场任务。

## 1.6 工作区界面一致性优化

现场影像工作区的三个子页已统一改为“现场影像甄别 / 我的影像数据 / 模型训练与版本”，不再显示①②③编号。影像数据与模型训练页的摘要指标改用与现场筛查、研究任务相同的科研KPI卡片，并统一白底圆角、海洋色顶部强调线、长文本自动换行和移动端响应式布局。

自有数据分析Hero删去实现细节型第二说明句，仅保留工作区定位。左侧四大工作区导航改为独立品牌卡 + 卡片式工作区选择，统一悬停、选中和Case任务面板视觉层级。

### 大模型连接状态修复
DeepSeek 结果解读现使用后端有效默认值，而不是只依赖浏览器输入框显示状态。读取 `/models` 后可在“实际调用模型”中直接选择服务返回的模型；页面会显示当前实际调用模型及连接参数是否完整。如果“生成大模型解读”未启用，会直接提示是结果摘要、远程配置还是发送授权缺失，避免出现“看起来都填了但按钮仍灰色”的情况。

### 2026-09-16 界面布局细化
- 现场影像甄别中的“拍照或上传 / 现场信息 / 甄别结果”取消步骤编号，并移除首屏额外科学边界提示。
- 摄像头区域增加中文权限提示，并对 Streamlit 标准摄像头权限帮助文本做中文化样式覆盖。
- 大模型结果解读 Hero 副标题改为与其他工作区统一的 tagline 字级与字重。
- 自有数据分析、现场影像甄别、大模型结果解读的成对卡片统一为等宽列，并在桌面端按同一行等高拉伸；窄屏自动恢复为自适应高度。

### 1.7 成对卡片严格等高
针对自有数据分析与大模型结果解读中仍可见的左右卡片高度差，桌面端现在直接对 Streamlit keyed container 及其 border wrapper 做伸展，并为对应卡片组设置统一高度下限。自有数据的远程服务区同时改为紧凑布局：API地址与模型ID同排、连接动作同排，模型列表和隐私说明折叠显示，减少不必要的纵向占用。900px以下继续使用自然高度，避免移动端空白。

### 界面展示优化（2026-09-16）

- 大模型结果解读的结果来源卡新增“当前输入概览”和只读摘要预览，使左右等高卡片保持信息密度一致。
- “特别想让大模型回答什么”使用面向普通用户的证据一致性与后续复核示例，不展示项目内部修改过程。
- 自有数据分析的观测数据、预测任务、远程服务、模型运行卡片重新组织内容，在保持等宽等高的同时补充字段结构、输出内容、连接状态、发送范围和运行前检查。

### 2026-09-16 card balance refinement
The Own Data observation/task pair and the LLM source/mode pair now use a stricter equal-height desktop grid. The observation card adds upload checks, the prediction-task card adds pre-run checks, and the LLM source card adds send-readiness information so the cards remain visually balanced without decorative blank space. The template-download and upload-status controls are now matched in width and height.

- 大模型结果解读的左右卡片采用严格等高布局；右侧新增输出长度、重点关注、关键数字核对清单等实际解读控制，使卡片在对齐的同时保持内容密度。

### 2026-09-16 最终卡片平衡
自有数据分析“观测数据 / 预测任务”和大模型结果解读“选择要解读的结果 / 解读方式”两组卡片改为由外层等高列直接承担白底、边框和顶部强调线，避免 Streamlit 内部容器高度不同导致底边错位。内部同时补充推荐数据组织、结果记录、可解释内容、结构化摘要下载和适用场景，使卡片在严格对齐的同时保持有效信息密度。


### 2026-09-16 卡片信息精简
自有数据分析和大模型结果解读的成对卡片继续保持等宽等高，但去除重复说明和过大的固定高度；核心信息改为更紧凑的字段/规则摘要，使留白更多出现在内容区块之间，而不是卡片顶部或底部。

- 2026-09-16：精简自有数据分析与大模型结果解读卡片的常驻文字；保留等宽等高布局，将字段要求、验证规则和远程调用说明收进折叠项，并用紧凑底部状态替代大段说明。

### Remote model connection status UI
The own-data workspace now renders model-list/test-connection feedback as a compact full-width status strip below the action buttons. This prevents success/error messages from being squeezed into a narrow button column and keeps the paired “远程大模型服务 / 模型与运行” cards aligned.

### 界面修复
- 远程服务连接按钮状态与服务商默认模型同步；成对卡片继续等宽等高，“开始分析”固定在运行卡片底部；结果解读左卡使用更大的只读摘要预览和卡内完整摘要折叠区，减少无效底部留白。

- 2026-09-16：修复自有数据分析远程模型“测试连接”误禁用；远程服务/模型运行卡改为严格等高外层卡片，开始分析固定在右卡底部；大模型解读左卡扩大摘要预览并把留白分配到内容区块之间。


### 2026-09-16 卡片最终间距与连接修复
- 自有数据分析的“测试连接”不再因前端/会话状态短暂不同步而变灰；点击后再校验 API 地址、Key 和模型。
- “远程大模型服务 / 模型与运行”桌面端使用同一显式卡片高度，模型运行卡在各功能组之间分配弹性间隔，并保持“开始分析”位于卡片底部。
- 大模型结果解读左侧摘要预览缩短，剩余高度均匀分配到卡内内容组之间，避免右卡底部出现大面积空白。


### 2026-09-16 最终卡片几何加固
- 自有数据分析的远程服务与模型运行卡在桌面端直接采用相同固定容器高度，避免 Streamlit wrapper 导致视觉高度不同；模型运行卡以多个弹性间隔分配空白，开始分析保持在卡片最底部。
- 大模型结果解读将只读摘要预览压缩为简短窗口，并在左右卡片内容组之间分配剩余空间，保持严格对齐且不在底部形成大片空白。

### 2026-09-17 · 自有数据双卡三层对称布局
- “远程大模型服务 / 模型与运行”保持 1:1 等宽、同高，但不再用多段弹性间隔把内容拉散。
- 两张卡片统一为“顶部主要配置 → 中部状态摘要 → 单一弹性区 → 底部最终动作”的三层结构。
- 左卡中部只显示连接状态与当前模型；底部固定远程发送授权和隐私说明。
- 右卡中部压缩为验证方式、模型数、预测时效和训练预算四项摘要；底部固定运行保存说明和“开始分析”。


### 界面细节
自有数据分析中的远程模型服务卡采用“配置—状态—授权”三层布局，与右侧模型运行卡保持等宽等高；剩余空间分布在层之间，不在卡片中部形成单块大面积空白。


### 2026-09-17 UI refinement
The own-data remote-service/model pair now uses content-driven equal-height grid cards without fixed minimum height or large blank spacer blocks.

### 2026-09-17 最终按钮位置微调
“模型与运行”卡片保持现有等宽等高结构与灰色运行记录说明位置，仅将“开始分析”主按钮轻微下移，匹配最终确认的参考版式。
