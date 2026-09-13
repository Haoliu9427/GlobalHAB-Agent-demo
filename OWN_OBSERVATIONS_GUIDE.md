# 自有数据分析：完整模型目录、历史验证与未来预测

入口：左侧工作区 → 自有数据分析。原有研究工作区与已封存成绩独立，上传过程保持CSV方式，没有修改网络上传设置。

## 操作

1. 上传有标签的历史观测CSV。填写物种、事件阈值与value含义/单位。
2. 选择“历史预测验证”，或“历史验证 + 最新时点未来预测”。
3. 选择7/14/30天时效，从完整目录多选模型。融合模型会自动运行其组件，页面明确列出实际执行列表；季节基线始终同时运行。
4. 开始分析，查看本次输入哈希、指标、未来概率与结果解读，下载结果包。数据或设置改变后旧成绩隐藏，不匹配的结果不复用。

## 全部模型如何对应

- 基线：Seasonal Climatology、Event Persistence。
- 统计/机器学习：Logistic、GAM (Spline Logistic)、Gaussian Naive Bayes、kNN、RBF-SVM、Decision Tree、Random Forest、Extra Trees、AdaBoost、Gradient Boosting、HistGradientBoosting、MLP、XGBoost、LightGBM。
- 时序与科学结构：Lightweight TCN、EcoTemporalNet、STS-Interaction GLM、STS-Gated TCN。
- 时序基础模型：Chronos-Bolt-tiny、small、base。
- 语言模型：Qwen2.5-0.5B/1.5B/3B-Instruct、SmolLM2-360M-Instruct。
- 融合：EcoFusion，以及上述七个基础模型分别与EcoFusion融合。

共35个可执行模型/方案条目。以模型家族计数，不把不同历史任务的超参数组合重复算作模型。过去的RandomForest与Random Forest统一为后者，LLM统一使用真实模型名。基础模型的选择数不等于已证明有效的模型数。

空间Durbin、TE/CTE、生物响应过程、养殖风险投影及Bayesian/Thompson/Random探索策略在同一目录说明其用途与原模块入口；它们不是同一目标的分类模型，不编造AP。中国近海分子检出率基线需要物种/检测方法数据，保留在中国近海调查。

原始合成环境模型在自有数据上重新训练。Lightweight TCN沿用两层因果卷积结构，输入维数随上传字段变化，因此参数量不是原合成六变量情况下的119。STS-Gated沿用原NumPy双分支和门控实现，但用本次训练期标准化的实测变量；不会把合成场景的常数当作真实海洋参数。这些适配不改写原实验成绩。

## 字段与适用条件

必需：station_id,date,available_at,latitude,longitude,observed_event,value,source。
可选：temperature,salinity,dissolved_oxygen,nitrate,phosphate,silicate,u_current,v_current。

observed_event只能是明确0/1，未知留空，不补0。同站同日只能一条记录。每个样本使用12次过去观测，包括历史事件和时间间隔；未来标签不进入特征。available_at不得晚于date，延迟观测需先按真实可用时间对齐。

STS另需local_raw_signal、upstream_raw_signal、circulation_residence_proxy、nitrate、phosphate、silicate，以及upstream_available_at。信号必须由提供者按物理上游方向和历史时滞预先对齐；代码检查字段、有限值和可用时间，不假装自动恢复真实流场。缺少条件时目录说明不适用，不能直接选择运行。

Chronos使用连续逐日的历史事件序列，要求所有历史连续、标签复测恰好在第N天。其他模型预测N至N+3天内首次复测的标签，不是期间任意事件。基础模型输入形式不同：Chronos只使用历史事件；语言模型使用字段和原始历史；其融合方案另用EcoFusion的环境信息。

## 划分、融合与解释

训练/验证/校准/时间测试按日期隔离并剔除标签跨界样本；有至少10站时另做站点留出。四个时间分区各至少20条且有两类。缺测处理和标准化只拟合训练期。表格模型参数固定，时序早停/训练预算及融合权重仅由验证期选择，概率校准只在校准期完成。

所有模型共享同一测试ID与哈希。非基础模型运行17/42/73三个种子；基础模型固定种子42推理一次，不宣称重复训练了三次LLM。融合权重0、0.25、0.5、0.75、1以验证期log-loss选取；即使选出0也保留结果。七种融合方案不能被解释为同时都优于组件。

结果解读根据本次指标生成，不调用语言模型编写泛泛的评价。提供AP、Brier、ECE、Top10%覆盖和按季度整块重采样的AP差值区间。少于4个季度，或区间未稳定高于0，或训练出现警告时，不宣称稳定优势。基线比较不是因果证明。

若运行表格模型，用验证期log-loss选择一个解释模型，逐变量遮蔽输入，报告测试AP变化；选择解释模型不看测试排名。这是预测敏感性，不是因果贡献。MLP等未收敛警告会保存在training_warnings.csv并写入解释。

## 未来预测的准确含义

未来预测使用各站最新有标签的12次历史观测；起点是origin_date，不自动等于今天。模型维持训练期权重，使用最新历史输入推理，不为凑预测而填未来事件标签。

future_predictions.csv的y和target_value为空，不计算未来AP、Brier或准确率。label_date表示目标窗口起点；一般任务目标为N至N+3天首次复测，Chronos对应第N天。预测是经历史校准的事件概率，不是死亡率、置信保证或自动运营指令。若历史不足，或无法形成有效训练/校准/测试，则不提供未经检验的预测。

## 安装与成本

先安装requirements.txt。时序模型和基础模型需要可选依赖：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
python -m streamlit run app.py
```

基础模型首次从Hugging Face下载到运行主机，本地CPU推理，不将观测送到远程聊天API。权重不包含在工程ZIP。大规格模型需要更多内存，例如3B float32权重约12GB，尚有额外运行内存；不要在内存不足的主机一次选择全部基础模型。公开预训练数据的重叠不能独立排除。模型标识与运行revision保存在审计文件。

网页限20MB/100000条原始记录，基础模型限2000条评估历史；超过上限不会偷偷抽取更小测试集。运行耗时可能较长，尤其语言模型逐条评分。融合耗时为组件成本之和，重用组件时可能重复计数，不是整项任务的墙钟时间。

## 下载与复现

ZIP包含protocol.json、metrics.csv、逐条测试预测、future_predictions.csv、paired_intervals.csv、input_sensitivity.csv、training_warnings.csv、融合选择、解释文本、训练权重和校准器。输入CSV请自行保留，包内记录其SHA256。临时目录清理，结果ZIP留在当前会话供下载，不写入GitHub。

```bash
python scripts/run_own_observations.py --input observations.csv --models Logistic "Random Forest" EcoFusion --description "填写真实事件定义" --future --output user_results.zip
```

对照软件测试使用显式合成单元测试数据，不是新增的真实海域成绩。模型目录的可选择状态仅检查依赖与数据条件，不代表所有规格都已完成真实生态数据评测；本轮基础模型检查范围见validation里的smoke记录。

## 本轮交付检查范围

26项软件测试通过，包含完整模型目录核对、传统/时序/STS/融合运行、共同留出样本、未来标签为空、数据改变后重算。工作区与上传后的运行界面检查通过。真实权重小样本推理检查覆盖Chronos-tiny、Chronos-small、Qwen-0.5B、SmolLM2-360M；Chronos-base、Qwen-1.5B和3B已配置接口，本轮未逐个加载权重测试，也不填入虚构成绩。测试数据不作为新增海域验证结果。

逐条输出的outside_training_range_features为超出训练期1%—99%范围的字段数，missing_fraction为历史输入缺测比例；它们是分布提示，不是预测出错概率。
