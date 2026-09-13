# 自有观测：模型训练与基础模型对照

入口位置：左侧工作区 → 自有数据分析。上传CSV后，可在“训练与验证”和“基础模型对照”之间选择。原有中国近海页和已封存实验保持独立，网页运行不覆盖原实验文件。

## 两个入口

- 训练与验证：Logistic、RandomForest、HistGradientBoosting、EcoTemporalNet可多选。前三种参数固定，TCN宽度16、参数少于50000，使用验证期早停；全部运行17/42/73三个种子。
- 基础模型对照：Chronos-Bolt-small、Qwen2.5-0.5B-Instruct可多选，并可加入传统模型/TCN同场对照。使用本次上传数据推理，独立校准，不复用香港模型或成绩。不是可任意加载所有厂商大模型的通用接口。

Chronos为时序基础模型，输入为连续逐日的历史0/1事件标签，预测所选第N天。必须所有历史连续、目标恰好在第N天；不满足条件会阻止运行，不伪造重采样标签。Qwen是语言模型，输入为字段名称、原始历史值、历史事件、采样间隔和用户填写的事件定义，读取0/1两个token的相对分数。两者均固定推理种子42；不是训练了三个大模型。

## 数据和目标

必需字段：station_id,date,available_at,latitude,longitude,observed_event,value,source。可选环境变量：temperature,salinity,dissolved_oxygen,nitrate,phosphate,silicate,u_current,v_current。页面可下载空模板。还需填写物种、事件阈值和value含义/单位，避免把不同端点混为一谈。

日期必须正确；同站同日只能有一条记录。observed_event只允许明确0/1，未知留空且不补0。当前适配器要求available_at不晚于观测日期，延迟数据必须先按实际可用时间处理。每条样本用12次过去观测预测N至N+3天内首次复测的标签，不是这段时间内任何一次事件。

按时间分训练/验证/校准/测试；标签越过分界的样本删除。有至少10个站点时启用额外站点留出。训练、验证、校准和时间测试各至少20条且有两类标签。预处理仅拟合训练期；概率校准仅拟合校准期；测试记录和哈希对所有模型一致。两入口使用同一数据、同一目标时划分一致，但不同模型的原生输入形式不同，协议有记录。

## 运行环境

基础网页依赖保持原requirements.txt。传统模型即可运行，无需安装大模型。TCN需要torch；基础模型需要可选环境。在工程根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
python -m streamlit run app.py
```

模型在运行Streamlit的主机上推理；不会调用远程聊天推理API。首次使用从Hugging Face下载权重并缓存在该主机，需要网络；权重未打包。Qwen采用CPU float32，模型参数约占2GB，运行仍需额外内存，建议使用内存充足的本机。Streamlit Cloud的实际资源额度未在本次验证；缺依赖会给出提示，资源不足可能使主机终止进程。

官方模型来源：https://huggingface.co/amazon/chronos-bolt-small 和 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct 。下载后记录实际revision。预训练数据与公开观测的重叠不能独立排除。

## 结果与操作

点击对应运行按钮后显示进度。CSV上限20MB/100000行；基础模型评估历史上限2000条，超出会拒绝整次运行，不自动缩减共同留出集。运行是同步的，重模型可能需要较长时间。任何失败不显示部分成绩为成功结果；切换数据或配置后旧成绩不会冒充当前成绩。

结果包括各模型/种子的AP、Brier、ECE、Top10%命中指标、耗时、参数量。基础模型和经典模型计算方式不同，耗时包含各自训练/加载与推理及校准，预处理时间另记。全局/空间测试分别报告；不根据单个最佳分数自动宣称稳定改进。要发表稳定优势需增加独立航次、配对区间和重复外部检验。

下载结果ZIP含协议、输入CSV哈希、分割行表、逐条预测、指标、训练权重和校准器；基础模型权重不包含。输入CSV不放入结果ZIP，请自行保留。临时运行目录自动清理，完成后的结果ZIP保存在当前会话内以供下载，不写入仓库。ZIP包含站点与结果信息，请自行决定分享范围。

命令行调用同一运行逻辑，输出文件必须未存在：

```bash
python scripts/run_own_observations.py --input observations.csv --models Logistic HistGradientBoosting --description "填写真实的物种、阈值和变量单位" --output my_results.zip
```

## Qwen在哪里

已完成的香港Qwen实验位于“研究与验证 → 真实数据训练与验证 → 连续观测预测 → 基础模型对照”，显示完整模型名Qwen2.5-0.5B-Instruct以及与EcoFusion的组合。原始证据文件保留LLM/LLM+EcoFusion命名，页面显示名称不改变实验数据。运行自有数据请选择“自有数据分析 → 基础模型对照”，Qwen与Chronos均在选择框中显示。

该模块独立于合成探索设置，进入时不会执行研究工作区的情景计算。上传逻辑与上一版一致。这里的预测结果是基于带标签历史数据的留出验证，尚不提供对无标签未来数据的一键运营预报。
