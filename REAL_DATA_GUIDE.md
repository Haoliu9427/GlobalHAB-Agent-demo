# 真实数据训练与验证说明

这次升级把预测模型接到了完整的公开观测快照上，提供训练、选模、校准、留出评估、保存模型和新输入预测的完整流程。原有情景推演、生物响应沙盘和传播机制分析继续保留；这些情景模块的分数没有改成真实观测概率，也不与新增实验混用。

## 1. 取得了哪些真实数据

| 数据 | 本次取得的内容 | 训练用途 |
|---|---|---|
| NOAA HABSOS | 223,394条原始记录，1953年至2026年9月；下载时冻结全部对象ID并逐页核对 | Karenia brevis浓度、同次采样温度/盐度/风速、位置和历史时序 |
| 香港渔农自然护理署AFCD | 1,931行赤潮报告档案；同一事件可能有多个物种行 | 去重后构造完整历史周序列，预测下一周是否有赤潮报告 |
| 原工程挪威监测 | 5,919条记录 | 保留原前向验证；没有直接拼接到另一物种/另一终点的训练标签 |
| 原工程南澳qPCR | 115条记录 | 保留事件回放 |

HABSOS是本工程目前规模最大的真实观测来源。没有完成全球所有数据库的同口径清点，因此不宣称“全球最大”。原始记录数包含被质控排除的记录；实际样本量以各实验的split_manifest.json为准。HABSOS同网格同日取最大浓度，并仅使用质量标记合格、单位可比的K. brevis记录。

香港是已经完成的中国水域验证，不能概括成渤海、黄海、东海、南海均已验证。香港档案中的空白周表示“没有报告”，不是“实测不存在藻华”；它与HABSOS浓度事件是两个独立任务。内地连续监测数据可通过现场CSV接口进入同一流程，但本交付没有伪造内地阴性样本或验证结果。

## 2. 算法改了什么

**EcoTemporalNet**是一个缺测感知的轻量多尺度时序卷积网络。输入先以训练期中位数填补、训练期尺度标准化，并附带每个变量的观测/缺测掩码。三个左侧填充的卷积分支分别读取3、5、9次观测范围，门控权重组合不同时间尺度，再与最新状态融合。参数上限50,000，实际参数量逐种子写入结果。HABSOS是不等间隔采样，模型同时输入历史采样间隔；卷积尺度表示观测次数，不可说成固定天数。

**EcoFusion**组合时序网络与HistGradientBoosting。权重仅在测试前的验证窗口从0、0.25、0.5、0.75、1中选择。保留Logistic、Random Forest、梯度提升、季节和持续性基线。融合权重可为0或1，不强迫复杂模型进入最终组合。

Chronos-Bolt-small是约4,772万参数的预训练时序基础模型，读取规则周序列，不是聊天LLM。另提供Qwen语言模型的固定提示词二分类分数，以及Ollama本地模型接口。Qwen分数通过0/1候选token的相对概率获得，再用测试前校准窗口校准，不把模型生成文字当作有效预测。两类模型均与相同香港样本ID比较；Chronos使用报告计数历史，其他模型还使用季节编码，这一输入差别必须保留在解释中。

公开模型的预训练语料与公共档案是否重叠无法独立排除，不能声称基础模型比较完全没有预训练污染。Qwen 0.5B是小型语言模型参照，不代表所有大语言模型。接入更大的模型可以复用命令，但需要重新完成同样的留出验证。

## 3. 如何保证结果可检查

按时间依次分为训练、选模、概率校准、最终测试；跨边界的未来标签会被清除。HABSOS额外按哈希固定留出约五分之一0.1°网格，这些网格不进入训练、选模、校准。邻近网格仍可能相关，所以该测试证明的是未见网格表现，不是整个陌生海盆的外推能力。所有模型使用完全相同的测试样本ID，保存SHA256。

每个任务运行17、42、73三个训练随机种子，报告AP、Brier、10箱ECE、ROC-AUC、前10%容量召回率、参数量与耗时。AP不是准确率。年份分组检查、20%人为缺测和小幅噪声扰动检验写入stability.csv；实测高温子集采用训练期温度90分位，不能等同于完整海洋热浪或台风验证。

季度块bootstrap比较成对AP差值，区间跨0则不宣称稳定优越。这里没有针对测试结果反复挑选最佳种子。传统模型使用固定配方，时序网络有两次容量实验，不等同于对所有传统模型穷尽最优调参。耗时随硬件和并发负载变化；基础模型耗时还可能包括首次下载。季节/持续性耗时记0表示未单独计时，不表示真实零成本。

概率校准使用独立校准窗口。附带的split-conformal集合只报告经验覆盖率；时序依赖下不承诺严格90%未来覆盖。历史档案未提供每条记录完整的发布延迟与修订历史，因此属于回顾性时间前向验证，不能冒充部署后实时验证。

## 4. Agent与解释

真实训练中的控制器先试宽度16，读取训练和验证损失；泛化差距大时尝试宽度8，否则尝试24，再按验证损失保留模型。每次选择有理由、参数、反馈和可见分区日志。它是受约束的反馈实验控制器，不使用LLM推理，能力范围就是这些明确的实验动作。原24选8的探索结论保留原定义，不用新增结果替换原科学结论。

遮蔽单个输入后的AP变化解释模型依赖哪些信息；它不证明因果作用。原工程的TE/CTE、反向路径检验、空间Durbin分析与真实流场回顾保留在原模块，不能把统计信息流或TCN的“因果卷积”宣称为已识别真实生物因果效应。

## 5. 运行

建议Python 3.12，在工程根目录运行。仅查看网页及已保存结果无需安装torch，也无需下载任何基础模型。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

需要重训时，先安装CPU版torch，避免无GPU设备下载大型CUDA依赖：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
python scripts/train_real.py --dataset habsos --horizon 7 --output outputs/my_habsos_7d
python scripts/train_real.py --dataset habsos --horizon 14 --output outputs/my_habsos_14d
python scripts/train_real.py --dataset habsos --horizon 30 --output outputs/my_habsos_30d
python scripts/train_real.py --dataset hong_kong --foundation --output outputs/my_hk
python scripts/compare_llm.py --experiment outputs/real_training/china_hk_7d --output outputs/my_llm
python scripts/audit_real.py outputs/my_habsos_7d
python -m pytest tests/test_real_training.py tests/test_release_smoke_fast.py -q
```

已有证据目录不会被训练命令覆盖。Chronos/Qwen权重首次运行需从Hugging Face下载；大权重未重复打入工程。自研模型权重、传统模型、校准器和预处理器均已打包。默认CPU两线程，小型网络不要求GPU。LLM比较时间较长，保持命令运行到status.json显示completed才算完成。Hugging Face语言模型评分每8条保存断点，中断后可对同一命令增加`--resume`；输入、模型revision或提示词发生变化会拒绝复用断点。

固定阈值预警可运行`python scripts/evaluate_warning_thresholds.py outputs/real_training/habsos_7d`。程序在校准期以10%报警预算确定阈值，测试期不调阈值，输出命中、误报、漏报及实际报警比例。该比例在未来可因分布变化而不同于10%。验证脚本`python scripts/verify_package.py`可离线核对交付包内的文件完整性。

数据快照已包含，无需联网重下。若要建立新快照，先复制工程，再运行`python scripts/fetch_public_snapshot.py`；已有冻结ID会复用，若重新获取全部最新成员，需在复制工程中清理该下载目录。实时服务记录可能修订，新快照不应覆盖已发表实验的输入。

## 6. 自有现场数据与新预测

现场CSV必须包含station_id、date、available_at、latitude、longitude、observed_event、value、source。observed_event必须是人工确认的0/1，未知留空；物种、浓度单位、阈值、采样方式与地域需附来源说明。可附temperature、salinity、dissolved_oxygen、nitrate、phosphate、silicate、u_current、v_current。不同物种/阈值不要无说明合并。

当前适配器要求记录在起报时已经可用；available_at晚于date会被拒绝，需要先按真实发布时刻建立独立的as-of数据表。接口不根据坐标自动认证数据来自中国或符合质量标准。

```bash
python scripts/train_real.py --dataset china_field --input my_observations.csv --output outputs/my_field
python scripts/predict_real.py --experiment outputs/real_training/china_hk_7d --history data/training_real/example_history_hk.json --output my_prediction.json
```

预测JSON要求features与训练清单顺序完全相同，histories形状为[样本数,历史长度,变量数]。示例取自已留出历史，只为演示调用。返回的是对应定义的事件概率，不自动转换为死亡率、停养指令或保障收益。

## 7. 文件位置

* `src/globalhab_demo/real_training/`：数据、模型、校准、比较、推理和界面。
* `data/training_real/raw/`：原始快照与来源、下载时间、SHA256。
* `outputs/real_training/`：实际实验、逐行预测、训练权重、选择日志、结果表。
* `validation/`：本次运行和测试记录。
* `REAL_RESULTS.md`：由已完成实验自动汇总的结果。
* `REAL_DATA_SOURCES.md`：权威来源链接与许可说明。

网页不自动在后台重训，也不把文件缺失替换成虚构指标。本次交付未写入GitHub或更新线上站点。
