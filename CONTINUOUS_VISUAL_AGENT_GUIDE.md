# GlobalHAB-Agent 持续学习型现场视觉 Agent

## 1. 工作区结构

“现场影像甄别”现在包含 3 个子页：

1. **现场影像甄别**：拍照/上传 → 图像质量门控 → 自适应视觉路由 → 不确定性/OOD → `低 / 中 / 高 / DEFER` 复核优先级。
2. **我的影像数据**：保存用户照片、人工/专家视觉标签、实验室证据等级和是否进入训练集；支持批量导入、主动学习排序与训练库 ZIP 导入/导出。
3. **模型训练与版本**：以公共视觉基线为起点，选择已确认训练数据，冻结 DINOv2 / ConvNeXt / EfficientNet backbone，只训练轻量分类/元数据融合头；自动留出验证后与公共基线在同一测试行上比较，再由用户显式决定注册或不注册。

核心闭环：

```text
公共视觉基线
  ↓
现场影像甄别
  ↓
不确定 / DEFER 样本优先人工确认
  ↓
人工/专家/显微镜/qPCR/毒素证据进入“我的影像数据”
  ↓
重新训练轻量头
  ↓
固定留出验证 + 与当前/公共基线比较
  ↓
候选模型 Candidate
  ├─ 通过门槛 + 用户确认 → Active
  └─ 未通过/用户拒绝 → Rejected / 保留审计
```

## 2. 视觉标签与证据等级

监督训练只针对水体**视觉现象**，不训练具体藻种或毒素：

- 正常/未见明显异常；
- 绿色水体异常；
- 红棕色水体异常；
- 高浑浊/泥沙样；
- 表层浮沫/漂浮物样；
- 不确定（不会进入监督训练）。

证据等级：

- 仅肉眼判断；
- 专业人员确认；
- 显微镜确认；
- qPCR确认；
- 毒素检测确认。

训练时证据等级会转为样本权重；实验室/专业证据权重高于仅肉眼判断。模型自己的预测结果**不会自动写回成训练标签**，避免错误伪标签自我强化。

## 3. 公共视觉基线

默认公共基线采用：

```text
公共预训练视觉编码器（可获取时）
+ 内置水体现象视觉原型
+ 透明颜色/纹理特征
+ 现场元数据
+ uncertainty / OOD / DEFER
```

项目不把大体量第三方照片直接塞进普通 Git 仓库。公开数据源、DOI 和用途记录在：

```text
data/field_visual/public/PUBLIC_DATA_SOURCES.json
```

当前提供的公开数据适配入口包括 **Algal Blooms Sweden - 2023**（Zenodo DOI `10.5281/zenodo.10599927`）。该记录包含 60 个波罗的海表层藻华观测照片；数据记录说明这些藻华经过当地信息中心核实，但没有进一步通过显微镜或遗传方法做物种级注释。因此它只适合做“藻华样表层现象”的领域适配/正样本表征，不用于具体藻种分类。

下载与适配：

```bash
python scripts/fetch_public_visual_data.py
python scripts/build_public_visual_adapter.py --backbone efficientnet
python scripts/build_public_visual_adapter.py --backbone convnext
python scripts/build_public_visual_adapter.py --backbone dinov2
```

`build_public_visual_adapter.py` 只把公共图片压缩成 embedding centroid 与相似度统计，写入：

```text
vision_models/public_baseline/adapters/
```

原始第三方图片仍保存在本地数据目录，不要求提交到代码仓库。公共正样本 adapter 只提供一个弱的 bloom-like 视觉支持信号，不会生成物种标签。

## 4. 我的影像数据

默认目录：

```text
data/field_visual/user_library/
├── records.csv
└── images/
```

每条记录包含：

- SHA256 / sample_id；
- 原图路径；
- 视觉标签；
- 证据等级；
- 是否进入训练；
- 站点/海域ID；
- 日期/时间；
- 水色、异味、泡沫/浮膜/漂浮物；
- 温度、盐度、DO、Chl-a；
- 最近一次筛查类别与复核优先级；
- 主动学习分数。

同一张图片按 SHA256 去重。影像库支持 ZIP 导出/合并导入，适合云端 Streamlit 本地磁盘可能重置的情况。

## 5. 主动学习

系统把以下情况的照片排在“待人工确认样本池”前面：

- predictive entropy 高；
- Top1/Top2 margin 小；
- 多 backbone 结果不一致；
- OOD；
- 已触发 DEFER。

主动学习分数只决定**人工标注优先级**，不会自动改标签，也不会自动重新训练。

## 6. 用户模型训练

用户模型使用：

```text
冻结视觉 embedding
+ 12维现场元数据
+ 轻量 multinomial Logistic / linear fusion head
```

默认只训练轻量头，不重新训练整个 DINOv2 / ConvNeXt / EfficientNet，因此少量到中等规模照片也可以在普通 CPU/单卡环境完成。

训练前要求：

- 至少 2 个有效视觉类别；
- 默认至少 12 张进入训练的照片；
- “不确定”不进入监督训练；
- 有站点ID时优先按站点/海域分组留出，减少空间泄漏；否则使用随机分层留出，并在模型卡中记录该限制。

## 7. 自动验证与模型晋级

候选模型在固定留出集上报告：

- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- Log Loss；
- ECE；
- 编码器来源；
- 同一留出集上的公共视觉基线指标。

首要原则：**训练完成不等于自动替换。**

候选版本先保存为：

```text
vision_models/user_models/<version_id>/
├── heads/
├── training_manifest_snapshot.csv
├── metrics.json
└── model_card.json
```

只有自动验证满足门槛，且使用的编码器不是“非预训练离线初始化”，页面才开放“注册为当前模型”。用户仍需显式点击注册。未通过或暂不注册的版本继续保存用于审计。

当前模型指针：

```text
vision_models/active_model.json
```

现场甄别会自动读取当前注册版本中的轻量头；切换回 `public-baseline-v1` 即恢复公共基线。

## 8. 科学边界

持续学习机制不改变以下边界：

- 手机照片不用于具体藻种确诊；
- 不用于毒素浓度判断；
- 视觉负结果不能排除肉眼不可见 HAB；
- 公共淡水/波罗的海影像不等同于南海或全球海洋 HAB 的直接验证；
- 用户模型只在其标签、地点、拍摄设备和环境条件覆盖范围内解释；
- 正式科学结论仍建议使用独立站点/年份和实验室确认标签复核。

## 9. 关键代码

```text
src/globalhab_demo/field_visual.py       # 三子页入口与现场甄别
src/globalhab_demo/adaptive_visual.py    # 自适应路由、backbone、uncertainty/OOD
src/globalhab_demo/visual_learning.py    # 影像库、主动学习、训练、版本与注册
scripts/fetch_public_visual_data.py      # 可选公开影像获取
scripts/build_public_visual_adapter.py   # 公共正样本 embedding adapter
scripts/train_visual_heads.py            # 命令行项目头训练入口
```

## 10. 推荐答辩表述

> 我们没有让模型把自己的预测自动当成新标签，而是把现场照片、专家判断和实验室证据保存在独立影像库里。模型对最不确定的照片优先请求人工确认；确认后的照片才进入轻量头重新训练。新版本必须在固定留出集上与当前公共基线比较，通过后还需要用户显式注册，因此整个视觉模块形成“筛查—确认—再训练—验证—版本晋级”的持续学习闭环。

## Case provenance and research loop

现场照片现在可以携带 `case_id`。从“研究与验证”生成的现场复核任务进入影像工作区后，视觉筛查先登记为独立现场证据；只有专业人员、显微镜、qPCR或毒素确认后的人工标签才适合被用户选择进入监督训练。`training_manifest_snapshot.csv` 会保留 `case_id`，因此新视觉模型版本可以追溯到哪些研究Case和确认样本，而不是用模型自己的预测结果自我训练。
