# GlobalHAB-Agent 自适应现场影像甄别路由

## 1. 模块定位

决赛工程在原“现场影像甄别”工作区上增加 **Adaptive Visual Screening Router (AVSR)**。它不是把手机照片包装成“藻种识别器”，而是把低成本影像证据组织成可检查的分层流程：

```text
图像质量门控
   ↓
自适应路由
   ├─ EfficientNet-B0：清晰颜色异常的轻量快速路径
   ├─ ConvNeXt-Tiny：浮沫、浑浊、复杂表层纹理路径
   └─ DINOv2：不确定/跨场景通用表征路径
   ↓
冻结视觉特征 + 轻量线性分类头
   ↓
现场元数据融合
   ↓
类别不确定性 + 分支一致性 + OOD
   ↓
低 / 中 / 高复核优先级，或 DEFER
```

输出仍然描述**视觉现象与复核优先级**，不是 HAB 发生概率，也不是藻种或毒素确诊。

## 2. 为什么采用自适应路由

不同照片的信息结构不同：

- 颜色变化明显、场景简单时，优先走 EfficientNet，减少不必要计算；
- 浮沫、漂浮物、浑浊和表面纹理复杂时，优先 ConvNeXt；
- 类别不明确、场景复杂或接近决策边界时，优先 DINOv2，并在条件允许时增加第二分支做一致性复核；
- 图像质量不合格时，质量门控直接 `DEFER / 需重拍`，不调用更复杂模型强行判断。

这与 GlobalHAB-Agent 原有“有限预算、按证据状态选择下一步实验”的思想一致：不是每张图都跑所有模型，而是让输入状态决定更合适的计算路径。

## 3. 三类视觉特征提取器

### EfficientNet-B0

- 默认作为轻量快速路径；
- 适合颜色异常较清晰、主体明确的照片；
- 本地 checkpoint 默认路径：`vision_models/efficientnet_b0.pth`；
- 也可设置环境变量 `GLOBALHAB_EFFICIENTNET_CHECKPOINT`。

### ConvNeXt-Tiny

- 优先处理纹理复杂、浮沫/漂浮物和浑浊样场景；
- 本地 checkpoint 默认路径：`vision_models/convnext_tiny.pth`；
- 也可设置 `GLOBALHAB_CONVNEXT_CHECKPOINT`。

### DINOv2

- 用作通用视觉表征与不确定场景复核；
- 默认读取 Hugging Face 兼容的本地目录 `vision_models/dinov2_local/`；
- 也可设置 `GLOBALHAB_DINOV2_MODEL_DIR`。

默认部署**不会联网下载任何权重**。开发环境如确需在线下载，可显式设置：

```bash
GLOBALHAB_ALLOW_MODEL_DOWNLOADS=1
```

生产/比赛发布建议继续使用冻结的本地权重与哈希审计。

## 4. 轻量分类器

深度视觉 backbone 只负责提取冻结特征，真正输出项目视觉类别的是一个小型多项线性头。运行时不使用 `pickle/joblib`，而是读取安全的 NumPy `.npz` 参数：

```text
vision_models/heads/
  efficientnet.npz
  convnext.npz
  dinov2.npz
```

每个头将：

```text
视觉 embedding + 12维现场元数据
```

拼接后做标准化与线性分类。

**只有“编码器 + 用项目标注数据训练得到的轻量头”同时存在时，该深度分支才会参与最终结果。** 如果只有通用 DINOv2 / ImageNet backbone，却没有现场标注数据训练头，系统会明确显示“未配置”，不会把通用特征伪装成已经验证的 HAB 分类器。

## 5. 现场元数据融合

当前固定编码 12 项：

- 异味；
- 泡沫；
- 浮膜；
- 漂浮物；
- 近期高温；
- 肉眼异常水色；
- 鱼贝异常/死亡；
- 水温；
- 盐度；
- DO；
- Chl-a；
- 仪器变量缺失比例。

数值变量只做透明缩放，缺失值保留为零并通过“缺失比例”显式记录。这样既能融合现场信息，也避免把缺失值悄悄当成正常值。

## 6. 不确定性与 DEFER

深度分支激活后同时计算：

1. **预测熵 entropy**：分布越平，越不确定；
2. **Top-1 / Top-2 margin**：第一和第二候选越接近，越不稳定；
3. **branch disagreement**：多分支输出差异；
4. **OOD**：输入特征相对训练特征标准化距离是否超过训练期阈值。

默认触发条件：

- entropy > 0.72；
- margin < 0.12；
- disagreement > 0.28；
- 任一分支触发 OOD。

任一条件成立，工作区输出：

```text
DEFER / 需人工复核
```

而不是强制给出高/中/低。

这些阈值是工程默认值，获得足够现场标注数据后应在独立验证集上重新校准。

## 7. 图像质量门控

质量门控仍然在所有深度模型之前，包括：

- 分辨率；
- 亮度/欠曝；
- 过曝；
- 强反光；
- 对比度；
- 锐度/纹理代理；
- 用户是否确认海面是照片主体。

不合格图像先 `DEFER / 需重拍`，避免“更复杂模型”掩盖输入质量问题。

## 8. 默认包为何不附带训练好的视觉头

当前项目没有一套已完成独立时间/地点验证、并具有 qPCR/显微镜/专家标签对应关系的海面照片训练集。因此发布包不会生成伪标签、合成一个“高准确率”视觉模型或把 ImageNet 模型直接称为 HAB 分类器。

工程已经提供完整接口和训练脚本；获得真实标注数据后再训练：

```bash
pip install -r requirements-vision.txt
python scripts/train_visual_heads.py \
  --manifest data/field_visual/your_manifest.csv \
  --backbone efficientnet
```

可分别训练：

```bash
--backbone efficientnet
--backbone convnext
--backbone dinov2
```

训练脚本会输出简单留出诊断、`.npz` 分类头和指标 sidecar。正式科研使用仍建议采用独立站点/年份留出，而不是随机图像切分。

## 9. Streamlit 界面

现场影像工作区现在支持：

- `自适应路由（推荐）`；
- `多模型一致性`；
- 指定 `EfficientNet`；
- 指定 `ConvNeXt`；
- 指定 `DINOv2`；
- `安全规则基线`。

“自适应视觉路由与深度模型状态”面板会实时显示三个分支的：

- 编码器是否可用；
- 训练头是否可用；
- 是否允许参与路由；
- 当前回退原因。

完成甄别后还会显示路由类型、已激活分支、各分支视觉类别、类别置信度、entropy、margin、OOD 与融合 disagreement。

## 10. 与大模型结果解读连接

现场影像结果写入 Streamlit `session_state` 后仍可“一键送入大模型结果解读”。发送给远程大模型的仍是结构化文字摘要，而不是原始照片。摘要新增：

- 自适应路由分支；
- 深度视觉是否真正激活；
- 不确定性；
- OOD / DEFER 原因。

大模型提示词继续禁止把视觉类别置信度改写为 HAB 概率、具体藻种或毒素结论。

## 11. 依赖与测试

基础工作区（无深度权重）仍只需要主 `requirements.txt`。

需要 DINOv2 / ConvNeXt / EfficientNet 时额外安装：

```bash
pip install -r requirements-vision.txt
```

测试：

```bash
pytest -q tests/test_field_visual.py tests/test_adaptive_visual.py
python scripts/check_adaptive_visual.py
python scripts/check_field_visual.py
python scripts/verify_release.py
```

## 12. 决赛答辩推荐表述

> 现场视觉层不是让一张手机照片替代实验室鉴定。系统先做图像质量门控，再根据颜色、纹理和场景复杂度自适应选择 EfficientNet、ConvNeXt 或 DINOv2；视觉特征与现场温盐、DO、Chl-a 和人工观察融合后，再通过熵、模型一致性和 OOD 检查决定给出复核优先级还是 DEFER。当前发布包没有伪造一个未经现场标注验证的深度视觉模型，因此没有训练头时会自动回退透明基线；得到真实标注照片后，可以直接在现有框架中训练和替换分类头。
