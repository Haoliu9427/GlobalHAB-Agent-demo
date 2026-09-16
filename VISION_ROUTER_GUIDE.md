# GlobalHAB-Agent 自适应现场影像甄别路由

## 1. 模块定位

现场影像工作区采用 **Adaptive Visual Screening Router (AVSR)**：

```text
图像质量门控
  ↓
自适应路由
  ├─ EfficientNet-B0：颜色异常明确的轻量路径
  ├─ ConvNeXt-Tiny：浮沫、浑浊与复杂表层纹理路径
  └─ DINOv2-small：不确定/复杂场景的通用表征路径
  ↓
视觉现象筛查头（项目训练头优先；否则内置原型头）
  ↓
12维现场元数据融合
  ↓
entropy / margin / disagreement / OOD
  ↓
低 / 中 / 高复核优先级，或 DEFER
```

输出描述**视觉现象与复核优先级**，不是HAB发生概率、具体藻种或毒素结论。

## 2. 深度分支现在如何真正运行

### EfficientNet-B0 与 ConvNeXt-Tiny

运行顺序为：

1. 若存在本地checkpoint，优先读取本地文件；
2. 否则默认尝试获取并缓存Torchvision公共预训练权重；
3. 如果部署环境完全离线且下载失败，默认启用**确定性离线原型编码初始化**，保证深度网络真实完成前向传播；
4. 若设置 `GLOBALHAB_STRICT_PRETRAINED=1`，则禁止第3步，预训练权重不可用时直接DEFER。

离线原型编码初始化不是预训练模型，因此在结果中会显式标记“非预训练”，其深度特征只作为内置原型筛查的一部分，不作为已验证模型性能证据。

### DINOv2-small

DINOv2使用 `facebook/dinov2-small`。优先读取 `vision_models/dinov2_local/` 或 `GLOBALHAB_DINOV2_MODEL_DIR` 指定目录；若允许下载，则首次使用时由Transformers获取并缓存公共模型。DINOv2没有随机权重冒充方案：依赖/权重不可用时，该分支不参与结果。

## 3. 视觉现象筛查头

系统有两级头：

### 3.1 项目训练头（优先）

若 `vision_models/heads/*.npz` 存在，则使用真实标注照片训练得到的线性融合头。训练头接收：

```text
视觉 embedding + 12维现场元数据
```

并支持基于训练特征分布的OOD距离。

### 3.2 内置视觉现象原型头（默认可用）

没有项目训练头时，不再显示“未配置”。系统会生成固定、可审计的水面视觉原型（正常蓝/蓝绿、绿色异常、红棕异常、浑浊泥沙、表层浮沫），通过当前深度编码器计算prototype embedding，再将：

- 深度embedding与prototype的余弦相似度；
- 透明颜色/纹理规则；
- 现场元数据先验；

融合为视觉现象筛查分数。该头用于让深度路由真实执行和提供低成本视觉初筛，**没有用真实HAB照片校准，因此不能表述为“训练好的HAB分类器”**。

## 4. 自适应路由

- 绿色/红棕颜色信号明确：优先 EfficientNet-B0；
- 浮沫、漂浮物、浑浊、复杂纹理：优先 ConvNeXt-Tiny；
- 类别不明确或复杂场景：优先 DINOv2；若DINOv2当前不可用，则选择下一可执行分支；
- 接近决策边界时增加第二分支做一致性复核；
- 多模型一致性模式会调用所有当前可运行的分支。

只有真实执行过forward的分支才会出现在“已激活分支”中。

## 5. 图像质量门控

在任何深度模型之前检查：分辨率、欠曝、过曝、反光、对比度、锐度/纹理，以及用户是否确认海面是照片主体。质量失败直接 `DEFER / 需重拍`。

## 6. 现场元数据融合

固定编码12项：异味、泡沫、浮膜、漂浮物、近期高温、异常水色、鱼贝异常/死亡、水温、盐度、DO、Chl-a、仪器变量缺失比例。

## 7. 不确定性与DEFER

深度分支运行后计算：

- 预测熵 entropy；
- Top1/Top2 margin；
- 多分支 disagreement；
- OOD（项目训练头使用训练分布距离；公共/自监督预训练+原型头使用prototype相似度）。

公共预训练/项目训练模型默认阈值：entropy > 0.72、margin < 0.12、disagreement > 0.28 或任一OOD。完全离线的确定性原型编码初始化因为不是学习得到的视觉专家，使用更宽松的 entropy/margin 工程阈值，但仍保留质量门控和多分支冲突DEFER。

若自适应模式没有任何深度分支真正执行，最终结果为：

```text
DEFER / 视觉引擎未就绪
```

透明规则基线仍可显示辅助线索，但不会冒充深度模型最终结论。

## 8. Streamlit界面

普通用户默认只看到拍照/上传、现场信息和最终结果。模型工程状态被收进折叠的 **“高级视觉模型信息”**。表格现在显示：编码器来源、筛查头类型、路由状态和简要说明，不再因为缺少项目训练头而把三条分支全部标成“未配置”。

首屏不再展示长串内部算法说明。

## 9. 运行与部署

主 `requirements.txt` 已包含：

```text
torch
torchvision
transformers
safetensors
```

默认：

```bash
GLOBALHAB_ALLOW_MODEL_DOWNLOADS=1
GLOBALHAB_STRICT_PRETRAINED=0
```

若比赛环境要求完全冻结且只接受已缓存/本地公共预训练权重：

```bash
GLOBALHAB_ALLOW_MODEL_DOWNLOADS=0
GLOBALHAB_STRICT_PRETRAINED=1
```

## 10. 项目训练头

获得真实标注现场照片后，可继续训练项目头：

```bash
python scripts/train_visual_heads.py --manifest your_manifest.csv --backbone efficientnet
python scripts/train_visual_heads.py --manifest your_manifest.csv --backbone convnext
python scripts/train_visual_heads.py --manifest your_manifest.csv --backbone dinov2
```

项目训练头一旦存在，会自动覆盖同分支的内置原型头。正式科学性能报告仍应采用独立站点/年份留出。

## 11. 与大模型结果解读连接

现场影像结果可以一键送入“大模型结果解读”。只发送结构化文字摘要，不自动发送原始照片。摘要包括实际执行的backbone、编码器来源、筛查头类型、entropy、margin、disagreement、OOD和DEFER原因。

## 12. 决赛答辩推荐表述

> 现场视觉层不是用一张手机照片替代实验室鉴定。系统先做图像质量门控，再根据颜色、纹理和场景复杂度自适应选择EfficientNet、ConvNeXt或DINOv2。项目训练头存在时优先使用；没有项目训练头时，深度编码器仍真实执行，并通过内置视觉现象原型头与现场元数据形成低成本筛查结果。最后由不确定性和DEFER决定是否值得进入qPCR、显微镜或毒素等进一步复核。

## 13. 持续学习与活动模型

路由器不再固定读取 `vision_models/heads/*.npz`。它会优先读取：

```text
vision_models/active_model.json
```

若当前版本是用户模型，则自动解析：

```text
vision_models/user_models/<active_version>/heads/<backbone>.npz
```

因此用户在“模型训练与版本”子页注册新版本后，后续现场甄别无需修改代码即可调用新头。切换回 `public-baseline-v1` 即恢复公共基线/内置原型头。

公开影像适配器（如果用户运行公开数据获取与适配脚本）位于：

```text
vision_models/public_baseline/adapters/
```

它只对“与公开已核验表层藻华照片的视觉表征相似度”提供弱支持，不增加具体藻种或毒素标签。
