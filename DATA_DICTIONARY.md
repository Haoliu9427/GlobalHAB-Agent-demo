# 合成示例数据字典

所有数值均由固定随机种子生成，仅用于验证程序能否恢复预设的“沿抽象传导路径、滞后14天”信号，不代表任何真实海区。

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

网页地图使用北阿拉斯加湾、加州沿岸、墨西哥湾、北大西洋中部和日本黑潮延伸区作为粗略演示锚点。`综合风险指数`与`预计藻华强度指数`均为0–100无量纲界面展示量，仅用于比较同一次假设情景下的空间相对差异。二者不等同于校准后的事件概率、藻细胞密度、叶绿素浓度或毒素浓度，也不能用于现场决策。
