# 开源与知识产权边界

## 本仓库公开

- 合成数据生成器；
- 轻量Logistic/RF实验；
- 前向时间+留一海区验证；
- Agent预算搜索、随机参照与负对照；
- HAB情景地图、养殖响应优先级；
- 测试、配置、日志、发现卡和运行清单。
- 7/14/30/60天过去窗口MAD多尺度异常检测与事件合并；
- 完整度/样本/时间/空间/事件/尺度诊断驱动的自适应路由；
- 离散TE/CTE、圆周移位置换、反向路径检验和BH-FDR；
- 匿名W矩阵上的空间Durbin直接、间接和总影响分解及块Bootstrap；
- 南澳qPCR真实事件回放、来源清单和数据充分性路由；
- 挪威沿岸14年真实监测回放、环境条件摘要和全球证据接口；
- 网箱鱼生物响应沙盘的状态方程、五项干预对照、参数敏感性包络和全部原型参数；

## 与生产实现的边界

- 本仓库包含可运行的公开研究实现；
- 不逐行复刻专利生产工程的深度路由网络结构、训练目标与专用参数；
- 不公开生产级多源数据清洗、真实海流图构造、分布式计算和业务阈值；
- 生物响应沙盘不包含任何受限养殖场参数、设备能力曲线或鱼种专用死亡/生长模型；
- 公开研究实现与生产实现不要求数值一致；
- 受限真实数据或任何访问凭证。

## 第三方真实数据

`data/real_case/raw`中的Murray等人qPCR工作簿与说明，以及对应派生CSV，适用Zenodo记录20227730声明的CC BY 4.0许可，不适用本仓库MIT许可。使用时应引用原论文和Zenodo DOI。NOAA OISST适配脚本不在仓库内重新授权NOAA数据。

## 公开数据与现场接口边界

Florida/Gulf模块中的HABSOS与NOAA CoastWatch为运行时公开数据适配器，不把第三方在线数据重新授权为本仓库MIT内容，也不随包固化一个“真实验证高分”。HYCOM、Copernicus Marine和HF-radar等替代流场通过用户导出的CSV接入，各自许可与引用责任保持不变。

现场前向验证模板只定义字段协议和质量门控，不包含未来合作方的真实业务/养殖数据。用户上传的出海、场站、毒素、鱼体或设备数据不因进入GlobalHAB-Agent而自动转为开源数据。

Bayesian实验设计模块只根据已执行实验的反馈更新代理模型；合成真值仅用于轨迹结束后的评价。真实海洋中的策略有效性需另行验证。


## 现场影像甄别边界

当前发布包中的现场影像工作区使用透明的规则与颜色/纹理特征基线，用于演示“拍照—初筛—补采—复核”的工作流。项目未随包提供经过藻种/毒素金标准标注并独立验证的视觉分类权重，因此不得把该模块宣传为具体藻种识别器、毒素检测器或业务级HAB视觉预警系统。未来视觉模型需要单独的数据许可、训练、校准与时空外部验证。


## Visual backbone assets

本仓库提供DINOv2 / ConvNeXt / EfficientNet适配、自适应路由和项目训练头格式，但不把通用预训练模型或内置视觉现象原型头宣传为经过真实HAB现场照片验证的分类器。公共预训练权重由其官方库在首次使用时按原许可获取/缓存，本包不重新分发第三方权重；用户也可提供本地checkpoint。EfficientNet/ConvNeXt在完全离线时可使用明确标注为“非预训练”的确定性原型编码初始化，仅用于低成本视觉现象筛查与工程保底。

## User visual data and public visual adapters

- User-uploaded field photos are local project/session assets and are not redistributed by this release.
- The visual library can be exported/imported as a ZIP so users can persist their own labelled data when a cloud Streamlit filesystem is ephemeral.
- Third-party public photos are not bundled into the normal source archive. The repository contains a source catalog and fetch/adapter scripts; users must review the upstream record/licence/terms before redistribution.
- Public bloom-positive adapters store aggregate embedding centroids/statistics only. They do not add species or toxin labels.
- User-trained lightweight heads remain versioned under `vision_models/user_models/`; a failed or rejected candidate is retained for audit and does not overwrite the active model.

## Case / Evidence Ledger boundary

`data/cases/cases.json` 只负责把研究候选、现场视觉、专业/实验室确认和解释记录串成可追溯Case。后续证据采用追加式登记，不覆盖原研究指标。视觉筛查不自动升级为HAB真值，大模型文本也不改变证据等级；真实确认仍以相应采样、检测方法和检测限为准。
