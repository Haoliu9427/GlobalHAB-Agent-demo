# Streamlit部署说明

本ZIP解压后即为仓库根目录：app.py、requirements.txt、src、data、outputs应在同一层。上传解压后的内容，不要只上传ZIP，也不要只替换app.py。

1. 将完整内容同步到Streamlit绑定的GitHub仓库及分支。
2. 确认GitHub根目录app.py包含“2026.09.14”。
3. 确认src/globalhab_demo/real_training/user_workbench.py包含DeepSeek、自行填写。
4. Streamlit部署入口保持app.py，等待依赖安装与重启；必要时在管理页面重启应用。
5. 侧栏应显示GlobalHAB-Agent · 2026.09.14。
6. 工作区 → 自有数据分析 → 远程大模型服务 → 自行填写。模型服务提供DeepSeek、Qwen、自定义兼容服务。

不要求为远程API安装torch。用户填写有效API地址、模型ID和密钥后测试连接，再选择远程大模型 · API预测。真实服务尚未实测；测试连接可能计费。其他本地深度模型仍需要其可选依赖。

如果GitHub没有变化，请检查上传目录/分支是否正确；若GitHub已有上述代码而页面版本未变化，请检查Streamlit绑定的仓库、分支和入口。勿把真实密钥提交到GitHub。

本版界面样式文件assets/interface.css必须与app.py一并上传。字体优先微软雅黑；设备未安装时使用系统中文后备字体，不包含字体授权文件。


## Adaptive vision runtime

主 `requirements.txt` 已包含 PyTorch / Torchvision / Transformers 视觉运行依赖。EfficientNet-B0 与 ConvNeXt-Tiny优先使用本地或公共预训练权重；首次使用可自动缓存官方权重，完全离线时可使用明确标记的确定性原型编码初始化。DINOv2-small需要Transformers与本地缓存或首次下载。项目 `.npz` 训练头存在时优先使用，否则使用内置视觉现象原型头。若希望比赛部署严格禁止任何非预训练离线路径，请设置 `GLOBALHAB_STRICT_PRETRAINED=1` 并提前缓存/配置权重。详见 `VISION_ROUTER_GUIDE.md`。

## Persistent field-photo learning data

The continuous-learning visual workspace writes user assets under `data/field_visual/user_library/` and model versions under `vision_models/user_models/`. On hosting products whose local filesystem is ephemeral, users should export the visual-library ZIP and model-version ZIP after a session and re-import them when needed. For a durable production deployment, mount these directories on persistent storage rather than relying on the application container filesystem.

## Case任务队列持久化

批量现场复核任务保存在 `data/cases/cases.json`。若部署平台使用临时文件系统，应用重启后本地Case状态、用户影像库和用户模型版本可能丢失；正式持续运行时建议挂载持久卷或将这些目录映射到持久化存储。取消和归档不会删除证据，永久删除需要用户确认。
