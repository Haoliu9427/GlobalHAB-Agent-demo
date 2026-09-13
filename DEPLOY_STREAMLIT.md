# Streamlit部署说明

本ZIP解压后即为仓库根目录：app.py、requirements.txt、src、data、outputs应在同一层。上传解压后的内容，不要只上传ZIP，也不要只替换app.py。

1. 将完整内容同步到Streamlit绑定的GitHub仓库及分支。
2. 确认GitHub根目录app.py包含“API工作台 2026.09.13”。
3. 确认src/globalhab_demo/real_training/user_workbench.py包含DeepSeek、自行填写。
4. Streamlit部署入口保持app.py，等待依赖安装与重启；必要时在管理页面重启应用。
5. 侧栏应显示GlobalHAB-Agent · API工作台 2026.09.13。
6. 工作区 → 自有数据分析 → 远程大模型服务 → 自行填写。模型服务提供DeepSeek、Qwen、自定义兼容服务。

不要求为远程API安装torch。用户填写有效API地址、模型ID和密钥后测试连接，再选择远程大模型 · API预测。真实服务尚未实测；测试连接可能计费。其他本地深度模型仍需要其可选依赖。

如果GitHub没有变化，请检查上传目录/分支是否正确；若GitHub已有上述代码而页面版本未变化，请检查Streamlit绑定的仓库、分支和入口。勿把真实密钥提交到GitHub。
