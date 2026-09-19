# 平台模型配置（管理员一次设置）
这是 HF3.8 预览版。访客选择模型，不填写密钥。没有真实凭证时，界面不显示虚构的可用模型。

已有网站 [qwen] 的 base_url、model、api_key 配置可直接读取，不需要更换格式。必须是实际在线推理服务，模型下载目录不等于推理服务。

多个模型可在 Streamlit Secrets 增加如下条目，每个模型重复一段：

```toml
[[platform_models]]
name = "Qwen · 研究助手"
base_url = "https://api-inference.modelscope.cn/v1"
model = "Qwen/Qwen3.5-35B-A3B"
api_key = "管理员填写真实Token"
enabled = true
```

不要把真实 Secrets 提交到 GitHub 或部署包。ModelScope 模型权限和额度以账号为准：
https://www.modelscope.cn/docs/model-service/API-Inference/intro

配置后：选择模型 → 检查模型连接 → 开始规划与科学检验。连接检查会真正请求一次简短模型回复，但不代表已验证科学规划能力；须完成候选、两种负对照和冻结后的独立测试。

本地预览不会自动同步云端 Secrets。自选服务入口独立使用访客配置；规则入口不调用大模型。

本版保留图像识别等工作区的依赖，不等于整个工程不再需要 torch。平台规划本身不下载模型权重。

## 2026-09-20 真实调用验证
已通过本地 Streamlit 页面调用魔搭托管 Qwen/Qwen3.5-35B-A3B，完成 6 次规划调用、4 次科学实验与最终独立测试。候选 AP 0.127，季节参照 AP 0.176，保留负结果。该结果仅说明流程可运行，不证明预测改进。尚未在 Streamlit Cloud 部署验证。
