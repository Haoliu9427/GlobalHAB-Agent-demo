# Prompt applicability

GlobalHAB-Agent 的核心实验选择器不依赖生成式大模型 Prompt，也不调用外部 LLM API。核心 Agent 是受固定预算、动作空间、验证规则与反馈字段约束的可审计实验决策器，因此不存在未公开的 System Prompt、API Key 或隐藏自然语言指令。

复赛要求中的“Prompt/Agent决策依据”通过以下文件提供：

- `prompts/AGENT_POLICY.md`：人类可读的动作空间、固定规则、反馈、停止条件与 Bayesian 策略审计说明；
- `src/globalhab_demo/agent.py`：当前透明受约束策略的可执行实现；
- `src/globalhab_demo/bayesian_design.py`：Bayesian Expected Improvement、Bayesian Information Gain 与 Thompson Sampling 的可执行实现。

这些文件是审计和复现材料，不是另一个隐藏 Prompt。实际行为以源码为准。
