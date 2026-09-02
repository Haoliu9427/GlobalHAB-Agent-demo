# Agent policy

GlobalHAB-Agent 的核心实验选择器不依赖生成式大模型 Prompt，也不调用外部 LLM API。实验决策由固定预算、动作空间、验证规则和反馈字段约束。

相关实现：

- `prompts/AGENT_POLICY.md`：动作空间、反馈字段、停止条件和策略定义；
- `src/globalhab_demo/agent.py`：受约束实验选择策略；
- `src/globalhab_demo/bayesian_design.py`：Bayesian Expected Improvement、Bayesian Information Gain 与 Thompson Sampling。

实际行为以源码为准。
