# GOAI 复赛代码与复现材料索引

| 复赛要求 | 对应内容 |
|---|---|
| 源代码 | `app.py`, `run_demo.py`, `src/globalhab_demo/` |
| 脚本 | `scripts/` |
| Prompt / Agent决策依据 | 核心Agent不依赖生成式Prompt；见 `prompts/README.md`, `prompts/AGENT_POLICY.md` |
| 依赖环境 | `requirements.txt`, `requirements-dev.txt`, `Dockerfile` |
| 模型或权重 | 运行时训练，不依赖预训练权重；见 `MODEL_CARD.md` |
| 配置 | `config/demo.json` |
| 最小复现 | `docs/MINIMAL_REPRODUCTION.md`, `scripts/run_minimal_reproduction.py` |
| Agent策略审计 | `scripts/run_agent_policy_benchmark.py`, `outputs/agent_policy_*` |
| 完整模型Benchmark | `scripts/run_broad_benchmark_audit.py`, `outputs/broad_benchmark_default*` |
| Florida/Gulf真实STS | `scripts/run_florida_sts_validation.py`, `src/globalhab_demo/florida_sts.py` |
| 后续出海/场站前向接口 | `scripts/run_field_forward_validation.py`, `data/field_validation/` |
| 结果产物 | `outputs/` |
| 日志 | `outputs/agent_log.csv`, `outputs/run_manifest.json`及各审计CSV/JSON |
| 数据与许可 | `DATA_DICTIONARY.md`, `THIRD_PARTY_DATA.md` |
| 开源与专利边界 | `OPEN_SOURCE_BOUNDARY.md`, `LICENSE` |
| 自动测试 | `tests/`, `.github/workflows/smoke-test.yml` |

## 评分维度对应

### 问题定义与环境设计（45%）
`README.md`, `docs/TECHNICAL_NOTE.md`, `prompts/AGENT_POLICY.md`, `src/globalhab_demo/agent.py`, `src/globalhab_demo/bayesian_design.py`。

### 探索过程与科学/研究信号（35%）
`outputs/agent_log.csv`, `outputs/discovery_card.json`, `outputs/negative_controls.csv`, `outputs/te_cte_lag_summary.csv`, `outputs/agent_policy_benchmark_default.csv`, `outputs/broad_benchmark_default.csv`, `outputs/norway_forward_benchmark_card.json`。

### 可检查性与可延续性（15%）
`outputs/run_manifest.json`, `MODEL_CARD.md`, `DATA_DICTIONARY.md`, `docs/MINIMAL_REPRODUCTION.md`, `scripts/verify_release.py`, `data/field_validation/`, `tests/`。

### 开源贡献（5%）
`LICENSE`, `OPEN_SOURCE_BOUNDARY.md`, `THIRD_PARTY_DATA.md`, 可执行源码、脚本和数据契约。
