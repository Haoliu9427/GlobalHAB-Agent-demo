# 最小复现说明

## 1. 安装

```bash
python -m pip install -r requirements.txt
```

推荐 Python 3.12。

## 2. 核心Agent闭环

```bash
python scripts/run_minimal_reproduction.py
```

该命令只重建24个候选实验并运行8步受约束Agent，不重跑TE/CTE、Durbin、真实事件回放和生物响应沙盘。典型CPU环境约十余秒到数十秒，具体取决于硬件。

输出：

- `outputs/minimal_agent_log.csv`
- `outputs/minimal_reproduction_card.json`

重点检查：24候选、8步预算、完全留区 + 前向时间阻断、默认配置恢复预注册downstream/14天模式、equal-budget random参照和负对照。

## 3. Agent策略Benchmark

```bash
python scripts/run_agent_policy_benchmark.py
```

在同一24动作和8步预算下比较当前策略、Bayesian EI、Bayesian Information Gain、Thompson Sampling和Random。隐藏14天真值只在轨迹结束后评分，不进入采集函数。

## 4. 完整模型Benchmark

```bash
python scripts/run_broad_benchmark_audit.py
```

所有模型使用相同完全留出海区和前向测试行；需要模型选择的方法只在外层训练期内部完成。

## 5. Florida/Gulf真实流场回顾验证

在线模式：

```bash
python scripts/run_florida_sts_validation.py --online
```

若公开服务暂时不可用，可导出HABSOS观测和HYCOM/Copernicus/HF-radar流场CSV后使用脚本上传参数运行。Florida结果不随包硬编码。

## 6. 现场前向验证

```bash
python scripts/run_field_forward_validation.py \
  --observations <field_observations.csv> \
  --currents <field_currents.csv>
```

模板位于 `data/field_validation/`。系统先做质量检查；通过后只用早期时间块选择lag，后期时间块一次性评价。数据不足返回DEFER。

## 7. 发布检查

```bash
python scripts/verify_release.py
python -m pytest -q tests/test_release_smoke_fast.py tests/test_bayesian_design.py tests/test_florida_sts.py tests/test_broad_benchmark.py
```

GitHub Actions使用相同的快速release smoke tests。完整 `python -m pytest -q` 可用于离线完整测试。

## 8. 启动网页

```bash
streamlit run app.py
```
