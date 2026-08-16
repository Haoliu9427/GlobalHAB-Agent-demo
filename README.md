# GlobalHAB-Agent：初赛最小可运行示例

本仓库用于展示“观察—提出假设—执行实验—获取反馈—更新搜索—形成发现卡”的最小闭环。它使用具有海洋科学约束的确定性合成数据，不声称已经获得真实全球有害藻华预测性能，也不用于业务预警。

## 30 秒运行

```bash
python -m pip install -r requirements.txt
python run_demo.py --config config/demo.json
```

运行完成后生成：

- `outputs/agent_log.csv`：每一步的观察、行动、反馈和预算；
- `outputs/discovery_card.json`：最佳候选假设、验证指标及适用边界；
- `outputs/risk_predictions.csv`：留出时段和留出海区的示例风险概率；
- `outputs/run_summary.md`：可直接查看的试跑摘要。

## 本示例证明什么

1. Agent 能在固定预算内选择“局地/沿流 × 7/14/30 d × 两类轻量模型”的候选实验；
2. 每个实验使用同一套前向时间与留一海区验证，避免随机划分造成的时空泄漏；
3. Agent 根据验证反馈调整后续实验优先级，并保留所有失败结果；
4. 最终输出结构化发现卡和逐样本风险概率，可复核每一步来自哪一项实验。
5. 网页将假设条件转换为可交互的空间预警示意，直观回答“什么条件、什么时间、哪里、相对多强”。

合成字段包括SST、季节气候态、MHW阈值与强度、硝酸盐、磷酸盐、硅酸盐和微塑料浓度。MHW强度在超过合成p90阈值时按`SST − climatological mean`计算；微塑料仅作为输运、停留和汇聚背景代理，用于传导路径加权，不作为流速/流向或HAB直接驱动。详见`DATA_DICTIONARY.md`。

## 本示例不证明什么

- 合成数据结果不是现实海区的 HAB 性能；
- 示例区域均为匿名合成区域，“沿流”是抽象有向关系，不等同于真实海流轨迹；
- 情景地图中的五个命名海区是粗略展示锚点，风险和强度均为0–100无量纲指数，不是事件概率、藻细胞密度、叶绿素或毒素浓度；
- 该示例不包含完整多尺度异常检测、深度自适应路由、TE/CTE 网络和空间乘数影响分解；
- 真实项目仍需接入卫星、再分析、毒素、物种、闭港及监测努力数据，并进行独立事件复核。

## 目录

```text
globalhab_agent_demo/
├── config/demo.json
├── data/                       # 运行后生成合成示例数据
├── outputs/                    # 运行后生成验证结果
├── src/globalhab_demo/
│   ├── agent.py                # 预算约束的假设搜索 Agent
│   ├── data.py                 # 合成数据生成
│   └── experiment.py           # 特征构造、模型训练与阻断验证
├── tests/test_smoke.py
├── app.py                      # Streamlit网页入口
├── run_demo.py
├── requirements.txt
├── Dockerfile
├── DATA_DICTIONARY.md
├── DEPLOY_STREAMLIT.md
└── OPEN_SOURCE_BOUNDARY.md
```

## 一键 Docker 试跑

```bash
docker build -t globalhab-demo .
docker run --rm -p 8501:8501 globalhab-demo
```

随后访问`http://localhost:8501`。该地址仅用于本机预览。
