# 最小可运行探索环境

本目录对应 GOAI AI for Research 开放探索赛题复赛材料 3：**最小可运行探索环境**。

## 最短运行

```bash
python -m pip install -r requirements.txt
python scripts/run_minimal_reproduction.py
```

启动网页：

```bash
streamlit run app.py
```

一键 smoke test：

```bash
python scripts/smoke_test.py
```

最小复现不依赖 NOAA / HYCOM 在线服务。Florida/Gulf 在线回顾模块属于扩展真实数据验证；外部服务不可用不影响核心环境安装和运行。
