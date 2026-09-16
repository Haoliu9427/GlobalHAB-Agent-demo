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


### 现场影像自适应路由

新增的现场影像模块采用“质量门控—自适应路由—视觉特征—视觉现象筛查头—现场元数据—不确定性/DEFER”结构，并支持 DINOv2、ConvNeXt、EfficientNet 三类视觉 backbone。项目真实训练头存在时优先使用；否则由内置视觉现象原型头保证深度路由可实际运行，同时明确其不是经过真实HAB现场照片校准的分类器。EfficientNet/ConvNeXt在离线环境也有显式标注的原型编码保底路径；没有任何深度分支成功执行时自适应模式直接DEFER。完整方法与训练接口见 `VISION_ROUTER_GUIDE.md`。

## Case证据闭环

决赛版新增统一Case：研究与验证可把 Risk / Route / Lag / Top-k 候选生成现场复核任务；现场影像甄别补充视觉与环境证据；专业/实验室确认后的照片可进入持续学习视觉训练库；大模型可对当前完整Case做综合解读并把解释保存回Case。各证据层保持独立，不用视觉或大模型结果覆盖原研究结论。
