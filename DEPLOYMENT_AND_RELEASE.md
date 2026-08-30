# Streamlit部署与完整上传检查

## 部署前本地验收

在仓库根目录运行：

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify_release.py
python -m pytest -q
python run_demo.py --config config/demo.json
```

四条命令全部通过后再上传。

## GitHub完整性

仓库根目录必须直接看到`app.py`、`run_demo.py`、`requirements.txt`、`src`、`data`和`tests`。尤其检查：

```text
src/globalhab_demo/bio_response.py
src/globalhab_demo/real_benchmark.py
scripts/verify_release.py
.github/workflows/smoke-test.yml
.streamlit/config.toml
```

不要只覆盖`app.py`。入口文件和`src`不同步会造成`ModuleNotFoundError`。

## Streamlit Community Cloud

- Repository：当前GitHub仓库；
- Branch：`main`；
- Main file path：`app.py`；
- Python：建议3.12；
- App visibility：比赛Demo设为公开。

Python版本是在首次部署的Advanced settings中选择的。已部署App不能原地修改Python版本；如确需变更，记录仓库、入口、URL和Secrets后删除并按原URL重新部署。

## 发布后验收

1. GitHub Actions显示绿色勾；
2. Streamlit日志出现依赖安装完成且没有ImportError；
3. 无痕窗口可以打开；
4. 六个工作区均可切换；
5. 挪威真实基准显示四个时间窗；
6. 生物响应页显示81情景稳健性；
7. 下载按钮可以生成CSV/JSON；
8. 页脚显示v3.7.0；
9. 手机和桌面端均无指标截断；
10. 上传比赛前再执行一次`python scripts/verify_release.py`。

## 版本发布原则

- 每次发布只保留一个`outputs`目录；
- ZIP不包含缓存、虚拟环境、论文PDF、专利交底书或凭证；
- ZIP解压后只有一层`globalhab_agent_demo`；
- 同一次提交同时更新入口、源码、测试、VERSION和README；
- 发布文件记录SHA-256，避免浏览器同名缓存造成版本混淆。
