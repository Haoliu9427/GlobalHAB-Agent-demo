# 数据与模型来源

1. NOAA NCEI, Harmful Algal Blooms Observing System: https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system 。本次服务快照： https://gis.ncdc.noaa.gov/arcgis/rest/services/ms/HABSOS_CellCounts/MapServer/0/query 。公开数据保留NOAA及原贡献机构归属；不能把项目代码许可证套用于第三方资料。
2. 香港政府开放数据，Red Tide Location: https://data.gov.hk/en-data/dataset/hk-afcd-afcdlist-red-tide-location 。CSV： https://redtide.afcd.gov.hk/data/RTMS_ob_RTLE.csv 。数据归属香港渔农自然护理署，使用应遵守香港政府开放数据使用条款并标注来源。
3. AFCD赤潮数据库介绍： https://www.afcd.gov.hk/english/fisheries/hkredtide/database/database.html 。档案是报告事件资料，不能提供每个无记录位置的监测阴性。
4. Chronos官方代码： https://github.com/amazon-science/chronos-forecasting 。模型卡： https://huggingface.co/amazon/chronos-bolt-small 。下载与再分发遵循模型卡及仓库许可；本包只含调用代码、模型ID和完成实验的revision，不内置预训练权重。
5. Qwen模型卡： https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct 。约0.49B参数，Apache-2.0。保留实际revision与固定提示词，未对该模型做HAB微调；不内置权重。

原挪威、南澳及流场来源与引用继续见THIRD_PARTY_DATA.md。数据规模与日期以raw/snapshot_manifest.json为准；模型结果以已完成实验文件为准。
