"""Curated external-event evidence used only for retrospective validation."""

SOUTH_AUSTRALIA_CASE = {
    "case_id": "SA_KARENIA_2025_2026",
    "title": "南澳大利亚 Karenia 复合藻华事件",
    "evidence_grade": "A",
    "period": "2025年3月起，持续超过12个月",
    "spatial_extent": "约20,000 km²",
    "confirmed_signals": [
        "Karenia cristata 在采样区占优势，并与另外四种 Karenia 共存",
        "定向qPCR、宏条形码、长读长测序与毒素分析形成多源证据链",
        "检测到BTX-2、BTX-3和BTX-B5等brevetoxins",
        "水动力过程被认为可能将细胞向半封闭近岸海域输送",
    ],
    "reported_impacts": [
        "约10^6只海洋动物、超过600个分类群死亡",
        "出现人类健康影响",
        "贝类生产受到毒素监测和收获限制影响",
    ],
    "aquaculture_interpretation": (
        "该案例证明预警不能只预测“是否水华”，还需区分藻种/毒素、"
        "水动力输运、养殖对象暴露与现场复核。它不提供可直接迁移到全球的统一阈值。"
    ),
    "model_use": (
        "仅作为外部事件证据卡和验证接口示例，不参与合成模型训练，"
        "不用于声称真实世界预测性能。"
    ),
    "sources": [
        {
            "label": "Murray et al., Nature Ecology & Evolution (2026)",
            "url": "https://doi.org/10.1038/s41559-026-03115-0",
        },
        {
            "label": "配套开放数据（Zenodo）",
            "url": "https://doi.org/10.5281/zenodo.20227730",
        },
        {
            "label": "Ruvindy et al., Environmental Science & Technology (2024)",
            "url": "https://doi.org/10.1021/acs.est.3c10502",
        },
    ],
}
