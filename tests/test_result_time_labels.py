from globalhab_demo.result_pool import beijing_time, display_record

def test_utc_day_rollover_and_legacy_times():
    assert beijing_time("2026-09-19T18:30:36+00:00") == "2026-09-20 02:30:36"
    assert beijing_time("2026-09-19T18:30:36Z") == "2026-09-20 02:30:36"
    assert beijing_time("2026-09-20T02:30:36+08:00") == "2026-09-20 02:30:36"
    assert beijing_time("2026-09-19T18:30:36") == "2026-09-20 02:30:36"
    assert beijing_time(None) == "时间未记录"

def test_legacy_science_label_preserves_data_scope():
    original = {"source":"智能研究", "evidence_type":"大模型规划与科学检验（合成）", "time":"2026-09-19T18:30:36+00:00"}
    view = display_record(original)
    assert view["证据类型"] == "大模型规划与科学检验"
    assert view["数据类型"] == "模拟数据"
    assert "（合成）" in original["evidence_type"]
