from pathlib import Path
import tempfile

from globalhab_demo.case_manager import (
    create_case_from_research, get_case, register_visual_evidence,
    register_lab_evidence, add_llm_note, case_summary, evidence_rows,
)


def test_full_case_loop():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case = create_case_from_research({
            "candidate_region": "南海测试区", "risk_score": 78.5, "forecast_window": "7–14天",
            "route": "downstream", "lag_days": 14, "model": "Logistic",
            "top_k_capacity": 0.2, "event_coverage": 0.769,
        }, root)
        assert get_case(case["case_id"], root)["status"] == "待现场复核"
        vr = {
            "screening_priority": "高", "effective_visual_category": "红棕色水体异常",
            "quality": {"quality_score": .91}, "backend": "adaptive",
            "adaptive_visual": {"active": True}, "metadata": {"location_text": "南海测试区"},
        }
        register_visual_evidence(case["case_id"], vr, vr["metadata"], root=root)
        register_lab_evidence(case["case_id"], "qPCR确认", "目标藻DNA检出", "红棕色水体异常", "Ct=24", root=root)
        add_llm_note(case["case_id"], "综合证据支持优先复核，但不能外推整个海域。", "科研结果解读", root=root)
        c = get_case(case["case_id"], root)
        assert c["status"] == "已有专业/实验室确认"
        assert len(evidence_rows(c)) == 4
        text = case_summary(c)
        assert "downstream" in text and "qPCR确认" in text and "红棕色" in text
