"""Non-UI smoke check for the cross-workspace Case/evidence loop."""
from pathlib import Path
import json
import tempfile

from globalhab_demo.case_manager import (
    create_case_from_research, get_case, register_visual_evidence,
    register_lab_evidence, add_llm_note, case_summary, evidence_rows,
)

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    case = create_case_from_research({
        "candidate_region": "南海测试区", "risk_score": 81.2,
        "forecast_window": "14天", "route": "downstream", "lag_days": 14,
        "model": "Logistic", "top_k_capacity": 0.2, "event_coverage": 0.769,
    }, root)
    result = {
        "screening_priority": "高", "effective_visual_category": "红棕色水体异常",
        "quality": {"quality_score": 0.92}, "backend": "adaptive visual router",
        "adaptive_visual": {"active": True, "uncertainty": {"defer": False}},
        "metadata": {"location_text": "南海测试区", "dissolved_oxygen_mg_l": 4.4},
    }
    register_visual_evidence(case["case_id"], result, result["metadata"], root=root)
    register_lab_evidence(case["case_id"], "qPCR确认", "目标藻DNA检出", "红棕色水体异常", "Ct=24", root=root)
    add_llm_note(case["case_id"], "综合证据支持优先复核，仍需保留空间外推边界。", "科研结果解读", root=root)
    final = get_case(case["case_id"], root)
    assertions = {
        "case_created": bool(final),
        "research_candidate": final["research"]["route"] == "downstream",
        "visual_registered": final["visual"]["visual_category"] == "红棕色水体异常",
        "lab_registered": len(final["lab_evidence"]) == 1,
        "llm_saved": len(final["llm_notes"]) == 1,
        "ledger_layers": len(evidence_rows(final)) == 4,
        "summary_complete": all(x in case_summary(final) for x in ["Route=downstream", "红棕色水体异常", "qPCR确认"]),
    }
    payload = {"status": "pass" if all(assertions.values()) else "fail", "assertions": assertions, "case_id": case["case_id"]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)
