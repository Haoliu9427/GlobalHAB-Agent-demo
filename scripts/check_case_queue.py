"""Non-UI smoke check for batch Case creation and lifecycle management."""
from pathlib import Path
import json
import tempfile

from globalhab_demo.case_manager import (
    archive_case,
    bulk_create_cases,
    cancel_case,
    case_status_code,
    mark_case_in_progress,
    next_review_case,
    restore_case,
)


def research(region: str, risk: float):
    return {
        "source": "queue-smoke",
        "candidate_region": region,
        "risk_score": risk,
        "issue_date": "2026-09-16",
        "forecast_window": "7–14天",
        "horizon_days": 14,
        "route": "downstream",
        "lag_days": 14,
        "scenario": {"mhw_intensity_c": 2.8},
    }


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    created, skipped = bulk_create_cases([research("A", 82), research("B", 76)], root)
    first, second = created
    mark_case_in_progress(first["case_id"], root)
    nxt = next_review_case(first["case_id"], root)
    cancel_case(second["case_id"], root=root)
    cancelled = case_status_code(__import__("globalhab_demo.case_manager", fromlist=["get_case"]).get_case(second["case_id"], root))
    restore_case(second["case_id"], root)
    archive_case(second["case_id"], root=root)
    archived = case_status_code(__import__("globalhab_demo.case_manager", fromlist=["get_case"]).get_case(second["case_id"], root))
    created2, skipped2 = bulk_create_cases([research("A", 82)], root)
    assertions = {
        "bulk_created": len(created) == 2 and not skipped,
        "next_task": bool(nxt and nxt["case_id"] == second["case_id"]),
        "cancelled": cancelled == "cancelled",
        "archived": archived == "archived",
        "deduplicated": len(created2) == 0 and len(skipped2) == 1,
    }
    payload = {"status": "pass" if all(assertions.values()) else "fail", "assertions": assertions}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)
