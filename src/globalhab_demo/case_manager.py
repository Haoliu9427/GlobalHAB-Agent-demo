"""Cross-workspace case and evidence ledger for GlobalHAB-Agent.

A Case is the shared unit that links:
research risk candidate -> field visual screening -> human/lab confirmation ->
visual learning -> LLM interpretation.

The ledger is intentionally additive. Visual screening and LLM notes never
rewrite upstream research metrics or promote a case to a confirmed HAB event.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import tempfile


CASE_SCHEMA_VERSION = "1.1"

CASE_STATUS_LABELS = {
    "pending_review": "待现场复核",
    "in_progress": "现场复核中",
    "visual_screened": "已完成视觉筛查",
    "visual_defer": "视觉DEFER",
    "lab_pending": "待实验室确认",
    "confirmed": "已有专业/实验室确认",
    "cancelled": "已取消复核",
    "archived": "已归档",
}
QUEUE_STATUS_CODES = {"pending_review", "in_progress", "visual_screened", "visual_defer", "lab_pending"}

LAB_METHODS = ["专业人员确认", "显微镜确认", "qPCR确认", "毒素检测确认"]
VISUAL_LABELS = ["正常/未见明显异常", "绿色水体异常", "红棕色水体异常", "高浑浊/泥沙样", "表层浮沫/漂浮物样", "不确定"]


def _root(root: Any | None = None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _paths(root: Any | None = None) -> dict[str, Path]:
    r = _root(root)
    d = r / "data" / "cases"
    return {"root": r, "dir": d, "ledger": d / "cases.json"}


def ensure_case_store(root: Any | None = None) -> dict[str, Path]:
    p = _paths(root)
    p["dir"].mkdir(parents=True, exist_ok=True)
    if not p["ledger"].exists():
        _atomic_write_json(p["ledger"], {"schema_version": CASE_SCHEMA_VERSION, "cases": []})
    return p


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_ledger(root: Any | None = None) -> dict[str, Any]:
    p = ensure_case_store(root)
    try:
        obj = json.loads(p["ledger"].read_text(encoding="utf-8"))
    except Exception:
        obj = {"schema_version": CASE_SCHEMA_VERSION, "cases": []}
    if not isinstance(obj, dict):
        obj = {"schema_version": CASE_SCHEMA_VERSION, "cases": []}
    obj.setdefault("schema_version", CASE_SCHEMA_VERSION)
    obj.setdefault("cases", [])
    return obj


def save_ledger(ledger: dict[str, Any], root: Any | None = None) -> None:
    p = ensure_case_store(root)
    ledger["schema_version"] = CASE_SCHEMA_VERSION
    _atomic_write_json(p["ledger"], ledger)


def list_cases(root: Any | None = None) -> list[dict[str, Any]]:
    cases = load_ledger(root).get("cases", [])
    return sorted(cases, key=lambda x: str(x.get("updated_at", "")), reverse=True)


def get_case(case_id: str | None, root: Any | None = None) -> dict[str, Any] | None:
    if not case_id:
        return None
    for case in load_ledger(root).get("cases", []):
        if str(case.get("case_id")) == str(case_id):
            return case
    return None




def case_status_code(case: dict[str, Any] | None) -> str:
    """Return normalized status code while remaining compatible with older ledgers."""
    if not case:
        return "pending_review"
    code = str(case.get("status_code") or "").strip()
    if code in CASE_STATUS_LABELS:
        return code
    label = str(case.get("status") or "")
    reverse = {v: k for k, v in CASE_STATUS_LABELS.items()}
    if label in reverse:
        return reverse[label]
    if "取消" in label:
        return "cancelled"
    if "归档" in label:
        return "archived"
    if "实验室" in label or "专业" in label:
        return "confirmed"
    if "视觉" in label and "DEFER" in label.upper():
        return "visual_defer"
    if "视觉" in label:
        return "visual_screened"
    if "进行" in label or "处理中" in label:
        return "in_progress"
    return "pending_review"


def _apply_status(case: dict[str, Any], status_code: str, reason: str | None = None) -> dict[str, Any]:
    if status_code not in CASE_STATUS_LABELS:
        raise ValueError(f"不支持的Case状态：{status_code}")
    now = _now()
    previous = case_status_code(case)
    case["status_code"] = status_code
    case["status"] = CASE_STATUS_LABELS[status_code]
    if reason:
        case["status_reason"] = reason.strip()
    case.setdefault("status_history", []).append({
        "from": previous,
        "to": status_code,
        "label": CASE_STATUS_LABELS[status_code],
        "reason": (reason or "").strip(),
        "at": now,
    })
    if isinstance(case.get("field_task"), dict):
        task_map = {
            "pending_review": "待拍照/现场补证据",
            "in_progress": "现场复核中",
            "visual_screened": "已完成视觉筛查，可继续实验室/专业确认",
            "visual_defer": "视觉结果DEFER，建议复拍或补充人工/实验室证据",
            "lab_pending": "等待实验室/专业确认",
            "confirmed": "已补充专业/实验室证据",
            "cancelled": "已取消现场复核",
            "archived": "已归档",
        }
        case["field_task"]["status"] = task_map.get(status_code, CASE_STATUS_LABELS[status_code])
    return case


def case_status_counts(root: Any | None = None) -> dict[str, int]:
    counts = {code: 0 for code in CASE_STATUS_LABELS}
    for case in list_cases(root):
        counts[case_status_code(case)] = counts.get(case_status_code(case), 0) + 1
    return counts


def queue_cases(root: Any | None = None, include_defer: bool = True) -> list[dict[str, Any]]:
    allowed = set(QUEUE_STATUS_CODES)
    if not include_defer:
        allowed.discard("visual_defer")
    return [c for c in list_cases(root) if case_status_code(c) in allowed]


def set_case_status(case_id: str, status_code: str, reason: str | None = None, root: Any | None = None) -> dict[str, Any]:
    return _update_case(case_id, lambda c: _apply_status(c, status_code, reason), root)


def mark_case_in_progress(case_id: str, root: Any | None = None) -> dict[str, Any]:
    return set_case_status(case_id, "in_progress", "开始/继续现场复核", root)


def cancel_case(case_id: str, reason: str | None = None, root: Any | None = None) -> dict[str, Any]:
    return set_case_status(case_id, "cancelled", reason or "用户取消本次现场复核", root)


def archive_case(case_id: str, reason: str | None = None, root: Any | None = None) -> dict[str, Any]:
    return set_case_status(case_id, "archived", reason or "用户归档", root)


def restore_case(case_id: str, root: Any | None = None) -> dict[str, Any]:
    case = get_case(case_id, root)
    if not case:
        raise ValueError(f"未找到Case：{case_id}")
    if case.get("lab_evidence"):
        target = "confirmed"
    elif case.get("visual"):
        target = "visual_screened"
    else:
        target = "pending_review"
    return set_case_status(case_id, target, "从取消/归档状态恢复", root)


def next_review_case(current_case_id: str | None = None, root: Any | None = None) -> dict[str, Any] | None:
    queue = queue_cases(root)
    if not queue:
        return None
    # Prefer pending tasks, then DEFER/lab-pending, and avoid returning the current case when possible.
    priority = {"pending_review": 0, "in_progress": 1, "visual_defer": 2, "visual_screened": 3, "lab_pending": 4}
    queue = sorted(queue, key=lambda c: (priority.get(case_status_code(c), 99), str(c.get("created_at", ""))))
    for case in queue:
        if str(case.get("case_id")) != str(current_case_id):
            return case
    return None


def _research_fingerprint(research: dict[str, Any]) -> str:
    keep = {
        "source": research.get("source"),
        "candidate_region": research.get("candidate_region") or research.get("region"),
        "issue_date": research.get("issue_date"),
        "forecast_window": research.get("forecast_window"),
        "horizon_days": research.get("horizon_days"),
        "route": research.get("route"),
        "lag_days": research.get("lag_days"),
        "scenario": research.get("scenario"),
    }
    return hashlib.sha256(json.dumps(_jsonable(keep), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def bulk_create_cases(
    research_items: list[dict[str, Any]],
    root: Any | None = None,
    deduplicate: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create several review tasks in one atomic ledger write. Returns (created, skipped)."""
    ledger = load_ledger(root)
    existing = {}
    for case in ledger.get("cases", []):
        fp = case.get("research_fingerprint") or _research_fingerprint(case.get("research") or {})
        if case_status_code(case) not in {"cancelled", "archived"}:
            existing[fp] = case
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for research in research_items:
        fp = _research_fingerprint(research)
        if deduplicate and fp in existing:
            skipped.append(existing[fp])
            continue
        now = _now()
        case_id = _new_case_id(research)
        region = research.get("candidate_region") or research.get("region") or "待定海域"
        window = research.get("forecast_window") or research.get("horizon_days") or ""
        case = {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": case_id,
            "title": f"{region}现场复核任务",
            "status_code": "pending_review",
            "status": CASE_STATUS_LABELS["pending_review"],
            "status_history": [{"from": None, "to": "pending_review", "label": CASE_STATUS_LABELS["pending_review"], "reason": "由风险候选批量生成", "at": now}],
            "research_fingerprint": fp,
            "created_at": now,
            "updated_at": now,
            "research": _jsonable(research),
            "field_task": {
                "status": "待拍照/现场补证据",
                "target_region": region,
                "forecast_window": window,
                "requested_evidence": ["海面/水色照片", "拍摄时间与位置", "水色/异味/泡沫", "可选DO/Chl-a/温盐"],
                "instructions": "优先在当前候选区拍摄海面，避开强逆光；视觉异常只作为现场证据层，必要时继续显微镜/qPCR/毒素复核。",
            },
            "visual": None,
            "field_metadata": {},
            "lab_evidence": [],
            "evidence": [],
            "llm_notes": [],
        }
        case["evidence"].append({
            "evidence_id": f"EV-{hashlib.sha256((case_id+'research').encode()).hexdigest()[:10]}",
            "type": "research_candidate",
            "grade": "C·模型风险候选",
            "source": "研究与验证",
            "created_at": now,
            "summary": {
                "candidate_region": research.get("candidate_region"),
                "risk_score": research.get("risk_score"),
                "route": research.get("route"),
                "lag_days": research.get("lag_days"),
                "top_k_capacity": research.get("top_k_capacity"),
                "event_coverage": research.get("event_coverage"),
            },
            "boundary": "研究候选用于安排现场复核，不等同于真实HAB确认。",
        })
        ledger.setdefault("cases", []).append(case)
        existing[fp] = case
        created.append(case)
    if created:
        save_ledger(ledger, root)
    return created, skipped


def case_training_usage(case_id: str, root: Any | None = None) -> dict[str, Any]:
    """Audit whether a Case has images in the visual library or prior model snapshots."""
    import csv
    r = _root(root)
    records_path = r / "data" / "field_visual" / "user_library" / "records.csv"
    sample_ids: list[str] = []
    future_training = 0
    if records_path.exists():
        try:
            with records_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("case_id") or "") == str(case_id):
                        sample_ids.append(str(row.get("sample_id") or ""))
                        if str(row.get("include_in_training") or "").strip().lower() in {"1", "true", "yes", "y"}:
                            future_training += 1
        except Exception:
            pass
    model_refs: list[str] = []
    model_root = r / "vision_models" / "user_models"
    if model_root.exists():
        for manifest in model_root.glob("*/training_manifest_snapshot.csv"):
            try:
                text = manifest.read_text(encoding="utf-8-sig", errors="ignore")
                if str(case_id) in text:
                    model_refs.append(manifest.parent.name)
            except Exception:
                continue
    return {
        "library_samples": len([x for x in sample_ids if x]),
        "future_training_samples": future_training,
        "trained_model_versions": model_refs,
    }


def remove_case_from_future_training(case_id: str, root: Any | None = None) -> int:
    import csv
    r = _root(root)
    path = r / "data" / "field_visual" / "user_library" / "records.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    changed = 0
    for row in rows:
        if str(row.get("case_id") or "") == str(case_id) and str(row.get("include_in_training") or "").strip().lower() in {"1", "true", "yes", "y"}:
            row["include_in_training"] = "False"
            changed += 1
    if changed and fieldnames:
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return changed


def delete_case_library_samples(case_id: str, root: Any | None = None) -> int:
    """Delete visual-library rows and image files linked only to this Case.

    Historical model weights are intentionally left untouched. Use with care when
    a model snapshot already references these samples because deleting the source
    rows/files reduces future reproducibility of that historical training run.
    """
    import csv
    r = _root(root)
    path = r / "data" / "field_visual" / "user_library" / "records.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    removed = [row for row in rows if str(row.get("case_id") or "") == str(case_id)]
    remaining = [row for row in rows if str(row.get("case_id") or "") != str(case_id)]
    if removed and fieldnames:
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(remaining)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    remaining_paths = {str(row.get("image_relpath") or "") for row in remaining}
    for row in removed:
        rel = str(row.get("image_relpath") or "").strip()
        if not rel or rel in remaining_paths:
            continue
        candidate = (r / rel).resolve()
        try:
            candidate.relative_to(r.resolve())
            if candidate.is_file():
                candidate.unlink()
        except Exception:
            continue
    return len(removed)


def delete_case(
    case_id: str,
    root: Any | None = None,
    remove_from_future_training: bool = False,
    delete_library_samples: bool = False,
) -> dict[str, Any]:
    usage = case_training_usage(case_id, root)
    if remove_from_future_training and not delete_library_samples:
        usage["removed_from_future_training"] = remove_case_from_future_training(case_id, root)
    if delete_library_samples:
        usage["deleted_library_samples"] = delete_case_library_samples(case_id, root)
    ledger = load_ledger(root)
    before = len(ledger.get("cases", []))
    ledger["cases"] = [c for c in ledger.get("cases", []) if str(c.get("case_id")) != str(case_id)]
    if len(ledger["cases"]) == before:
        raise ValueError(f"未找到Case：{case_id}")
    save_ledger(ledger, root)
    usage["deleted_case_id"] = case_id
    return usage


def _new_case_id(research: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    basis = json.dumps(_jsonable(research), ensure_ascii=False, sort_keys=True) + _now()
    return f"HAB-{stamp}-{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:6].upper()}"


def create_case_from_research(
    research: dict[str, Any],
    root: Any | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    ledger = load_ledger(root)
    now = _now()
    case_id = _new_case_id(research)
    region = research.get("candidate_region") or research.get("region") or "待定海域"
    window = research.get("forecast_window") or research.get("horizon_days") or ""
    case = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "title": title or f"{region}现场复核任务",
        "status_code": "pending_review",
        "status": CASE_STATUS_LABELS["pending_review"],
        "status_history": [{"from": None, "to": "pending_review", "label": CASE_STATUS_LABELS["pending_review"], "reason": "由风险候选生成", "at": now}],
        "research_fingerprint": _research_fingerprint(research),
        "created_at": now,
        "updated_at": now,
        "research": _jsonable(research),
        "field_task": {
            "status": "待拍照/现场补证据",
            "target_region": region,
            "forecast_window": window,
            "requested_evidence": ["海面/水色照片", "拍摄时间与位置", "水色/异味/泡沫", "可选DO/Chl-a/温盐"],
            "instructions": "优先在当前高风险候选区拍摄海面，避开强逆光；视觉异常只作为现场证据层，必要时继续显微镜/qPCR/毒素复核。",
        },
        "visual": None,
        "field_metadata": {},
        "lab_evidence": [],
        "evidence": [],
        "llm_notes": [],
    }
    case["evidence"].append({
        "evidence_id": f"EV-{hashlib.sha256((case_id+'research').encode()).hexdigest()[:10]}",
        "type": "research_candidate",
        "grade": "C·模型风险候选",
        "source": "研究与验证",
        "created_at": now,
        "summary": {
            "candidate_region": research.get("candidate_region"),
            "risk_score": research.get("risk_score"),
            "route": research.get("route"),
            "lag_days": research.get("lag_days"),
            "top_k_capacity": research.get("top_k_capacity"),
            "event_coverage": research.get("event_coverage"),
        },
        "boundary": "研究候选用于安排现场复核，不等同于真实HAB确认。",
    })
    ledger["cases"].append(case)
    save_ledger(ledger, root)
    return case


def _update_case(case_id: str, updater, root: Any | None = None) -> dict[str, Any]:
    ledger = load_ledger(root)
    for i, case in enumerate(ledger.get("cases", [])):
        if str(case.get("case_id")) == str(case_id):
            updated = updater(case)
            updated["updated_at"] = _now()
            ledger["cases"][i] = updated
            save_ledger(ledger, root)
            return updated
    raise ValueError(f"未找到Case：{case_id}")


def register_visual_evidence(
    case_id: str,
    result: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    sample_id: str | None = None,
    root: Any | None = None,
) -> dict[str, Any]:
    metadata = _jsonable(metadata or result.get("field_metadata") or {})
    visual = {
        "sample_id": sample_id,
        "screening_priority": result.get("screening_priority"),
        "visual_category": result.get("effective_visual_category") or (result.get("visual") or {}).get("category"),
        "quality_score": (result.get("quality") or {}).get("quality_score"),
        "backend": result.get("backend"),
        "adaptive_visual": _jsonable(result.get("adaptive_visual") or {}),
        "registered_at": _now(),
    }

    def updater(case: dict[str, Any]) -> dict[str, Any]:
        case["visual"] = visual
        case["field_metadata"] = metadata
        is_defer = str(result.get("screening_priority") or "").upper().startswith("DEFER")
        _apply_status(case, "visual_defer" if is_defer else "visual_screened", "完成现场视觉筛查")
        evidence = case.setdefault("evidence", [])
        sig = json.dumps(visual, ensure_ascii=False, sort_keys=True)
        ev_id = "EV-" + hashlib.sha256((case_id + sig).encode("utf-8")).hexdigest()[:10]
        evidence = [x for x in evidence if x.get("evidence_id") != ev_id]
        evidence.append({
            "evidence_id": ev_id,
            "type": "visual_screening",
            "grade": "B·现场视觉筛查",
            "source": "现场影像甄别",
            "created_at": _now(),
            "summary": visual,
            "boundary": "视觉证据不能单独确认具体藻种、毒素或HAB业务事件。",
        })
        case["evidence"] = evidence
        return case

    return _update_case(case_id, updater, root)


def register_lab_evidence(
    case_id: str,
    method: str,
    conclusion: str,
    visual_label: str | None = None,
    value_text: str | None = None,
    notes: str | None = None,
    sample_id: str | None = None,
    root: Any | None = None,
) -> dict[str, Any]:
    if method not in LAB_METHODS:
        raise ValueError("不支持的确认方式。")
    item = {
        "method": method,
        "conclusion": conclusion.strip(),
        "visual_label": visual_label,
        "value_text": (value_text or "").strip(),
        "notes": (notes or "").strip(),
        "sample_id": sample_id,
        "created_at": _now(),
    }

    def updater(case: dict[str, Any]) -> dict[str, Any]:
        case.setdefault("lab_evidence", []).append(item)
        _apply_status(case, "confirmed", f"登记{method}")
        grade = "A·实验室/专业确认"
        case.setdefault("evidence", []).append({
            "evidence_id": "EV-" + hashlib.sha256((case_id + json.dumps(item, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()[:10],
            "type": "lab_confirmation",
            "grade": grade,
            "source": method,
            "created_at": item["created_at"],
            "summary": item,
            "boundary": "证据解释仍受采样时间、位置、检测方法和检测限约束；不得外推为整个海域连续状态。",
        })
        return case

    return _update_case(case_id, updater, root)


def add_llm_note(
    case_id: str,
    text: str,
    mode: str,
    source_summary: str | None = None,
    root: Any | None = None,
) -> dict[str, Any]:
    note = {
        "created_at": _now(),
        "mode": mode,
        "text": text,
        "source_summary_sha256": hashlib.sha256((source_summary or "").encode("utf-8")).hexdigest(),
    }

    def updater(case: dict[str, Any]) -> dict[str, Any]:
        case.setdefault("llm_notes", []).append(note)
        case.setdefault("evidence", []).append({
            "evidence_id": "EV-" + hashlib.sha256((case_id + note["created_at"] + text).encode("utf-8")).hexdigest()[:10],
            "type": "llm_interpretation",
            "grade": "解释记录·不改变证据等级",
            "source": "大模型结果解读",
            "created_at": note["created_at"],
            "summary": {"mode": mode, "text_preview": text[:800]},
            "boundary": "大模型文本只解释已登记证据，不改变上游指标、概率、模型权重或事件确认状态。",
        })
        return case

    return _update_case(case_id, updater, root)


def evidence_rows(case: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not case:
        return []
    rows = []
    for ev in case.get("evidence", []):
        summary = ev.get("summary") or {}
        if ev.get("type") == "research_candidate":
            detail = f"{summary.get('candidate_region','')} · risk={summary.get('risk_score','NA')} · {summary.get('route','NA')} × {summary.get('lag_days','NA')}d"
        elif ev.get("type") == "visual_screening":
            detail = f"{summary.get('visual_category','NA')} · {summary.get('screening_priority','NA')} · quality={summary.get('quality_score','NA')}"
        elif ev.get("type") == "lab_confirmation":
            detail = f"{summary.get('method','')} · {summary.get('conclusion','')} · {summary.get('value_text','')}"
        else:
            detail = str(summary.get("text_preview") or summary)[:180]
        rows.append({
            "时间": ev.get("created_at"),
            "证据层": ev.get("grade"),
            "来源": ev.get("source"),
            "内容": detail,
            "边界": ev.get("boundary"),
        })
    return rows


def case_summary(case: dict[str, Any] | None, include_llm_history: bool = False) -> str:
    if not case:
        return "## 当前完整Case\n- 当前没有激活的研究Case。请先在“研究与验证 → 风险研判”生成现场复核任务。"
    research = case.get("research") or {}
    visual = case.get("visual") or {}
    field = case.get("field_metadata") or {}
    lines = [
        "## 当前完整Case",
        f"- Case ID={case.get('case_id')}；状态={case.get('status')}；标题={case.get('title')}。",
        "### 研究风险候选",
        f"- 候选海区={research.get('candidate_region','NA')}；风险指数={research.get('risk_score','NA')}；预测窗口={research.get('forecast_window','NA')}。",
        f"- Route={research.get('route','NA')}；Lag={research.get('lag_days','NA')}天；模型={research.get('model','NA')}。",
        f"- Top-k容量={research.get('top_k_capacity','NA')}；事件覆盖={research.get('event_coverage','NA')}。",
        "### 现场影像证据",
    ]
    if visual:
        lines.extend([
            f"- 视觉类别={visual.get('visual_category','NA')}；复核优先级={visual.get('screening_priority','NA')}；图像质量={visual.get('quality_score','NA')}。",
            f"- 视觉后端={visual.get('backend','NA')}。",
        ])
    else:
        lines.append("- 尚未登记现场影像甄别结果。")
    lines.append("### 现场环境/元数据")
    if field:
        keep = ["capture_date","capture_time","location_text","water_color","odor","surface_signs","recent_heat","mass_mortality","water_temp_c","salinity","dissolved_oxygen_mg_l","chlorophyll_a"]
        lines.append("- " + "；".join(f"{k}={field.get(k)}" for k in keep if field.get(k) not in (None, "", [], "未观察", "未知")))
    else:
        lines.append("- 尚未登记现场元数据。")
    lines.append("### 专业/实验室证据")
    lab = case.get("lab_evidence") or []
    if lab:
        for item in lab[-8:]:
            lines.append(f"- {item.get('method')}：{item.get('conclusion')}；{item.get('value_text','')}；视觉标签={item.get('visual_label','NA')}。")
    else:
        lines.append("- 尚未补充专业人员、显微镜、qPCR或毒素确认。")
    lines.extend([
        "### 证据边界",
        "- 研究候选、视觉筛查和实验室/专业确认是不同证据层；视觉结果不能自动改写研究模型标签。",
        "- 大模型只能综合解释已登记证据，不改变概率、指标、模型权重或事件确认状态。",
    ])
    if include_llm_history and case.get("llm_notes"):
        lines.append("### 历史大模型解释记录")
        for note in case.get("llm_notes", [])[-3:]:
            lines.append(f"- {note.get('created_at')} · {note.get('mode')}：{str(note.get('text',''))[:600]}")
    return "\n".join(lines)


def export_case_json(case_id: str, root: Any | None = None) -> bytes:
    case = get_case(case_id, root)
    if not case:
        raise ValueError("未找到Case。")
    return json.dumps(case, ensure_ascii=False, indent=2).encode("utf-8")
