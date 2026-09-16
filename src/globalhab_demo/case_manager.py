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


CASE_SCHEMA_VERSION = "1.0"
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
        "status": "待现场复核",
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
        case["field_task"]["status"] = "已完成视觉筛查，待确认/补实验室证据"
        case["status"] = "已有现场视觉证据"
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
        case["status"] = "已有专业/实验室确认"
        case["field_task"]["status"] = "已补充专业/实验室证据"
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
