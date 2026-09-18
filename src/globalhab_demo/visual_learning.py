"""Persistent visual data library and lightweight model lifecycle for GlobalHAB-Agent.

This module turns the field-photo workspace into a small continuous-learning
visual agent. It deliberately separates three concerns:

1. user photo library + evidence labels;
2. candidate lightweight heads trained on frozen visual encoders;
3. version registry / validation / promotion to the active model.

The code does not claim species or toxin identification. Labels describe broad
water-surface visual phenomena only. Large backbone weights remain managed by
``adaptive_visual``; model versions store only lightweight ``.npz`` heads,
metrics, manifests and model cards.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
import hashlib
import html
import json
import math
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from globalhab_demo.adaptive_visual import (
    META_FEATURE_NAMES,
    VISUAL_CLASSES,
    LinearFusionHead,
    _ENCODER_RUNTIME_SOURCE,
    _allow_downloads,
    _load_encoder,
    _model_asset_paths,
    _prototype_predict,
    metadata_vector,
)
from globalhab_demo.field_visual import DEFAULT_BACKEND, load_image


LABELS = list(VISUAL_CLASSES)
TRAINABLE_LABELS = [x for x in VISUAL_CLASSES if x != "不确定"]
EVIDENCE_LEVELS = [
    "仅肉眼判断",
    "专业人员确认",
    "显微镜确认",
    "qPCR确认",
    "毒素检测确认",
]
EVIDENCE_WEIGHTS = {
    "仅肉眼判断": 0.55,
    "专业人员确认": 0.78,
    "显微镜确认": 1.00,
    "qPCR确认": 1.00,
    "毒素检测确认": 1.00,
}
MODEL_NAMES = ("efficientnet", "convnext", "dinov2")
MODEL_DISPLAY = {
    "efficientnet": "EfficientNet-B0",
    "convnext": "ConvNeXt-Tiny",
    "dinov2": "DINOv2",
}

LIBRARY_COLUMNS = [
    "sample_id", "case_id", "image_relpath", "filename", "sha256", "source", "label",
    "evidence_level", "include_in_training", "site_id", "capture_date",
    "capture_time", "location_text", "water_color", "odor", "surface_signs",
    "recent_heat", "mass_mortality", "water_temp_c", "salinity",
    "dissolved_oxygen_mg_l", "chlorophyll_a", "notes", "metadata_json",
    "screening_priority", "screening_category", "active_learning_score",
    "created_at", "updated_at",
]

PUBLIC_SOURCE_CATALOG = [
    {
        "dataset_id": "algal-blooms-sweden-2023",
        "name": "Algal Blooms Sweden - 2023",
        "doi": "10.5281/zenodo.10599927",
        "url": "https://zenodo.org/records/10599927",
        "download_url": "https://zenodo.org/records/10599927/files/algal_blooms_sweden_2023.zip?download=1",
        "observations": 60,
        "scope": "Baltic Sea summer surface bloom photographs",
        "evidence": "Bloom occurrence verified by Swedish information centres; no microscopy/genetic taxonomic annotation in the dataset record.",
        "role": "Optional public positive-domain adapter / representation calibration; not a species classifier.",
    },
]


@dataclass
class TrainingOutcome:
    version_id: str
    status: str
    n_total: int
    n_classes: int
    classes: list[str]
    backbones_requested: list[str]
    backbones_trained: list[str]
    skipped_backbones: list[str]
    candidate_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    promotion_recommendation: str
    promotion_reasons: list[str]
    eligible_for_activation: bool
    model_dir: str


def _root(root: Any | None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _paths(root: Any | None = None) -> dict[str, Path]:
    r = _root(root)
    return {
        "root": r,
        "library_root": r / "data" / "field_visual" / "user_library",
        "images": r / "data" / "field_visual" / "user_library" / "images",
        "records": r / "data" / "field_visual" / "user_library" / "records.csv",
        "public_root": r / "data" / "field_visual" / "public",
        "public_catalog": r / "data" / "field_visual" / "public" / "PUBLIC_DATA_SOURCES.json",
        "models": r / "vision_models",
        "public_baseline": r / "vision_models" / "public_baseline",
        "user_models": r / "vision_models" / "user_models",
        "active_model": r / "vision_models" / "active_model.json",
    }


def ensure_visual_learning_store(root: Any | None = None) -> dict[str, Path]:
    p = _paths(root)
    for key in ("library_root", "images", "public_root", "models", "public_baseline", "user_models"):
        p[key].mkdir(parents=True, exist_ok=True)
    if not p["records"].exists():
        pd.DataFrame(columns=LIBRARY_COLUMNS).to_csv(p["records"], index=False)
    if not p["public_catalog"].exists():
        p["public_catalog"].write_text(json.dumps(PUBLIC_SOURCE_CATALOG, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_card = p["public_baseline"] / "model_card.json"
    if not baseline_card.exists():
        baseline_card.write_text(json.dumps({
            "version_id": "public-baseline-v1",
            "display_name": "公共视觉基线",
            "status": "baseline",
            "method": "公共预训练视觉编码器优先 + 内置水体现象原型头 + 透明颜色/纹理 + 现场元数据",
            "trained_supervised_head": False,
            "public_source_catalog": str(p["public_catalog"].relative_to(p["root"])),
            "scientific_boundary": "公共基线用于视觉现象筛查，不代表经过全球海洋HAB现场照片监督校准。",
            "created_at": _now(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not p["active_model"].exists():
        p["active_model"].write_text(json.dumps({
            "active_version": "public-baseline-v1",
            "activated_at": _now(),
            "reason": "初始公共视觉基线",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_library(root: Any | None = None) -> pd.DataFrame:
    p = ensure_visual_learning_store(root)
    try:
        df = pd.read_csv(p["records"])
    except Exception:
        df = pd.DataFrame(columns=LIBRARY_COLUMNS)
    for col in LIBRARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    object_cols = [c for c in LIBRARY_COLUMNS if c not in {"include_in_training", "water_temp_c", "salinity", "dissolved_oxygen_mg_l", "chlorophyll_a", "active_learning_score"}]
    for col in object_cols:
        df[col] = df[col].astype("object")
    if not df.empty:
        df["include_in_training"] = df["include_in_training"].map(
            lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"} if not isinstance(x, bool) else x
        ).astype(bool)
    return df[LIBRARY_COLUMNS]


def save_library(df: pd.DataFrame, root: Any | None = None) -> None:
    p = ensure_visual_learning_store(root)
    out = df.copy()
    for col in LIBRARY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out[LIBRARY_COLUMNS].to_csv(p["records"], index=False)


def _safe_ext(filename: str | None) -> str:
    ext = Path(filename or "photo.jpg").suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png"} else ".jpg"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def active_learning_score_from_result(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    adaptive = result.get("adaptive_visual") or {}
    u = adaptive.get("uncertainty") or {}
    entropy = float(u.get("entropy") or 0.0)
    disagreement = float(u.get("disagreement") or 0.0)
    margin = float(u.get("margin") or 1.0)
    ood = 1.0 if bool(u.get("ood_flag")) else 0.0
    defer = 1.0 if bool(u.get("defer")) or str(result.get("screening_priority", "")).startswith("DEFER") else 0.0
    score = 0.35 * entropy + 0.25 * disagreement + 0.20 * (1.0 - np.clip(margin, 0, 1)) + 0.10 * ood + 0.10 * defer
    return float(np.clip(score, 0.0, 1.0))


def save_image_sample(
    raw: bytes,
    filename: str | None,
    metadata: dict[str, Any],
    label: str,
    evidence_level: str,
    include_in_training: bool,
    root: Any | None = None,
    screening_result: dict[str, Any] | None = None,
    source: str = "用户上传",
) -> tuple[str, bool]:
    if label not in LABELS:
        raise ValueError("不支持的视觉标签。")
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError("不支持的证据等级。")
    image = load_image(raw)  # validates bytes
    _ = image
    p = ensure_visual_learning_store(root)
    digest = hashlib.sha256(raw).hexdigest()
    sample_id = digest[:16]
    df = load_library(root)
    existing = df[df["sha256"].astype(str) == digest] if not df.empty else pd.DataFrame()
    now = _now()
    ext = _safe_ext(filename)
    image_name = f"{sample_id}{ext}"
    image_path = p["images"] / image_name
    if not image_path.exists():
        image_path.write_bytes(raw)
    rel = str(image_path.relative_to(p["root"]))
    surface_signs = metadata.get("surface_signs") or []
    row = {
        "sample_id": sample_id,
        "case_id": metadata.get("case_id"),
        "image_relpath": rel,
        "filename": filename or image_name,
        "sha256": digest,
        "source": source,
        "label": label,
        "evidence_level": evidence_level,
        "include_in_training": bool(include_in_training),
        "site_id": (metadata.get("site_id") or metadata.get("location_text") or "").strip(),
        "capture_date": metadata.get("capture_date"),
        "capture_time": metadata.get("capture_time"),
        "location_text": metadata.get("location_text"),
        "water_color": metadata.get("water_color"),
        "odor": metadata.get("odor"),
        "surface_signs": ";".join(str(x) for x in surface_signs),
        "recent_heat": metadata.get("recent_heat"),
        "mass_mortality": metadata.get("mass_mortality"),
        "water_temp_c": metadata.get("water_temp_c"),
        "salinity": metadata.get("salinity"),
        "dissolved_oxygen_mg_l": metadata.get("dissolved_oxygen_mg_l"),
        "chlorophyll_a": metadata.get("chlorophyll_a"),
        "notes": metadata.get("notes"),
        "metadata_json": json.dumps(_jsonable(metadata), ensure_ascii=False),
        "screening_priority": (screening_result or {}).get("screening_priority"),
        "screening_category": (screening_result or {}).get("effective_visual_category"),
        "active_learning_score": active_learning_score_from_result(screening_result),
        "created_at": now,
        "updated_at": now,
    }
    if not existing.empty:
        idx = existing.index[0]
        row["created_at"] = df.loc[idx, "created_at"] if pd.notna(df.loc[idx, "created_at"]) else now
        for key, value in row.items():
            df.loc[idx, key] = value
        created = False
    else:
        if df.empty:
            df = pd.DataFrame([row], columns=LIBRARY_COLUMNS)
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        created = True
    save_library(df, root)
    return sample_id, created


def update_library_annotations(edited: pd.DataFrame, root: Any | None = None) -> None:
    current = load_library(root)
    if current.empty:
        return
    allowed = {str(x) for x in current["sample_id"]}
    incoming = edited.copy()
    for _, row in incoming.iterrows():
        sid = str(row.get("sample_id", ""))
        if sid not in allowed:
            continue
        mask = current["sample_id"].astype(str) == sid
        label = str(row.get("label", "不确定"))
        evidence = str(row.get("evidence_level", "仅肉眼判断"))
        if label not in LABELS:
            label = "不确定"
        if evidence not in EVIDENCE_LEVELS:
            evidence = "仅肉眼判断"
        current.loc[mask, "label"] = label
        current.loc[mask, "evidence_level"] = evidence
        current.loc[mask, "include_in_training"] = bool(row.get("include_in_training", False))
        current.loc[mask, "site_id"] = str(row.get("site_id", "") or "")
        current.loc[mask, "notes"] = str(row.get("notes", "") or "")
        current.loc[mask, "updated_at"] = _now()
    save_library(current, root)


def _metadata_from_record(row: pd.Series) -> dict[str, Any]:
    try:
        if pd.notna(row.get("metadata_json")) and str(row.get("metadata_json")).strip():
            obj = json.loads(str(row.get("metadata_json")))
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    surface = str(row.get("surface_signs") or "")
    return {
        "location_text": "" if pd.isna(row.get("location_text")) else str(row.get("location_text")),
        "water_color": "不确定" if pd.isna(row.get("water_color")) else str(row.get("water_color")),
        "odor": "未观察" if pd.isna(row.get("odor")) else str(row.get("odor")),
        "surface_signs": [x.strip() for x in surface.split(";") if x.strip()],
        "recent_heat": "未知" if pd.isna(row.get("recent_heat")) else str(row.get("recent_heat")),
        "mass_mortality": "未观察" if pd.isna(row.get("mass_mortality")) else str(row.get("mass_mortality")),
        "water_temp_c": None if pd.isna(row.get("water_temp_c")) else float(row.get("water_temp_c")),
        "salinity": None if pd.isna(row.get("salinity")) else float(row.get("salinity")),
        "dissolved_oxygen_mg_l": None if pd.isna(row.get("dissolved_oxygen_mg_l")) else float(row.get("dissolved_oxygen_mg_l")),
        "chlorophyll_a": None if pd.isna(row.get("chlorophyll_a")) else float(row.get("chlorophyll_a")),
    }


def eligible_training_records(root: Any | None = None) -> pd.DataFrame:
    df = load_library(root)
    if df.empty:
        return df
    mask = (
        df["include_in_training"].fillna(False).astype(bool)
        & df["label"].astype(str).isin(TRAINABLE_LABELS)
        & df["image_relpath"].notna()
    )
    return df.loc[mask].reset_index(drop=True)


def library_summary(root: Any | None = None) -> dict[str, Any]:
    df = load_library(root)
    train = eligible_training_records(root)
    return {
        "n_images": int(len(df)),
        "n_training": int(len(train)),
        "n_classes": int(train["label"].nunique()) if not train.empty else 0,
        "n_high_evidence": int(train["evidence_level"].isin(["显微镜确认", "qPCR确认", "毒素检测确认"]).sum()) if not train.empty else 0,
        "n_active_learning": int((pd.to_numeric(df.get("active_learning_score", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 0.55).sum()) if not df.empty else 0,
    }


def export_library_zip(root: Any | None = None) -> bytes:
    p = ensure_visual_learning_store(root)
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(p["records"], arcname="records.csv")
        for img in sorted(p["images"].glob("*")):
            if img.is_file():
                zf.write(img, arcname=f"images/{img.name}")
        zf.writestr("README.txt", "GlobalHAB-Agent field visual library export. Keep records.csv with images/.\n")
    return bio.getvalue()


def import_library_zip(raw: bytes, root: Any | None = None) -> dict[str, int]:
    p = ensure_visual_learning_store(root)
    added = 0
    skipped = 0
    with zipfile.ZipFile(BytesIO(raw), "r") as zf:
        names = set(zf.namelist())
        if "records.csv" not in names:
            raise ValueError("训练库ZIP缺少 records.csv。")
        incoming = pd.read_csv(BytesIO(zf.read("records.csv")))
        current = load_library(root)
        seen = set(current["sha256"].astype(str)) if not current.empty else set()
        new_rows = []
        for _, row in incoming.iterrows():
            digest = str(row.get("sha256", ""))
            sid = str(row.get("sample_id", ""))
            image_rel = str(row.get("image_relpath", ""))
            image_name = Path(image_rel).name if image_rel else ""
            arc = f"images/{image_name}"
            if not digest or digest in seen or not image_name or arc not in names:
                skipped += 1
                continue
            data = zf.read(arc)
            if hashlib.sha256(data).hexdigest() != digest:
                skipped += 1
                continue
            dest = p["images"] / image_name
            dest.write_bytes(data)
            record = {col: row.get(col, np.nan) for col in LIBRARY_COLUMNS}
            record["image_relpath"] = str(dest.relative_to(p["root"]))
            record["updated_at"] = _now()
            new_rows.append(record)
            seen.add(digest)
            added += 1
        if new_rows:
            current = pd.concat([current, pd.DataFrame(new_rows)], ignore_index=True)
            save_library(current, root)
    return {"added": added, "skipped": skipped}


def _ece(y_true: np.ndarray, probs: np.ndarray, classes: list[str], bins: int = 8) -> float:
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred = np.asarray([classes[i] for i in pred_idx])
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def _convert_binary_head(clf: Any) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32)
    if len(clf.classes_) == 2 and coef.shape[0] == 1:
        coef = np.vstack([-0.5 * coef[0], 0.5 * coef[0]]).astype(np.float32)
        intercept = np.asarray([-0.5 * intercept[0], 0.5 * intercept[0]], dtype=np.float32)
    return coef, intercept


def _encoder_source(name: str, root: Path) -> str:
    key = (name, str(_model_asset_paths(root)[name]["encoder"]), _allow_downloads())
    return str(_ENCODER_RUNTIME_SOURCE.get(key, MODEL_DISPLAY.get(name, name)))


def _prepare_features(
    df: pd.DataFrame,
    backbone: str,
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[Any], str]:
    encoder = _load_encoder(backbone, root)
    rows = []
    labels = []
    weights = []
    metas: list[dict[str, Any]] = []
    visuals: list[Any] = []
    p = _paths(root)
    for _, row in df.iterrows():
        path = root / str(row["image_relpath"])
        try:
            image = load_image(path.read_bytes())
            emb = np.asarray(encoder(image), dtype=np.float32).reshape(-1)
            meta = _metadata_from_record(row)
            meta_vec = metadata_vector(meta)
            _, visual = DEFAULT_BACKEND.analyse(image)
            rows.append(np.concatenate([emb, meta_vec]).astype(np.float32))
            labels.append(str(row["label"]))
            weights.append(float(EVIDENCE_WEIGHTS.get(str(row.get("evidence_level")), 0.55)))
            metas.append(meta)
            visuals.append(visual)
        except Exception:
            continue
    if not rows:
        raise ValueError(f"{MODEL_DISPLAY.get(backbone, backbone)} 没有成功读取任何训练样本。")
    X = np.vstack(rows)
    y = np.asarray(labels)
    w = np.asarray(weights, dtype=np.float64)
    return X, y, w, metas, visuals, _encoder_source(backbone, root)


def _split_indices(df: pd.DataFrame, y: np.ndarray, seed: int = 42, test_size: float = 0.25) -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    idx = np.arange(len(y))
    groups = df.iloc[: len(y)]["site_id"].fillna("").astype(str).to_numpy() if "site_id" in df.columns else np.asarray([""] * len(y))
    valid_groups = [g for g in groups if g]
    if len(set(valid_groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(idx, y, groups))
        return tr, te, "站点/海域分组留出"
    counts = pd.Series(y).value_counts()
    stratify = y if len(counts) >= 2 and int(counts.min()) >= 2 else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
    return np.asarray(tr), np.asarray(te), "随机分层留出（建议后续补充独立站点/年份验证）"


def _metrics(y_true: np.ndarray, pred: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    try:
        class_to_idx = {str(c): i for i, c in enumerate(classes)}
        true_prob = np.asarray([float(probs[i, class_to_idx[str(y)]]) for i, y in enumerate(y_true)], dtype=float)
        ll = float(-np.mean(np.log(np.clip(true_prob, 1e-12, 1.0))))
    except Exception:
        ll = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss": ll,
        "ece": _ece(y_true, probs, classes),
    }


def _prototype_baseline_metrics(
    backbone: str,
    root: Path,
    df: pd.DataFrame,
    test_idx: np.ndarray,
) -> dict[str, float]:
    encoder = _load_encoder(backbone, root)
    y_true: list[str] = []
    preds: list[str] = []
    probs_rows: list[np.ndarray] = []
    classes = list(VISUAL_CLASSES)
    for pos in test_idx:
        row = df.iloc[int(pos)]
        try:
            image = load_image((root / str(row["image_relpath"])).read_bytes())
            emb = np.asarray(encoder(image), dtype=np.float32)
            meta = _metadata_from_record(row)
            _, visual = DEFAULT_BACKEND.analyse(image)
            probs, _, _ = _prototype_predict(backbone, encoder, emb, meta, visual, root)
            y_true.append(str(row["label"]))
            preds.append(classes[int(np.argmax(probs))])
            probs_rows.append(np.asarray(probs, dtype=float))
        except Exception:
            continue
    if not probs_rows:
        return {}
    return _metrics(np.asarray(y_true), np.asarray(preds), np.vstack(probs_rows), classes)


def _version_id(prefix: str = "user") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _model_card_path(root: Path, version_id: str) -> Path:
    p = _paths(root)
    if version_id == "public-baseline-v1":
        return p["public_baseline"] / "model_card.json"
    return p["user_models"] / version_id / "model_card.json"


def list_model_versions(root: Any | None = None) -> list[dict[str, Any]]:
    p = ensure_visual_learning_store(root)
    cards: list[dict[str, Any]] = []
    baseline = p["public_baseline"] / "model_card.json"
    if baseline.exists():
        try:
            cards.append(json.loads(baseline.read_text(encoding="utf-8")))
        except Exception:
            pass
    for card_path in sorted(p["user_models"].glob("*/model_card.json"), reverse=True):
        try:
            cards.append(json.loads(card_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    active = get_active_model(root).get("active_version")
    for card in cards:
        card["is_active"] = card.get("version_id") == active
    return cards


def get_active_model(root: Any | None = None) -> dict[str, Any]:
    p = ensure_visual_learning_store(root)
    try:
        return json.loads(p["active_model"].read_text(encoding="utf-8"))
    except Exception:
        return {"active_version": "public-baseline-v1"}


def resolve_active_head(root: Any | None, backbone: str) -> Path | None:
    p = ensure_visual_learning_store(root)
    active = get_active_model(root).get("active_version", "public-baseline-v1")
    if active == "public-baseline-v1":
        legacy = p["models"] / "heads" / f"{backbone}.npz"
        return legacy if legacy.is_file() else None
    candidate = p["user_models"] / str(active) / "heads" / f"{backbone}.npz"
    return candidate if candidate.is_file() else None


def active_head_kind(root: Any | None, backbone: str) -> str:
    active = get_active_model(root).get("active_version", "public-baseline-v1")
    head = resolve_active_head(root, backbone)
    if head is None:
        return "内置视觉现象原型头"
    if active == "public-baseline-v1":
        return "项目训练头"
    return f"用户模型 · {active}"


def activate_model(version_id: str, root: Any | None = None, reason: str = "用户确认注册") -> None:
    p = ensure_visual_learning_store(root)
    card_path = _model_card_path(p["root"], version_id)
    if not card_path.exists():
        raise ValueError("找不到该模型版本。")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if version_id != "public-baseline-v1" and not bool(card.get("eligible_for_activation", False)):
        raise ValueError("该候选模型未通过自动验证门槛，不能注册为当前模型。")
    p["active_model"].write_text(json.dumps({
        "active_version": version_id,
        "activated_at": _now(),
        "reason": reason,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if version_id != "public-baseline-v1":
        card["status"] = "active"
        card["activated_at"] = _now()
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")


def reject_model(version_id: str, root: Any | None = None, reason: str = "用户决定不注册") -> None:
    r = _root(root)
    card_path = _model_card_path(r, version_id)
    if not card_path.exists() or version_id == "public-baseline-v1":
        raise ValueError("该模型不能标记为拒绝。")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["status"] = "rejected"
    card["rejected_at"] = _now()
    card["rejection_reason"] = reason
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")


def train_candidate(
    root: Any | None = None,
    backbones: Iterable[str] = ("efficientnet",),
    seed: int = 42,
    test_size: float = 0.25,
    min_samples: int = 12,
    min_classes: int = 2,
) -> TrainingOutcome:
    from sklearn.linear_model import LogisticRegression

    r = _root(root)
    p = ensure_visual_learning_store(r)
    df = eligible_training_records(r)
    if len(df) < min_samples:
        raise ValueError(f"可训练照片只有 {len(df)} 张，至少需要 {min_samples} 张。")
    if df["label"].nunique() < min_classes:
        raise ValueError(f"当前只有 {df['label'].nunique()} 个有效类别，至少需要 {min_classes} 类。")

    version_id = _version_id()
    model_dir = p["user_models"] / version_id
    heads_dir = model_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(model_dir / "training_manifest_snapshot.csv", index=False)

    candidate_metrics: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    trained: list[str] = []
    skipped: list[str] = []
    pretrained_sources_ok = True
    split_notes: dict[str, str] = {}

    # Split is deterministic and shared across backbones.
    y_all = df["label"].astype(str).to_numpy()
    train_idx, test_idx, split_note = _split_indices(df, y_all, seed=seed, test_size=test_size)

    for backbone in [b for b in backbones if b in MODEL_NAMES]:
        try:
            X, y, sample_w, metas, visuals, source = _prepare_features(df, backbone, r)
            if len(y) != len(df):
                # Keep scientific behaviour explicit: do not silently compare on a
                # different sample set if some files failed for this backbone.
                raise ValueError("存在无法读取/编码的训练图片，请先在影像库中检查文件完整性。")
            mean = X[train_idx].mean(axis=0)
            scale = X[train_idx].std(axis=0)
            scale = np.where(scale < 1e-6, 1.0, scale)
            Z_train = (X[train_idx] - mean) / scale
            Z_test = (X[test_idx] - mean) / scale
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=seed)
            clf.fit(Z_train, y[train_idx], sample_weight=sample_w[train_idx])
            pred = clf.predict(Z_test)
            probs = clf.predict_proba(Z_test)
            metrics = _metrics(y[test_idx], pred, probs, [str(x) for x in clf.classes_])
            metrics.update({
                "n_total": int(len(y)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "classes": [str(x) for x in clf.classes_],
                "validation_split": split_note,
                "encoder_source": source,
            })
            candidate_metrics[backbone] = metrics
            baseline_metrics[backbone] = _prototype_baseline_metrics(backbone, r, df, test_idx)
            split_notes[backbone] = split_note
            if "非预训练" in source:
                pretrained_sources_ok = False

            # Refit on all eligible data after frozen holdout diagnostics.
            full_mean = X.mean(axis=0)
            full_scale = X.std(axis=0)
            full_scale = np.where(full_scale < 1e-6, 1.0, full_scale)
            Z_full = (X - full_mean) / full_scale
            clf.fit(Z_full, y, sample_weight=sample_w)
            coef, intercept = _convert_binary_head(clf)
            rms_z = np.sqrt(np.mean(np.square(Z_full), axis=1))
            ood_threshold = float(np.quantile(rms_z, 0.99))
            emb_dim = int(X.shape[1] - len(META_FEATURE_NAMES))
            out = heads_dir / f"{backbone}.npz"
            np.savez_compressed(
                out,
                coef=coef,
                intercept=intercept,
                mean=full_mean.astype(np.float32),
                scale=full_scale.astype(np.float32),
                classes=np.asarray([str(x) for x in clf.classes_], dtype="U64"),
                ood_threshold=np.asarray(ood_threshold, dtype=np.float32),
                embedding_dim=np.asarray(emb_dim, dtype=np.int32),
                metadata_dim=np.asarray(len(META_FEATURE_NAMES), dtype=np.int32),
            )
            trained.append(backbone)
        except Exception as exc:
            skipped.append(f"{MODEL_DISPLAY.get(backbone, backbone)}: {type(exc).__name__}: {exc}")

    if not trained:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise ValueError("没有任何视觉分支成功完成训练；请检查视觉依赖、预训练权重和训练图片。")

    # Promotion recommendation uses same-holdout comparison against the public
    # prototype baseline where available. It is intentionally conservative.
    reasons: list[str] = []
    passes = 0
    for b in trained:
        cand = candidate_metrics.get(b, {})
        base = baseline_metrics.get(b, {})
        cf1 = float(cand.get("macro_f1", 0.0))
        cba = float(cand.get("balanced_accuracy", 0.0))
        cece = float(cand.get("ece", 1.0))
        if base:
            bf1 = float(base.get("macro_f1", 0.0))
            bba = float(base.get("balanced_accuracy", 0.0))
            bece = float(base.get("ece", 1.0))
            ok = cf1 >= bf1 - 0.01 and cba >= bba - 0.01 and cece <= bece + 0.05
            if ok:
                passes += 1
            else:
                reasons.append(f"{MODEL_DISPLAY[b]} 在同一留出集上未同时满足相对公共基线的F1/平衡准确率/校准门槛。")
        else:
            ok = cf1 >= 0.45 and cba >= 0.45 and cece <= 0.25
            if ok:
                passes += 1
            else:
                reasons.append(f"{MODEL_DISPLAY[b]} 未达到首个候选模型的最低绝对诊断门槛。")
    min_class_count = int(df["label"].value_counts().min())
    if min_class_count < 3:
        reasons.append("至少一个类别少于3张训练照片，版本只能作为工程候选。")
    if len(test_idx) < 4:
        reasons.append("独立留出样本过少，暂不建议自动注册。")
    if not pretrained_sources_ok:
        reasons.append("至少一个分支使用非预训练离线编码初始化；可运行但不适合作为正式注册模型。")

    eligible = bool(passes >= 1 and min_class_count >= 3 and len(test_idx) >= 4 and pretrained_sources_ok)
    recommendation = "建议注册" if eligible else "暂不注册"
    if eligible:
        reasons.append("至少一个训练分支在同一留出集上达到相对公共基线的性能/校准门槛，且样本与编码器状态满足注册条件。")

    card = {
        "version_id": version_id,
        "display_name": f"用户视觉模型 {version_id}",
        "status": "candidate",
        "created_at": _now(),
        "training_samples": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "evidence_counts": {str(k): int(v) for k, v in df["evidence_level"].value_counts().to_dict().items()},
        "backbones_requested": list(backbones),
        "backbones_trained": trained,
        "skipped_backbones": skipped,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics_same_holdout": baseline_metrics,
        "validation_split": split_note,
        "eligible_for_activation": eligible,
        "promotion_recommendation": recommendation,
        "promotion_reasons": reasons,
        "scientific_boundary": "候选头只学习用户标注的水体视觉现象，不用于具体藻种/毒素确诊。首次注册前应优先使用独立站点/年份和实验室确认标签。",
    }
    (model_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps({
        "candidate": candidate_metrics,
        "public_baseline_same_holdout": baseline_metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainingOutcome(
        version_id=version_id,
        status="candidate",
        n_total=int(len(df)),
        n_classes=int(df["label"].nunique()),
        classes=sorted(df["label"].astype(str).unique().tolist()),
        backbones_requested=list(backbones),
        backbones_trained=trained,
        skipped_backbones=skipped,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        promotion_recommendation=recommendation,
        promotion_reasons=reasons,
        eligible_for_activation=eligible,
        model_dir=str(model_dir.relative_to(r)),
    )


def model_package_bytes(version_id: str, root: Any | None = None) -> bytes:
    p = ensure_visual_learning_store(root)
    if version_id == "public-baseline-v1":
        folder = p["public_baseline"]
    else:
        folder = p["user_models"] / version_id
    if not folder.exists():
        raise ValueError("找不到该模型版本。")
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in folder.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=str(Path(version_id) / file.relative_to(folder)))
    return bio.getvalue()


def training_manifest_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "image_path", "label", "evidence_level", "site_id", "capture_date",
        "location_text", "water_color", "odor", "surface_signs", "recent_heat",
        "mass_mortality", "water_temp_c", "salinity", "dissolved_oxygen_mg_l",
        "chlorophyll_a", "notes",
    ])


def public_source_catalog(root: Any | None = None) -> list[dict[str, Any]]:
    p = ensure_visual_learning_store(root)
    try:
        obj = json.loads(p["public_catalog"].read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else PUBLIC_SOURCE_CATALOG
    except Exception:
        return PUBLIC_SOURCE_CATALOG


# ------------------------------- Streamlit UI -------------------------------

def _fmt_metric(x: Any) -> str:
    try:
        v = float(x)
        return f"{v:.3f}" if math.isfinite(v) else "NA"
    except Exception:
        return "NA"


def _render_learning_kpis(items: list[tuple[str, str, str]], *, key: str, columns: int = 4) -> None:
    """Render visual-learning summaries with the same project KPI cards as screening."""
    import streamlit as st

    cards = "".join(
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-note">{html.escape(str(note))}</div>'
        '</div>'
        for label, value, note in items
    )
    grid_class = " kpi-3" if columns == 3 else " kpi-4" if columns == 4 else ""
    with st.container(key=key):
        st.markdown(f'<div class="kpi-grid{grid_class}">{cards}</div>', unsafe_allow_html=True)


def render_library_tab(root: Any | None = None) -> None:
    """Render the persistent/user-exportable photo library subpage."""
    import streamlit as st

    r = _root(root)
    ensure_visual_learning_store(r)
    summary = library_summary(r)
    st.markdown("### 我的影像数据")
    st.caption("把现场照片、视觉标签和实验室/专家证据保存为训练资产。模型自己的预测不会自动变成训练标签。")
    _render_learning_kpis([
        ("影像总数", str(summary["n_images"]), "已保存的现场/导入影像"),
        ("进入训练集", str(summary["n_training"]), "已确认并纳入后续训练"),
        ("有效视觉类别", str(summary["n_classes"]), "当前训练资产覆盖的类别"),
        ("高信息量待确认", str(summary["n_active_learning"]), "优先建议人工复核的样本"),
    ], key="vision_library_summary", columns=4)

    result = st.session_state.get("field_visual_result")
    raw = st.session_state.get("field_visual_image_bytes")
    with st.container(border=True, key="vision_library_recent_card"):
        st.markdown("#### 保存最近一次甄别照片")
        if result and raw:
            default_label = result.get("effective_visual_category", "不确定")
            if default_label not in LABELS:
                default_label = "不确定"
            c1, c2 = st.columns(2)
            label = c1.selectbox("人工/专家视觉标签", LABELS, index=LABELS.index(default_label), key="vision_library_label")
            evidence = c2.selectbox("证据等级", EVIDENCE_LEVELS, key="vision_library_evidence")
            include = st.checkbox("加入后续训练集", value=label != "不确定", key="vision_library_include")
            st.caption("建议：只有人工/专家确认后的标签才加入训练。若后续获得显微镜、qPCR或毒素结果，可回来提高证据等级。")
            if st.button("保存到我的影像数据", type="primary", use_container_width=True, key="vision_library_save_current"):
                try:
                    sid, created = save_image_sample(
                        raw=raw,
                        filename=st.session_state.get("field_visual_image_name", "field_capture.jpg"),
                        metadata=result.get("field_metadata") or {},
                        label=label,
                        evidence_level=evidence,
                        include_in_training=include,
                        root=r,
                        screening_result=result,
                    )
                    st.success(("已新增" if created else "已更新") + f"影像记录：{sid}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("先在“现场影像甄别”子页完成一次拍照/上传与甄别，再把照片保存到训练库。")

    with st.container(border=True, key="vision_library_batch_card"):
        st.markdown("#### 批量补充已标注照片")
        files = st.file_uploader(
            "上传多张JPG / JPEG / PNG",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="vision_library_batch",
        )
        b1, b2, b3 = st.columns(3)
        batch_label = b1.selectbox("本批标签", LABELS, key="vision_batch_label")
        batch_evidence = b2.selectbox("本批证据等级", EVIDENCE_LEVELS, key="vision_batch_evidence")
        batch_site = b3.text_input("站点/海域ID", key="vision_batch_site", placeholder="例如 SA_ST01")
        batch_include = st.checkbox("本批加入训练集", value=batch_label != "不确定", key="vision_batch_include")
        if st.button("保存本批照片", use_container_width=True, key="vision_batch_save"):
            if not files:
                st.warning("请先上传照片。")
            else:
                n_ok = 0
                n_fail = 0
                for f in files:
                    try:
                        metadata = {
                            "site_id": batch_site.strip(),
                            "location_text": batch_site.strip(),
                            "water_color": "不确定",
                            "odor": "未观察",
                            "surface_signs": [],
                            "recent_heat": "未知",
                            "mass_mortality": "未观察",
                            "sea_surface_confirmed": True,
                            "notes": "批量导入；可在下方表格继续修改。",
                        }
                        save_image_sample(
                            raw=f.getvalue(), filename=f.name, metadata=metadata,
                            label=batch_label, evidence_level=batch_evidence,
                            include_in_training=batch_include, root=r, screening_result=None,
                        )
                        n_ok += 1
                    except Exception:
                        n_fail += 1
                st.success(f"已处理 {n_ok} 张；失败 {n_fail} 张。")
                st.rerun()

    df = load_library(r)
    if df.empty:
        st.caption("影像库目前为空。")
    else:
        st.markdown("#### 影像库与标签管理")
        view_cols = [
            "sample_id", "case_id", "filename", "label", "evidence_level", "include_in_training",
            "site_id", "capture_date", "screening_priority", "active_learning_score", "notes",
        ]
        editable = df[view_cols].copy()
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            disabled=["sample_id", "case_id", "filename", "capture_date", "screening_priority", "active_learning_score"],
            column_config={
                "label": st.column_config.SelectboxColumn("视觉标签", options=LABELS, required=True),
                "evidence_level": st.column_config.SelectboxColumn("证据等级", options=EVIDENCE_LEVELS, required=True),
                "include_in_training": st.column_config.CheckboxColumn("进入训练"),
                "active_learning_score": st.column_config.NumberColumn("主动学习分数", format="%.3f"),
            },
            key="vision_library_editor",
        )
        if st.button("保存标签与训练集选择", type="primary", use_container_width=True, key="vision_library_save_edits"):
            update_library_annotations(edited, r)
            st.success("影像标签和训练集选择已保存。")
            st.rerun()

        scores = pd.to_numeric(df["active_learning_score"], errors="coerce").fillna(0)
        queue = df.assign(_score=scores).sort_values("_score", ascending=False).head(10)
        with st.expander("主动学习：优先人工确认这些照片", expanded=False):
            st.caption("优先显示模型不确定、跨分支不一致、OOD或DEFER较高的样本；先标这些照片通常比随机标注更省人工。")
            st.dataframe(
                queue[["sample_id", "case_id", "filename", "label", "evidence_level", "screening_priority", "_score"]].rename(columns={"_score": "信息优先级"}),
                use_container_width=True, hide_index=True,
            )

    io1, io2 = st.columns(2)
    io1.download_button(
        "导出我的影像库 ZIP",
        data=export_library_zip(r),
        file_name="GlobalHAB_visual_library.zip",
        mime="application/zip",
        use_container_width=True,
    )
    imported = io2.file_uploader("导入已有影像库 ZIP", type=["zip"], key="vision_library_import")
    if imported is not None and st.button("合并导入影像库", use_container_width=True, key="vision_library_import_btn"):
        try:
            report = import_library_zip(imported.getvalue(), r)
            st.success(f"新增 {report['added']} 张；跳过 {report['skipped']} 张。")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.caption("部署说明：本地运行时影像库保存在工程目录；部分云端Streamlit环境的本地磁盘可能在重启后重置，因此长期积累时请定期导出ZIP并在需要时重新导入。")


def render_training_tab(root: Any | None = None) -> None:
    """Render public baseline, candidate training, validation and model versioning."""
    import streamlit as st

    r = _root(root)
    ensure_visual_learning_store(r)
    active = get_active_model(r)
    summary = library_summary(r)
    cards = list_model_versions(r)

    st.markdown("### 模型训练与版本")
    st.caption("公共基线 → 选择训练数据 → 训练轻量头 → 自动留出验证 → 与公共基线比较 → 决定是否注册为当前模型。")

    _render_learning_kpis([
        ("当前模型", str(active.get("active_version", "public-baseline-v1")), "现场甄别当前调用的视觉版本"),
        ("可训练照片", str(summary["n_training"]), "已确认且允许进入训练的样本"),
        ("有效类别", str(summary["n_classes"]), "当前可用于监督训练的类别"),
    ], key="vision_training_summary", columns=3)

    with st.container(border=True, key="vision_training_baseline_card"):
        st.markdown("#### 公共视觉基线")
        st.write("默认基线使用公共预训练视觉编码器（可获取时）+ 内置水体现象原型 + 透明颜色/纹理 + 现场元数据。它用于视觉筛查，不声称已经完成全球海洋HAB监督验证。")
        source_rows = public_source_catalog(r)
        if source_rows:
            st.dataframe(pd.DataFrame(source_rows)[["name", "doi", "observations", "scope", "role"]], use_container_width=True, hide_index=True)
        st.caption("工程同时提供公开数据获取/适配脚本；大体量公共照片不直接塞进代码仓库，只保存来源清单、适配器/轻量头、模型卡和哈希，便于复现和控制仓库体积。")

    train_df = eligible_training_records(r)
    with st.container(border=True, key="vision_training_data_card"):
        st.markdown("#### 训练数据检查")
        if train_df.empty:
            st.warning("当前没有被标记为“进入训练”的已标注照片。请先到“我的影像数据”选择训练样本。")
        else:
            counts = train_df["label"].value_counts().rename_axis("视觉类别").reset_index(name="照片数")
            st.dataframe(counts, use_container_width=True, hide_index=True)
            st.caption(f"训练照片 {len(train_df)} 张；高证据等级 {summary['n_high_evidence']} 张。标签为“不确定”的照片不会进入监督训练。")

    with st.container(border=True, key="vision_training_candidate_card"):
        st.markdown("#### 训练候选版本")
        selected = st.multiselect(
            "选择视觉分支",
            options=list(MODEL_NAMES),
            default=["efficientnet"],
            format_func=lambda x: MODEL_DISPLAY[x],
            key="vision_train_backbones",
        )
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("留出比例", 0.20, 0.40, 0.25, 0.05, key="vision_train_test_size")
        seed = c2.number_input("随机种子", min_value=1, max_value=999999, value=42, step=1, key="vision_train_seed")
        min_samples = c3.number_input("最少训练照片", min_value=10, max_value=500, value=12, step=1, key="vision_train_min_samples")
        st.caption("默认冻结视觉backbone，只重新训练轻量分类/元数据融合头；如果有站点ID，验证优先按站点分组留出，减少同地点泄漏。")
        if st.button("训练新的候选模型", type="primary", use_container_width=True, key="vision_train_candidate"):
            if not selected:
                st.warning("至少选择一个视觉分支。")
            else:
                try:
                    with st.spinner("正在提取冻结视觉特征、训练轻量头并进行留出验证……"):
                        outcome = train_candidate(
                            r, selected, seed=int(seed), test_size=float(test_size), min_samples=int(min_samples)
                        )
                    st.session_state["visual_training_outcome"] = asdict(outcome)
                    st.success(f"候选版本已生成：{outcome.version_id} · {outcome.promotion_recommendation}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    outcome = st.session_state.get("visual_training_outcome")
    if outcome:
        st.markdown("#### 最近一次候选模型验证")
        rows = []
        for b, metrics in (outcome.get("candidate_metrics") or {}).items():
            base = (outcome.get("baseline_metrics") or {}).get(b, {})
            rows.append({
                "分支": MODEL_DISPLAY.get(b, b),
                "候选 Macro-F1": _fmt_metric(metrics.get("macro_f1")),
                "公共基线 Macro-F1": _fmt_metric(base.get("macro_f1")),
                "候选平衡准确率": _fmt_metric(metrics.get("balanced_accuracy")),
                "公共基线平衡准确率": _fmt_metric(base.get("balanced_accuracy")),
                "候选 ECE": _fmt_metric(metrics.get("ece")),
                "公共基线 ECE": _fmt_metric(base.get("ece")),
                "编码器": metrics.get("encoder_source", "NA"),
            })
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        if outcome.get("eligible_for_activation"):
            st.success("自动验证建议：可以注册为当前模型。仍建议保留独立站点/年份和实验室确认样本做后续审计。")
        else:
            st.warning("自动验证建议：暂不注册。候选版本仍被保存，可继续补数据后训练新版本。")
        for reason in outcome.get("promotion_reasons") or []:
            st.write("- " + reason)
        act1, act2 = st.columns(2)
        if act1.button("注册该候选为当前模型", disabled=not bool(outcome.get("eligible_for_activation")), use_container_width=True, key="vision_activate_latest"):
            try:
                activate_model(outcome["version_id"], r, reason="通过自动验证后由用户确认注册")
                st.success("已切换当前视觉模型。后续现场甄别将自动调用该版本中可用的训练头。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if act2.button("保留但不注册", use_container_width=True, key="vision_reject_latest"):
            try:
                reject_model(outcome["version_id"], r, reason="用户在模型训练工作区选择暂不注册")
                st.success("候选版本已保留并标记为不注册。")
                st.session_state.pop("visual_training_outcome", None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.markdown("#### 模型版本库")
    cards = list_model_versions(r)
    if cards:
        rows = []
        for card in cards:
            rows.append({
                "版本": card.get("version_id"),
                "状态": "当前" if card.get("is_active") else card.get("status", "candidate"),
                "训练照片": card.get("training_samples", "—"),
                "训练分支": ", ".join(card.get("backbones_trained") or []) if card.get("backbones_trained") else "公共基线",
                "自动建议": card.get("promotion_recommendation", "—"),
                "可注册": card.get("eligible_for_activation", card.get("version_id") == "public-baseline-v1"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        version_ids = [str(c.get("version_id")) for c in cards]
        selected_version = st.selectbox("选择模型版本", version_ids, key="vision_version_select")
        selected_card = next((c for c in cards if str(c.get("version_id")) == selected_version), None)
        v1, v2 = st.columns(2)
        try:
            pkg = model_package_bytes(selected_version, r)
            v1.download_button(
                "下载该模型版本 ZIP",
                data=pkg,
                file_name=f"GlobalHAB_visual_model_{selected_version}.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception:
            pass
        can_activate = bool(selected_version == "public-baseline-v1" or (selected_card or {}).get("eligible_for_activation"))
        if v2.button("切换为当前模型", disabled=not can_activate, use_container_width=True, key="vision_activate_selected"):
            try:
                activate_model(selected_version, r, reason="用户从版本库手动切换")
                st.success(f"当前模型已切换到 {selected_version}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.caption("模型晋级不是“训练完自动覆盖”。候选版本先通过固定留出诊断，再由用户显式注册；未通过门槛的版本可以保留用于审计，但不会替换当前模型。")
