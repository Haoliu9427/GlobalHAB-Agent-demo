"""Train a lightweight visual head on frozen backbone embeddings + field metadata.

This script intentionally does NOT ship trained weights. Users must provide a labelled field-image
manifest whose labels are supported visual phenomena. Site/time holdouts are strongly recommended.
The resulting safe `.npz` head can be loaded by the Streamlit adaptive visual router.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.adaptive_visual import (  # noqa: E402
    META_FEATURE_NAMES,
    VISUAL_CLASSES,
    _load_encoder,
    metadata_vector,
)
from globalhab_demo.field_visual import load_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--backbone", required=True, choices=["dinov2", "convnext", "efficientnet"])
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--group-column", default="site_id", help="Use site/station grouping for validation when present.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_surface(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return []
    for sep in (";", "|", "、", ","):
        text = text.replace(sep, ";")
    return [x.strip() for x in text.split(";") if x.strip()]


def row_metadata(row: pd.Series) -> dict:
    def val(key: str, default=None):
        x = row.get(key, default)
        return default if pd.isna(x) else x
    return {
        "location_text": val("location_text", ""),
        "water_color": val("water_color", "不确定"),
        "odor": val("odor", "未观察"),
        "surface_signs": parse_surface(val("surface_signs", "")),
        "recent_heat": val("recent_heat", "未知"),
        "mass_mortality": val("mass_mortality", "未观察"),
        "water_temp_c": val("water_temp_c"),
        "salinity": val("salinity"),
        "dissolved_oxygen_mg_l": val("dissolved_oxygen_mg_l"),
        "chlorophyll_a": val("chlorophyll_a"),
    }


def convert_binary_head(clf: LogisticRegression) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32)
    if len(clf.classes_) == 2 and coef.shape[0] == 1:
        coef = np.vstack([-0.5 * coef[0], 0.5 * coef[0]]).astype(np.float32)
        intercept = np.asarray([-0.5 * intercept[0], 0.5 * intercept[0]], dtype=np.float32)
    return coef, intercept


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    required = {"image_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit("manifest missing: " + ", ".join(sorted(missing)))
    df = df.dropna(subset=["image_path", "label"]).copy()
    bad = sorted(set(df["label"].astype(str)) - set(VISUAL_CLASSES))
    if bad:
        raise SystemExit("unsupported labels: " + ", ".join(bad))
    if df["label"].nunique() < 2:
        raise SystemExit("at least two visual classes are required")

    encoder = _load_encoder(args.backbone, ROOT)
    X_rows: list[np.ndarray] = []
    y: list[str] = []
    kept_rows: list[int] = []
    embed_dim = None
    for i, row in df.iterrows():
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = (args.manifest.parent / path).resolve()
        try:
            raw = path.read_bytes()
            image = load_image(raw)
            emb = np.asarray(encoder(image), dtype=np.float32).reshape(-1)
            meta = metadata_vector(row_metadata(row))
            if embed_dim is None:
                embed_dim = emb.size
            if emb.size != embed_dim:
                raise ValueError("inconsistent embedding dimension")
            X_rows.append(np.concatenate([emb, meta]).astype(np.float32))
            y.append(str(row["label"]))
            kept_rows.append(i)
        except Exception as exc:
            print(f"SKIP row={i} image={path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if len(X_rows) < 10:
        raise SystemExit("too few readable labelled images; need at least 10 for a smoke-level fit")

    X = np.vstack(X_rows)
    y_arr = np.asarray(y)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    Z = (X - mean) / scale

    kept = df.loc[kept_rows].reset_index(drop=True)
    idx = np.arange(len(kept))
    if args.group_column in kept.columns and kept[args.group_column].notna().sum() >= 4 and kept[args.group_column].nunique() >= 2:
        groups = kept[args.group_column].fillna("__missing__").astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(idx, y_arr, groups))
        split_note = f"group holdout by {args.group_column}"
    else:
        stratify = y_arr if min(pd.Series(y_arr).value_counts()) >= 2 else None
        train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=stratify)
        split_note = "random holdout (use site/time grouping for scientific validation)"

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=args.seed)
    clf.fit(Z[train_idx], y_arr[train_idx])
    pred = clf.predict(Z[test_idx])
    proba = clf.predict_proba(Z[test_idx])
    metrics = {
        "validation_split": split_note,
        "n_total": int(len(Z)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": float(accuracy_score(y_arr[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr[test_idx], pred)),
        "log_loss": float(log_loss(y_arr[test_idx], proba, labels=clf.classes_)),
        "classes": [str(x) for x in clf.classes_],
        "warning": "These are engineering diagnostics, not publication-grade HAB validation. Use independent site/time holdouts and laboratory-confirmed labels.",
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Refit on all data after the holdout diagnostic; validation metrics above stay frozen in sidecar.
    clf.fit(Z, y_arr)
    coef, intercept = convert_binary_head(clf)
    rms_z = np.sqrt(np.mean(np.square(Z), axis=1))
    ood_threshold = float(np.quantile(rms_z, 0.99))

    output = args.output or (ROOT / "vision_models" / "heads" / f"{args.backbone}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coef=coef,
        intercept=intercept,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        classes=np.asarray([str(x) for x in clf.classes_], dtype="U64"),
        ood_threshold=np.asarray(ood_threshold, dtype=np.float32),
        embedding_dim=np.asarray(int(embed_dim or 0), dtype=np.int32),
        metadata_dim=np.asarray(len(META_FEATURE_NAMES), dtype=np.int32),
    )
    sidecar = output.with_suffix(".metrics.json")
    sidecar.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved head: {output}")
    print(f"saved metrics: {sidecar}")


if __name__ == "__main__":
    main()
