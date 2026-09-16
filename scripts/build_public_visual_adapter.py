#!/usr/bin/env python
"""Build positive-domain embedding adapters from a public bloom photo manifest.

This is deliberately a one-class/domain adapter rather than a multi-class HAB
classifier: the referenced public surface-bloom dataset verifies bloom events
but does not provide microscopy/genetic species labels or normal-water controls.
Only aggregate centroids/statistics are saved into the model repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.adaptive_visual import _load_encoder, _normalise_embedding  # noqa: E402
from globalhab_demo.field_visual import load_image  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "field_visual" / "public" / "algal-blooms-sweden-2023" / "public_manifest.csv")
    p.add_argument("--backbone", choices=["efficientnet", "convnext", "dinov2"], default="efficientnet")
    args = p.parse_args()
    manifest = args.manifest.resolve()
    df = pd.read_csv(manifest)
    if "image_path" not in df.columns:
        raise SystemExit("manifest requires image_path")
    encoder = _load_encoder(args.backbone, ROOT)
    feats = []
    skipped = 0
    for rel in df["image_path"].dropna().astype(str):
        path = Path(rel)
        if not path.is_absolute():
            path = manifest.parent / path
        try:
            img = load_image(path.read_bytes())
            feats.append(_normalise_embedding(encoder(img)))
        except Exception as exc:
            skipped += 1
            print(f"SKIP {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if len(feats) < 5:
        raise SystemExit("too few readable public images; need at least 5")
    arr = np.stack(feats, axis=0)
    centroid = _normalise_embedding(arr.mean(axis=0))
    sims = arr @ centroid
    out_dir = ROOT / "vision_models" / "public_baseline" / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.backbone}.npz"
    np.savez_compressed(
        out,
        centroid=centroid.astype(np.float32),
        similarity_p05=np.asarray(float(np.quantile(sims, 0.05)), dtype=np.float32),
        similarity_median=np.asarray(float(np.median(sims)), dtype=np.float32),
        n_images=np.asarray(len(feats), dtype=np.int32),
    )
    card = {
        "backbone": args.backbone,
        "n_images": len(feats),
        "skipped": skipped,
        "manifest": str(manifest.relative_to(ROOT) if str(manifest).startswith(str(ROOT)) else manifest),
        "role": "positive-domain embedding adapter; not a species/toxin classifier",
    }
    (out_dir / f"{args.backbone}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), **card}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
