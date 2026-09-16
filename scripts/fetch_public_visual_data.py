#!/usr/bin/env python
"""Fetch an optional public surface-bloom photo dataset for visual-domain adaptation.

The release does not bundle hundreds of MB of third-party photos. This script
retrieves the cited public dataset into ``data/field_visual/public`` and creates
an auditable manifest. Run only when network access and the source licence/terms
are acceptable for your deployment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = {
    "id": "algal-blooms-sweden-2023",
    "name": "Algal Blooms Sweden - 2023",
    "doi": "10.5281/zenodo.10599927",
    "record_url": "https://zenodo.org/records/10599927",
    "download_url": "https://zenodo.org/records/10599927/files/algal_blooms_sweden_2023.zip?download=1",
    "md5": "2833e9f80b512bd9d3dd8379dfef93ba",
    "observations": 60,
    "boundary": "Verified bloom occurrences in the dataset record, but not taxonomically annotated by microscopy or genetic analysis.",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest_resolved)):
            raise RuntimeError(f"unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out = ROOT / "data" / "field_visual" / "public" / DATASET["id"]
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "source.zip"
    if not archive.exists() or args.force:
        print(f"Downloading {DATASET['download_url']}")
        urllib.request.urlretrieve(DATASET["download_url"], archive)
    got = md5(archive)
    if got != DATASET["md5"]:
        raise SystemExit(f"MD5 mismatch: expected {DATASET['md5']} got {got}")
    extract = out / "extracted"
    if args.force and extract.exists():
        import shutil
        shutil.rmtree(extract)
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        safe_extract(zf, extract)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(x for x in extract.rglob("*") if x.is_file() and x.suffix.lower() in image_exts)
    manifest = out / "public_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "source_dataset", "evidence_level", "note"])
        w.writeheader()
        for img in images:
            w.writerow({
                "image_path": str(img.relative_to(out)),
                "source_dataset": DATASET["name"],
                "evidence_level": "官方/专业机构事件核验（非物种实验室确证）",
                "note": "Positive bloom-domain image; use for representation/domain adapter, not species classification.",
            })
    (out / "source_record.json").write_text(json.dumps(DATASET, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": DATASET["name"], "images_found": len(images), "manifest": str(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
