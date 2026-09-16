#!/usr/bin/env python
from __future__ import annotations

import json
import tempfile
from io import BytesIO
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.visual_learning import (  # noqa: E402
    ensure_visual_learning_store,
    get_active_model,
    library_summary,
    list_model_versions,
    save_image_sample,
)


def img_bytes(rgb):
    b = BytesIO()
    Image.new("RGB", (640, 480), rgb).save(b, format="JPEG")
    return b.getvalue()


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ensure_visual_learning_store(root)
        save_image_sample(
            img_bytes((40, 120, 150)), "blue.jpg",
            {"location_text": "demo-a", "site_id": "demo-a", "surface_signs": []},
            "正常/未见明显异常", "专业人员确认", True, root,
        )
        save_image_sample(
            img_bytes((65, 150, 70)), "green.jpg",
            {"location_text": "demo-b", "site_id": "demo-b", "surface_signs": [], "water_color": "绿色"},
            "绿色水体异常", "专业人员确认", True, root,
        )
        summary = library_summary(root)
        active = get_active_model(root)
        versions = list_model_versions(root)
        checks = {
            "three_subpage_backend_store": summary["n_images"] == 2,
            "training_selection": summary["n_training"] == 2,
            "public_baseline_active": active.get("active_version") == "public-baseline-v1",
            "version_registry": any(v.get("version_id") == "public-baseline-v1" for v in versions),
        }
        payload = {"status": "pass" if all(checks.values()) else "fail", "assertions": checks, "summary": summary}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
