"""Offline smoke check for the operational adaptive visual router."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.adaptive_visual import get_backbone_status, metadata_vector  # noqa: E402
from globalhab_demo.field_visual import make_result  # noqa: E402


def image(base=(55, 155, 90)) -> Image.Image:
    w, h = 800, 600
    x = np.linspace(-50, 50, w)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c, value in enumerate(base):
        arr[:, :, c] = value + x[None, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    # Force the fully offline path: EfficientNet/ConvNeXt must still be able to
    # execute a real deep forward pass via the explicitly labelled prototype
    # encoder initialisation.
    os.environ["GLOBALHAB_ALLOW_MODEL_DOWNLOADS"] = "0"
    os.environ["GLOBALHAB_STRICT_PRETRAINED"] = "0"
    md = {
        "sea_surface_confirmed": True,
        "odor": "未观察",
        "surface_signs": [],
        "recent_heat": "未知",
        "water_color": "绿色",
        "mass_mortality": "未观察",
        "location_text": "adaptive-smoke",
        "capture_date": "2026-09-16",
    }
    statuses = get_backbone_status(ROOT)
    result = make_result(image(), md, root=ROOT, requested_mode="自适应路由（推荐）")
    adaptive = result["adaptive_visual"]
    route = adaptive["route"]
    branches = adaptive.get("branches") or []
    assertions = {
        "schema_v3": result["schema"] == "globalhab-field-visual-screening-v3",
        "router_present": "adaptive_visual" in result,
        "metadata_vector_12": metadata_vector(md).shape == (12,),
        "route_reason_present": bool(route.get("reason")),
        "deep_branch_executed": adaptive.get("active") is True and len(branches) >= 1,
        "real_feature_dim": bool(branches and int(branches[0].get("feature_dim", 0)) > 100),
        "prototype_head_present": bool(branches and branches[0].get("head_kind") in {"内置视觉现象原型头", "项目训练头"}),
        "scientific_boundary": any("HAB发生概率" in x for x in result["prohibited_claims"]),
    }
    payload = {
        "status": "pass" if all(assertions.values()) else "fail",
        "assertions": assertions,
        "backbone_status": {
            k: {"encoder": v.encoder_available, "head": v.head_kind, "ready": v.ready, "source": v.encoder_source}
            for k, v in statuses.items()
        },
        "route": route,
        "active": adaptive["active"],
        "branches": branches,
        "priority": result["screening_priority"],
    }
    out = ROOT / "validation" / "adaptive_visual_router.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
