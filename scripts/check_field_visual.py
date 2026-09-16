"""Non-UI smoke check for the field visual screening workspace."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.field_visual import make_result, result_summary  # noqa: E402


def image(base: tuple[int, int, int]) -> Image.Image:
    w, h = 800, 600
    x = np.linspace(-50, 50, w)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c, value in enumerate(base):
        arr[:, :, c] = value + x[None, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    md = {
        "sea_surface_confirmed": True,
        "odor": "未观察",
        "surface_signs": [],
        "recent_heat": "未知",
        "water_color": "不确定",
        "mass_mortality": "未观察",
        "location_text": "non-ui-smoke",
        "capture_date": "2026-09-16",
    }
    # Baseline mode remains available as a transparent reference and should
    # preserve the original broad visual-screening behaviour.
    green = make_result(image((55, 155, 90)), md, requested_mode="安全规则基线")
    blue = make_result(image((70, 130, 175)), md, requested_mode="安全规则基线")
    assertions = {
        "green_visual_type": green["visual"]["category"] == "绿色水体异常",
        "green_not_probability": "不是HAB发生概率" in green["priority_reason"],
        "blue_low_priority": blue["screening_priority"] == "低",
        "species_boundary_present": any("确诊具体藻种" in x for x in green["prohibited_claims"]),
        "llm_text_summary": "最近一次现场影像甄别" in result_summary(green),
    }
    payload = {
        "status": "pass" if all(assertions.values()) else "fail",
        "backend": green["backend"],
        "assertions": assertions,
        "green_category": green["visual"]["category"],
        "green_priority": green["screening_priority"],
        "blue_category": blue["visual"]["category"],
        "blue_priority": blue["screening_priority"],
    }
    out = ROOT / "validation" / "field_visual_workspace.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
