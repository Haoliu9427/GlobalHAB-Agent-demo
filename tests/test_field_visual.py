from io import BytesIO

import numpy as np
from PIL import Image

from globalhab_demo.field_visual import load_image, make_result, result_summary


def _gradient_image(base: tuple[int, int, int], size=(800, 600)) -> Image.Image:
    w, h = size
    x = np.linspace(-50, 50, w)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c, value in enumerate(base):
        arr[:, :, c] = value + x[None, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _metadata(**extra):
    base = {
        "sea_surface_confirmed": True,
        "odor": "未观察",
        "surface_signs": [],
        "recent_heat": "未知",
        "water_color": "不确定",
        "mass_mortality": "未观察",
        "location_text": "测试海域",
        "capture_date": "2026-09-16",
    }
    base.update(extra)
    return base


def test_green_visual_signal_is_screened_not_species_label():
    result = make_result(_gradient_image((55, 155, 90)), _metadata())
    assert result["visual"]["category"] == "绿色水体异常"
    assert result["screening_priority"] in {"中", "高"}
    assert "不是藻种/毒素确诊器" in result["scientific_role"]
    assert any("不能仅凭普通海面照片确诊具体藻种" in x for x in result["prohibited_claims"])


def test_blue_water_can_return_low_priority():
    result = make_result(_gradient_image((70, 130, 175)), _metadata())
    assert result["visual"]["category"] == "正常/未见明显异常"
    assert result["screening_priority"] == "低"


def test_bad_image_or_wrong_target_defers():
    result = make_result(_gradient_image((70, 130, 175)), _metadata(sea_surface_confirmed=False))
    assert result["screening_priority"].startswith("DEFER")


def test_field_context_can_raise_followup_priority_without_becoming_probability():
    result = make_result(
        _gradient_image((70, 130, 175)),
        _metadata(
            odor="有明显异味",
            surface_signs=["泡沫"],
            recent_heat="是",
            water_color="绿色",
        ),
    )
    assert result["screening_priority"] in {"中", "高"}
    assert "不是HAB发生概率" in result["priority_reason"]
    assert len(result["field_context_flags"]) >= 3


def test_result_summary_never_sends_image_bytes():
    result = make_result(_gradient_image((160, 110, 60)), _metadata())
    text = result_summary(result)
    assert "最近一次现场影像甄别" in text
    assert "红棕色水体异常" in text
    assert "HAB概率" in text
    assert "image_bytes" not in text


def test_load_image_roundtrip_png():
    img = _gradient_image((55, 155, 90), size=(640, 480))
    buf = BytesIO()
    img.save(buf, format="PNG")
    loaded = load_image(buf.getvalue())
    assert loaded.size == (640, 480)
    assert loaded.mode == "RGB"
