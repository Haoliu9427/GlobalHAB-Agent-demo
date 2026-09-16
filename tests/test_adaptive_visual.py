from pathlib import Path

import numpy as np
from PIL import Image

from globalhab_demo.adaptive_visual import (
    BackboneStatus,
    LinearFusionHead,
    VISUAL_CLASSES,
    metadata_vector,
    plan_route,
    run_adaptive_route,
)
from globalhab_demo.field_visual import make_result


def _status(name: str, ready: bool = True) -> BackboneStatus:
    display = {"efficientnet": "EfficientNet-B0", "convnext": "ConvNeXt-Tiny", "dinov2": "DINOv2"}[name]
    return BackboneStatus(
        name=name,
        display_name=display,
        encoder_available=ready,
        head_available=ready,
        ready=ready,
        encoder_source="test",
        head_path="test.npz" if ready else None,
        note="test",
    )


def _quality(suitable=True, score=0.9):
    return {"suitable": suitable, "quality_score": score}


def _visual(category="绿色水体异常", anomaly=0.7, score=0.8, foam=0.01, turbid=0.02, sat=0.2):
    return {
        "category": category,
        "visual_anomaly_score": anomaly,
        "category_score": score,
        "white_low_saturation_fraction": foam,
        "turbid_fraction": turbid,
        "mean_saturation": sat,
    }


def test_adaptive_router_prefers_light_path_for_clear_colour_signal():
    statuses = {k: _status(k) for k in ("efficientnet", "convnext", "dinov2")}
    route = plan_route(_quality(), _visual(), statuses, "自适应路由")
    assert route.route_family == "colour_route"
    assert route.selected[0] == "efficientnet"


def test_adaptive_router_prefers_convnext_for_surface_texture():
    statuses = {k: _status(k) for k in ("efficientnet", "convnext", "dinov2")}
    route = plan_route(
        _quality(),
        _visual(category="表层浮沫/漂浮物样", anomaly=0.6, foam=0.2, turbid=0.1),
        statuses,
        "自适应路由",
    )
    assert route.route_family == "texture_route"
    assert route.selected[0] == "convnext"


def test_quality_gate_routes_to_defer_before_deep_models():
    statuses = {k: _status(k) for k in ("efficientnet", "convnext", "dinov2")}
    route = plan_route(_quality(suitable=False, score=0.2), _visual(), statuses, "自适应路由")
    assert route.route_family == "quality_defer"
    assert route.fallback_to_heuristic is True
    assert route.selected == []


def test_metadata_vector_is_fixed_and_finite():
    vec = metadata_vector({
        "odor": "有明显异味",
        "surface_signs": ["泡沫", "漂浮物"],
        "recent_heat": "是",
        "water_color": "红棕色",
        "mass_mortality": "观察到",
        "water_temp_c": 30.0,
        "salinity": 33.0,
        "dissolved_oxygen_mg_l": 4.2,
        "chlorophyll_a": 8.0,
    })
    assert vec.shape == (12,)
    assert np.isfinite(vec).all()
    assert vec[0] == 1.0
    assert vec[1] == 1.0


def test_numpy_linear_head_roundtrip(tmp_path: Path):
    # Two-class head with 3 embedding features + 12 metadata features.
    nfeat = 15
    coef = np.vstack([np.zeros(nfeat), np.zeros(nfeat)]).astype(np.float32)
    coef[1, 0] = 2.0
    path = tmp_path / "head.npz"
    np.savez_compressed(
        path,
        coef=coef,
        intercept=np.asarray([0.0, 0.0], dtype=np.float32),
        mean=np.zeros(nfeat, dtype=np.float32),
        scale=np.ones(nfeat, dtype=np.float32),
        classes=np.asarray(["正常/未见明显异常", "绿色水体异常"], dtype="U64"),
        ood_threshold=np.asarray(3.0, dtype=np.float32),
    )
    head = LinearFusionHead(path)
    probs, ood = head.predict(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), np.zeros(12, dtype=np.float32))
    assert probs.shape == (2,)
    assert probs[1] > probs[0]
    assert ood is not None


def test_no_deep_assets_falls_back_without_fabricating_model_result(tmp_path: Path):
    img = Image.new("RGB", (640, 480), (70, 130, 175))
    q = _quality()
    v = _visual(category="正常/未见明显异常", anomaly=0.05, score=0.8)
    result = run_adaptive_route(img, {}, q, v, root=tmp_path, requested_mode="自适应路由")
    assert result["active"] is False
    assert result["branches"] == []
    assert "未使用未训练" in result["scientific_note"]


def test_field_result_contract_contains_router_but_preserves_scientific_boundary(tmp_path: Path):
    # Use a gradient to avoid quality-gate rejection from a perfectly flat synthetic image.
    w, h = 640, 480
    x = np.linspace(-30, 30, w)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c, value in enumerate((55, 155, 90)):
        arr[:, :, c] = value + x[None, :]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    md = {
        "sea_surface_confirmed": True,
        "odor": "未观察",
        "surface_signs": [],
        "recent_heat": "未知",
        "water_color": "不确定",
        "mass_mortality": "未观察",
    }
    result = make_result(img, md, root=tmp_path, requested_mode="自适应路由")
    assert result["schema"] == "globalhab-field-visual-screening-v2"
    assert "adaptive_visual" in result
    assert result["adaptive_visual"]["active"] is False
    assert any("未配置训练头" in x for x in result["prohibited_claims"])


def test_active_routed_branch_with_local_calibrated_head(monkeypatch, tmp_path: Path):
    import globalhab_demo.adaptive_visual as av

    head_dir = tmp_path / "vision_models" / "heads"
    head_dir.mkdir(parents=True)
    nfeat = 3 + 12
    coef = np.zeros((len(VISUAL_CLASSES), nfeat), dtype=np.float32)
    green_idx = VISUAL_CLASSES.index("绿色水体异常")
    coef[green_idx, 0] = 8.0
    np.savez_compressed(
        head_dir / "efficientnet.npz",
        coef=coef,
        intercept=np.zeros(len(VISUAL_CLASSES), dtype=np.float32),
        mean=np.zeros(nfeat, dtype=np.float32),
        scale=np.ones(nfeat, dtype=np.float32),
        classes=np.asarray(VISUAL_CLASSES, dtype="U64"),
        ood_threshold=np.asarray(100.0, dtype=np.float32),
    )
    statuses = {
        "efficientnet": _status("efficientnet", True),
        "convnext": _status("convnext", False),
        "dinov2": _status("dinov2", False),
    }
    monkeypatch.setattr(av, "get_backbone_status", lambda root=None: statuses)
    monkeypatch.setattr(av, "_load_encoder", lambda name, root: (lambda image: np.asarray([1.0, 0.0, 0.0], dtype=np.float32)))

    img = Image.new("RGB", (640, 480), (55, 155, 90))
    result = av.run_adaptive_route(
        img,
        {},
        _quality(),
        _visual(category="绿色水体异常", anomaly=0.7, score=0.8),
        root=tmp_path,
        requested_mode="自适应路由",
    )
    assert result["active"] is True
    assert result["route"]["selected"] == ["efficientnet"]
    assert result["ensemble"]["predicted_class"] == "绿色水体异常"
    assert result["ensemble"]["visual_class_confidence"] > 0.9
    assert result["uncertainty"]["defer"] is False
