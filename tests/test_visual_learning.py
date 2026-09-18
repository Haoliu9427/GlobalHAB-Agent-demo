from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from globalhab_demo import visual_learning as vl


def _img_bytes(rgb):
    from io import BytesIO
    b = BytesIO()
    Image.new("RGB", (640, 480), rgb).save(b, format="JPEG")
    return b.getvalue()


def test_library_save_and_update(tmp_path):
    vl.ensure_visual_learning_store(tmp_path)
    sid, created = vl.save_image_sample(
        _img_bytes((30, 120, 150)), "a.jpg",
        {"location_text": "site-a", "site_id": "site-a", "surface_signs": []},
        "正常/未见明显异常", "专业人员确认", True, tmp_path,
    )
    assert created
    df = vl.load_library(tmp_path)
    assert len(df) == 1
    assert df.loc[0, "sample_id"] == sid
    edited = df[["sample_id", "filename", "label", "evidence_level", "include_in_training", "site_id", "capture_date", "screening_priority", "active_learning_score", "notes"]].copy()
    edited.loc[0, "label"] = "绿色水体异常"
    vl.update_library_annotations(edited, tmp_path)
    assert vl.load_library(tmp_path).loc[0, "label"] == "绿色水体异常"


def test_model_registry_defaults_to_public_baseline(tmp_path):
    vl.ensure_visual_learning_store(tmp_path)
    assert vl.get_active_model(tmp_path)["active_version"] == "public-baseline-v1"
    cards = vl.list_model_versions(tmp_path)
    assert any(c["version_id"] == "public-baseline-v1" for c in cards)


def test_library_export_ignores_deployment_dotfiles(tmp_path):
    paths = vl.ensure_visual_learning_store(tmp_path)
    (paths["images"] / ".gitkeep").write_text("", encoding="utf-8")
    with zipfile.ZipFile(BytesIO(vl.export_library_zip(tmp_path))) as archive:
        assert "images/.gitkeep" not in archive.namelist()


def test_train_candidate_and_activate(tmp_path, monkeypatch):
    vl.ensure_visual_learning_store(tmp_path)
    # 16 photos, two classes, four sites so group holdout is possible.
    for i in range(8):
        vl.save_image_sample(
            _img_bytes((30, 110 + i, 145)), f"n{i}.jpg",
            {"location_text": f"site-{i%4}", "site_id": f"site-{i%4}", "surface_signs": [], "water_color": "常规蓝/蓝绿"},
            "正常/未见明显异常", "专业人员确认", True, tmp_path,
        )
    for i in range(8):
        vl.save_image_sample(
            _img_bytes((60, 145 + i, 65)), f"g{i}.jpg",
            {"location_text": f"site-{i%4}", "site_id": f"site-{i%4}", "surface_signs": [], "water_color": "绿色"},
            "绿色水体异常", "专业人员确认", True, tmp_path,
        )

    def fake_encoder(image):
        arr = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
        means = arr.mean(axis=(0, 1))
        return np.asarray([means[0], means[1], means[2], means[1] - means[0]], dtype=np.float32)

    monkeypatch.setattr(vl, "_load_encoder", lambda name, root: fake_encoder)
    monkeypatch.setattr(vl, "_encoder_source", lambda name, root: "test public pretrained encoder")
    outcome = vl.train_candidate(tmp_path, ["efficientnet"], min_samples=12, seed=42, test_size=0.25)
    assert outcome.backbones_trained == ["efficientnet"]
    card_path = tmp_path / outcome.model_dir / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["version_id"] == outcome.version_id
    # Test registry resolution even if automatic recommendation happens to be conservative.
    card["eligible_for_activation"] = True
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    vl.activate_model(outcome.version_id, tmp_path)
    assert vl.get_active_model(tmp_path)["active_version"] == outcome.version_id
    assert vl.resolve_active_head(tmp_path, "efficientnet").is_file()
