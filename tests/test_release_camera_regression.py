from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from globalhab_demo import release_ui

ROOT = Path(__file__).resolve().parents[1]

def test_release_label_difference_keeps_compatible_homepage():
    view = release_ui.load_view("homepage_hifi", "different-release-label")
    assert callable(view.render)

def test_incomplete_module_is_still_rejected(monkeypatch, tmp_path):
    (tmp_path / "homepage_hifi.py").write_text("UI_REVISION = 'test'\n", encoding="utf-8")
    monkeypatch.setattr(release_ui, "__file__", str(tmp_path / "release_ui.py"))
    with pytest.raises(RuntimeError, match="missing render"):
        release_ui.load_view("homepage_hifi", "test")

def test_full_homepage_and_camera_navigation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    app.query_params["workspace"] = "影像识别"
    app.run()
    assert not app.exception
    assert len(app.get("camera_input")) == 1
    assert not any(u.key in ("vision_upload", "vision_native_camera", "vision_system_camera") for u in app.get("file_uploader"))
    app.radio(key="vision_source_mode").set_value("上传图片").run()
    assert not app.exception
    assert len(app.get("camera_input")) == 0
    assert sum(u.key == "vision_upload" for u in app.get("file_uploader")) == 1
    app.radio(key="vision_source_mode").set_value("现场拍照").run()
    assert not app.exception
    assert len(app.get("camera_input")) == 1
    assert not any("点击下方" in c.value or "建议避开逆光" in c.value for c in app.caption)
