from pathlib import Path
import types
from globalhab_demo.release_ui import load_view
ROOT = Path(__file__).resolve().parents[1]
REV = (ROOT / "BUILD_ID.txt").read_text().strip()

def test_homepage_evidence_and_rule_links():
    home = load_view("homepage_hifi", REV)
    assert len(home._agent_trace_figure(ROOT).data) > 0
    assert len(home._combined_evidence_figure(ROOT).data) >= 4
    assert "controller=rules" in home._workspace_flow_html(0, 0)
    assert home._read_json(ROOT / "outputs/norway_forward_benchmark_card.json")["model_average_precision"] > 0

def test_stale_import_does_not_override_current_view(monkeypatch):
    import sys
    old = types.ModuleType("globalhab_demo.scientific_agent_ui")
    old.UI_REVISION = "old"
    monkeypatch.setitem(sys.modules, old.__name__, old)
    ui = load_view("scientific_agent_ui", REV)
    assert ui.UI_REVISION == REV
    assert callable(ui.render)
    assert callable(ui._publish_scientific_result)

def test_connection_errors_do_not_expose_response_body():
    ui = load_view("scientific_agent_ui", REV)
    for code in (401, 402, 403, 404, 429, 503):
        hint = ui.connection_error_hint(ValueError(f"HTTP {code}: secret-private-body"))
        assert str(code) in hint
        assert "secret-private-body" not in hint
