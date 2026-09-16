from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
FIELD = ROOT / "src" / "globalhab_demo" / "field_visual.py"
LLM = ROOT / "src" / "globalhab_demo" / "real_training" / "result_interpreter.py"


def test_case_evidence_function_not_shadowed_in_app():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "evidence_rows":
                    bad.append(getattr(node, "lineno", None))
    assert not bad, f"evidence_rows function is shadowed at lines {bad}"
    text = APP.read_text(encoding="utf-8")
    assert "sandbox_evidence_rows" in text
    assert "rows = evidence_rows(current_case)" in text


def test_case_context_cards_use_project_kpi_style():
    app = APP.read_text(encoding="utf-8")
    field = FIELD.read_text(encoding="utf-8")
    assert 'key="field_task_from_research"' in app
    assert 'kpi_grid([' in app
    assert 'key="vision_case_context"' in field
    assert '_render_context_kpis([' in field


def test_workspace_hero_copy_regression():
    app = APP.read_text(encoding="utf-8")
    llm = LLM.read_text(encoding="utf-8")
    assert "OWN DATA WORKBENCH" in app
    assert "<h1>自有数据分析</h1>" in app
    assert "让大模型解释已经计算完成的科学结果，而不是替代模型计算" not in llm
