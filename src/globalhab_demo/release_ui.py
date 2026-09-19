"""Load release UI source verbatim, isolated from retained imported UI modules."""
from pathlib import Path
import types

def load_view(name, revision):
    if name not in {"homepage_hifi", "research_entry", "scientific_agent_ui", "scientific_agent_results", "ui_system", "result_pool", "real_training.result_interpreter"}:
        raise ValueError("Unknown view")
    path = Path(__file__).parent / (name.replace(".", "/") + ".py")
    view = types.ModuleType("globalhab_demo._release_" + name)
    view.__package__ = "globalhab_demo" + ("." + name.rsplit(".", 1)[0] if "." in name else "")
    view.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), view.__dict__)
    if getattr(view, "UI_REVISION", None) != revision:
        raise RuntimeError("页面模块版本不一致，请完整更新工程并重启应用。")
    return view
