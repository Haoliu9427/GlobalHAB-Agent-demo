"""Load release UI source verbatim, isolated from retained imported UI modules."""
from pathlib import Path
import types
import logging

def load_view(name, revision):
    if name not in {"homepage_hifi", "research_entry", "scientific_agent_ui", "scientific_agent_results", "ui_system", "result_pool", "real_training.result_interpreter"}:
        raise ValueError("Unknown view")
    path = Path(__file__).parent / (name.replace(".", "/") + ".py")
    view = types.ModuleType("globalhab_demo._release_" + name)
    view.__package__ = "globalhab_demo" + ("." + name.rsplit(".", 1)[0] if "." in name else "")
    view.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), view.__dict__)
    required = {
        "homepage_hifi": ("render",),
        "research_entry": ("select_mode",),
        "scientific_agent_ui": ("render",),
        "scientific_agent_results": ("render_results",),
        "ui_system": ("render_top_navigation",),
        "result_pool": ("render_selection", "publish_scientific_result"),
        "real_training.result_interpreter": ("render",),
    }[name]
    missing = [attr for attr in required if not callable(getattr(view, attr, None))]
    if missing:
        raise RuntimeError(f"Incomplete page module {name}: missing {', '.join(missing)}")
    if name == "research_entry" and not getattr(view, "MODES", None):
        raise RuntimeError("Incomplete research entry: missing modes")
    if getattr(view, "UI_REVISION", None) != revision:
        logging.getLogger(__name__).warning(
            "Page release label differs: %s (page=%s, app=%s); loaded current source with required entrypoints",
            name, getattr(view, "UI_REVISION", None), revision,
        )
    return view
