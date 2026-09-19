"""Load release UI source verbatim, isolated from retained imported UI modules."""
from pathlib import Path
import types
import logging
import hashlib
import sys

def load_view(name, revision):
    if name not in {"homepage_hifi", "research_entry", "scientific_agent_ui", "scientific_agent_results", "ui_system", "result_pool", "real_training.result_interpreter", "field_visual"}:
        raise ValueError("Unknown view")
    path = Path(__file__).parent / (name.replace(".", "/") + ".py")
    source = path.read_text(encoding="utf-8")
    module_name = "globalhab_demo._release_" + name
    if name == "field_visual":
        module_name += "_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    view = types.ModuleType(module_name)
    view.__package__ = "globalhab_demo" + ("." + name.rsplit(".", 1)[0] if "." in name else "")
    view.__file__ = str(path)
    # Dataclasses resolve annotations through sys.modules during execution.
    previous = sys.modules.get(module_name)
    if name == "field_visual":
        sys.modules[module_name] = view
    try:
        exec(compile(source, str(path), "exec"), view.__dict__)
    except BaseException:
        if name == "field_visual":
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        raise
    required = {
        "homepage_hifi": ("render",),
        "field_visual": ("render", "load_image"),
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
    if name != "field_visual" and getattr(view, "UI_REVISION", None) != revision:
        logging.getLogger(__name__).warning(
            "Page release label differs: %s (page=%s, app=%s); loaded current source with required entrypoints",
            name, getattr(view, "UI_REVISION", None), revision,
        )
    return view
