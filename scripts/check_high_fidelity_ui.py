from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

checks: dict[str, bool] = {}
app = (ROOT / "app.py").read_text(encoding="utf-8")
home = (ROOT / "src" / "globalhab_demo" / "homepage_hifi.py").read_text(encoding="utf-8")
ui = (ROOT / "src" / "globalhab_demo" / "ui_system.py").read_text(encoding="utf-8")
css = (ROOT / "assets" / "high_fidelity.css").read_text(encoding="utf-8")

checks["sidebar_collapsed_by_default"] = 'initial_sidebar_state="collapsed"' in app
checks["high_fidelity_loaded_last"] = 'assets" / "high_fidelity.css' in app and app.index('blue_theme.css') < app.index('high_fidelity.css')
checks["top_navigation_has_side_brand"] = all(x in ui for x in ['top-brand-side', 'nav_keys', 'st.button'])
checks["reference_banner_asset_exists"] = (ROOT / "assets" / "home_ocean_banner.png").is_file()
checks["reference_layout_upper_grid"] = 'hf-upper-grid' in home and 'hf-workspace-flow' in home
checks["real_sa_station_heatmap"] = 'sa_qpcr_observations.csv' in home and 'pivot_table' in home and 'k_cristata_cells_l' in home
checks["evidence_matrix_uses_real_metrics"] = all(x in home for x in ['mean_cte_bits','effect_per_1sd','detection_share','model_average_precision'])
checks["norway_is_compact_matrix"] = '_norway_panel_html' in home and 'hf-norway-table' in home
checks["homepage_no_bottom_jump_buttons"] = 'home_to_research' not in home and '_workspace_jump(' not in home
checks["map_explanatory_note_hidden"] = 'map-mode-note{display:none' in (ROOT / "assets" / "blue_theme.css").read_text(encoding="utf-8")
checks["responsive_panels_stay_contained"] = all(x in css for x in [
    ".hf-upper-grid>*,.hf-research-panel,.hf-case-panel",
    "box-sizing:border-box",
    "max-width:100%",
])
checks["responsive_flows_scroll_inside_cards"] = all(x in css for x in [
    ".hf-research-flow,.hf-workspace-flow",
    "overflow-x:auto",
])

result = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_passed": all(checks.values())}
out = ROOT / "validation" / "high_fidelity" / "high_fidelity_checks.json"
try:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
except OSError as exc:
    # Read-only deployments can still run the verification; persisting the
    # report is helpful but must not change the result of the checks.
    print(f"warning: could not persist {out.name}: {exc}")
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["all_passed"] else 1)
