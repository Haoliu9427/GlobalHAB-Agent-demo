from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

checks: dict[str, bool] = {}
app = (ROOT / "app.py").read_text(encoding="utf-8")
home = (ROOT / "src" / "globalhab_demo" / "homepage.py").read_text(encoding="utf-8")
ui = (ROOT / "src" / "globalhab_demo" / "ui_system.py").read_text(encoding="utf-8")
css = (ROOT / "assets" / "high_fidelity.css").read_text(encoding="utf-8")

checks["sidebar_collapsed_by_default"] = 'initial_sidebar_state="collapsed"' in app
checks["high_fidelity_loaded_last"] = 'assets" / "high_fidelity.css' in app and app.index('blue_theme.css') < app.index('high_fidelity.css')
checks["top_navigation_has_side_brand"] = 'top-brand-side' in ui and 'format_func=' in ui
checks["reference_banner_asset_exists"] = (ROOT / "assets" / "home_ocean_banner.png").is_file()
checks["reference_layout_upper_grid"] = 'hf-upper-grid' in home and 'hf-workspace-flow' in home
checks["real_sa_station_heatmap"] = 'sa_qpcr_observations.csv' in home and 'pivot_table' in home and 'k_cristata_cells_l' in home
checks["evidence_matrix_uses_real_metrics"] = all(x in home for x in ['mean_cte_bits','effect_per_1sd','detection_share','model_average_precision'])
checks["norway_is_compact_matrix"] = '_norway_panel_html' in home and 'hf-norway-table' in home
checks["homepage_no_bottom_jump_buttons"] = 'home_to_research' not in home and '_workspace_jump(' not in home
checks["map_explanatory_note_hidden"] = 'map-mode-note{display:none' in (ROOT / "assets" / "blue_theme.css").read_text(encoding="utf-8")
checks["reference_preview_exists"] = (ROOT / "validation" / "high_fidelity" / "homepage_preview_1448.png").is_file()
checks["reference_preview_html_exists"] = (ROOT / "validation" / "high_fidelity" / "homepage_preview_light.html").is_file()

result = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_passed": all(checks.values())}
out = ROOT / "validation" / "high_fidelity" / "high_fidelity_checks.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["all_passed"] else 1)
