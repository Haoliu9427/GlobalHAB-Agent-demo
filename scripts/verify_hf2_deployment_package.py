from pathlib import Path

root = Path(__file__).resolve().parents[1]
checks = {
    "flat_root_app": (root / "app.py").is_file(),
    "alternate_entrypoint": (root / "streamlit_app.py").is_file(),
    "unique_homepage": (root / "src/globalhab_demo/homepage_hifi.py").is_file(),
    "hf_css": (root / "assets/high_fidelity.css").is_file(),
    "build_id": (root / "BUILD_ID.txt").read_text(encoding="utf-8").strip() == "HF2-20260918",
}
app = (root / "app.py").read_text(encoding="utf-8")
ui = (root / "src/globalhab_demo/ui_system.py").read_text(encoding="utf-8")
home = (root / "src/globalhab_demo/homepage_hifi.py").read_text(encoding="utf-8")
checks.update({
    "app_uses_unique_homepage": "globalhab_demo.homepage_hifi" in app,
    "button_navigation": "st.button(" in ui and "workspace_mode\",\n                horizontal=True" not in ui,
    "no_sankey_homepage": "Sankey" not in home and "go.Sankey" not in home,
    "research_flow": "研究思路概览" in home and "hf-research-flow" in home,
    "sa_heatmap": "go.Heatmap" in home and "南澳真实事件回放" in home,
})
failed = [k for k, v in checks.items() if not v]
for key, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {key}")
if failed:
    raise SystemExit("HF2 package verification failed: " + ", ".join(failed))
print("HF2 package verification passed.")
