"""Fail fast when a web entrypoint and its source tree are out of sync."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

REQUIRED_FILES = (
    "app.py",
    "PACKAGE_MANIFEST_SHA256.txt",
    "LLM_INTERPRETATION_GUIDE.md",
    "FIELD_VISUAL_SCREENING_GUIDE.md",
    "VISION_ROUTER_GUIDE.md",
    "CONTINUOUS_VISUAL_AGENT_GUIDE.md",
    "PUBLIC_VISUAL_DATA_SOURCES.md",
    "CASE_EVIDENCE_LOOP_GUIDE.md",
    "requirements-vision.txt",
    "run_demo.py",
    "requirements.txt",
    "src/globalhab_demo/__init__.py",
    "src/globalhab_demo/workflow.py",
    "src/globalhab_demo/bio_response.py",
    "src/globalhab_demo/real_benchmark.py",
    "src/globalhab_demo/sts_gated_tcn.py",
    "src/globalhab_demo/broad_benchmark.py",
    "src/globalhab_demo/bayesian_design.py",
    "src/globalhab_demo/florida_sts.py",
    "src/globalhab_demo/field_visual.py",
    "src/globalhab_demo/adaptive_visual.py",
    "src/globalhab_demo/visual_learning.py",
    "src/globalhab_demo/case_manager.py",
    "src/globalhab_demo/real_training/result_interpreter.py",
    "data/field_validation/field_observations_template.csv",
    "data/field_validation/field_currents_template.csv",
    "prompts/README.md",
    "prompts/AGENT_POLICY.md",
    "docs/MINIMAL_REPRODUCTION.md",
    "scripts/run_minimal_reproduction.py",
    "scripts/smoke_test.py",
    "scripts/run_agent_policy_benchmark.py",
    "scripts/run_broad_benchmark_audit.py",
    "scripts/run_florida_sts_validation.py",
    "scripts/check_field_visual.py",
    "scripts/check_adaptive_visual.py",
    "scripts/check_visual_learning.py",
    "scripts/check_case_loop.py",
    "scripts/train_visual_heads.py",
    "scripts/fetch_public_visual_data.py",
    "scripts/build_public_visual_adapter.py",
    "data/field_visual/visual_training_manifest_template.csv",
    "data/field_visual/public/PUBLIC_DATA_SOURCES.json",
    "data/field_visual/user_library/records.csv",
    "vision_models/active_model.json",
    "vision_models/public_baseline/model_card.json",
    "vision_models/README.md",
    "vision_models/heads/README.md",
    "scripts/run_field_forward_validation.py",
    "data/real_case/derived/sa_qpcr_observations.csv",
    "data/real_case_norway/derived/norway_hab_observations.csv",
)

REQUIRED_MODULES = (
    "globalhab_demo.workflow",
    "globalhab_demo.event_risk",
    "globalhab_demo.bio_response",
    "globalhab_demo.real_benchmark",
    "globalhab_demo.sts_gated_tcn",
    "globalhab_demo.broad_benchmark",
    "globalhab_demo.bayesian_design",
    "globalhab_demo.florida_sts",
    "globalhab_demo.field_visual",
    "globalhab_demo.adaptive_visual",
    "globalhab_demo.visual_learning",
    "globalhab_demo.case_manager",
    "globalhab_demo.real_training.result_interpreter",
)


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing release files: " + ", ".join(missing))
    imported = []
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
        imported.append(name)
    version_text = (ROOT / "VERSION.md").read_text(encoding="utf-8")
    if "4.1" not in version_text:
        raise SystemExit("VERSION.md does not declare v4.1")
    print(json.dumps({
        "status": "pass",
        "required_files": len(REQUIRED_FILES),
        "imported_modules": imported,
        "python": sys.version.split()[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
