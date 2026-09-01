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
    "run_demo.py",
    "requirements.txt",
    "src/globalhab_demo/__init__.py",
    "src/globalhab_demo/workflow.py",
    "src/globalhab_demo/bio_response.py",
    "src/globalhab_demo/real_benchmark.py",
    "data/real_case/derived/sa_qpcr_observations.csv",
    "data/real_case_norway/derived/norway_hab_observations.csv",
)

REQUIRED_MODULES = (
    "globalhab_demo.workflow",
    "globalhab_demo.event_risk",
    "globalhab_demo.bio_response",
    "globalhab_demo.real_benchmark",
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
    if "3.7.1" not in version_text:
        raise SystemExit("VERSION.md does not declare v3.7.1")
    print(json.dumps({
        "status": "pass",
        "required_files": len(REQUIRED_FILES),
        "imported_modules": imported,
        "python": sys.version.split()[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
