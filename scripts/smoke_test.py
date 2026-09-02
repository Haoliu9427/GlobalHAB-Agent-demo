"""One-command release smoke test for the public repository."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run([py, "-m", "compileall", "-q", "app.py", "run_demo.py", "scripts", "src", "tests"])
    run([py, "scripts/verify_release.py"])
    run([
        py, "-m", "pytest", "-q",
        "tests/test_release_smoke_fast.py",
        "tests/test_bayesian_design.py",
        "tests/test_florida_sts.py",
        "tests/test_broad_benchmark.py",
    ])
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
