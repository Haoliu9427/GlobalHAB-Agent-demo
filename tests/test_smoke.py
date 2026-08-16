import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_and_writes_auditable_outputs(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_demo.py"),
            "--config",
            "config/demo.json",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    log = pd.read_csv(tmp_path / "agent_log.csv")
    card = json.loads((tmp_path / "discovery_card.json").read_text(encoding="utf-8"))
    assert len(log) == 8
    assert log["budget_remaining"].iloc[-1] == 0
    assert card["validation"]["random_split_used"] is False
    assert card["synthetic_ground_truth"]["recovered_by_agent"] is True
    assert (tmp_path / "risk_predictions.csv").exists()
