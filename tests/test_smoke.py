import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo import project_synthetic_scenario


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


def test_synthetic_scenario_map_has_legible_spatial_output():
    result = project_synthetic_scenario(
        issue_date=pd.Timestamp("2026-08-16").date(),
        horizon_days=14,
        mhw_intensity_c=2.8,
        nitrate_mmol_m3=5.5,
        phosphate_mmol_m3=0.75,
        silicate_mmol_m3=7.0,
        transport_proxy=0.85,
    )
    assert len(result) == 5
    assert result["候选海区"].nunique() == 5
    assert result["综合风险指数"].between(0, 100).all()
    assert result["预计藻华强度指数"].between(0, 100).all()
    assert result["预计时间"].eq("2026-08-30").all()
