import pandas as pd

from globalhab_demo.florida_sts import (
    field_quality_gate,
    run_forward_field_validation,
    run_retrospective_sts,
)


def _toy_field_data():
    dates = pd.date_range("2025-01-01", periods=120)
    stations = [
        ("A", 27.5, -84.0),
        ("B", 27.5, -83.7),
        ("C", 27.5, -83.4),
    ]
    pulse_a = set(range(5, 100, 21))
    obs = []
    current = []
    for i, date in enumerate(dates):
        for name, lat, lon in stations:
            cells = 1_000.0
            if name == "A" and i in pulse_a:
                cells = 90_000.0
            elif name == "B" and (i - 7) in pulse_a:
                cells = 220_000.0
            elif name == "C" and (i - 14) in pulse_a:
                cells = 240_000.0
            obs.append({
                "date": date,
                "station_id": name,
                "latitude": lat,
                "longitude": lon,
                "cell_count": cells,
            })
            current.append({
                "date": date,
                "latitude": lat,
                "longitude": lon,
                "u_ms": 0.0497,
                "v_ms": 0.0,
            })
    return pd.DataFrame(obs), pd.DataFrame(current)


def test_flow_constrained_retrospective_and_forward_gate():
    obs, current = _toy_field_data()
    quality = field_quality_gate(obs, current, event_threshold=100_000, strict_forward=True)
    assert quality["status"] == "ready"

    retrospective = run_retrospective_sts(obs, current, [3, 7, 14, 21], 100_000)
    assert retrospective["best_lag"] == 7

    forward = run_forward_field_validation(obs, current, [3, 7, 14, 21], 100_000, 0.30)
    assert forward["status"] == "evaluated"
    assert forward["selected_lag"] == 7
    summary = forward["test_summary"]
    assert summary["flow_ap"] > summary["reverse_ap"]
    assert summary["flow_ap"] > summary["no_flow_ap"]
