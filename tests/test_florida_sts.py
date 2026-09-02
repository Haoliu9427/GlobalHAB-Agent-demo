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


def test_habsos_fetch_enforces_sample_date_window(monkeypatch):
    import json
    from urllib.parse import unquote_plus
    import globalhab_demo.florida_sts as florida_sts

    captured = {"url": None}

    inside = int(pd.Timestamp("2018-08-15", tz="UTC").timestamp() * 1000)
    outside = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)
    payload = {
        "features": [
            {"attributes": {
                "OBJECTID": 1, "LONGITUDE": -84.0, "LATITUDE": 27.5,
                "STATE_ID": "FL", "SAMPLE_DATE": inside,
                "GENUS": "Karenia", "SPECIES": "brevis", "CELLCOUNT": 150000,
            }},
            {"attributes": {
                "OBJECTID": 2, "LONGITUDE": -84.1, "LATITUDE": 27.6,
                "STATE_ID": "FL", "SAMPLE_DATE": outside,
                "GENUS": "Karenia", "SPECIES": "brevis", "CELLCOUNT": 150000,
            }},
        ]
    }

    def fake_download(url, timeout=45):
        captured["url"] = url
        return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(florida_sts, "_download_bytes", fake_download)
    result = florida_sts.fetch_habsos("2018-08-01", "2018-08-31", max_records=100)
    decoded = unquote_plus(captured["url"])
    assert "SAMPLE_DATE >= TIMESTAMP '2018-08-01 00:00:00'" in decoded
    assert "SAMPLE_DATE < TIMESTAMP '2018-09-01 00:00:00'" in decoded
    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2018-08-15")
