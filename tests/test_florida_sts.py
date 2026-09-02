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
    assert "SAMPLE_DATE >= DATE '2018-08-01'" in decoded
    assert "SAMPLE_DATE < DATE '2018-09-01'" in decoded
    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2018-08-15")


def test_hycom_ncss_url_and_netcdf_parser(tmp_path):
    import numpy as np
    from scipy.io import netcdf_file
    import globalhab_demo.florida_sts as florida_sts

    url = florida_sts.build_hycom_gom_url("2018-08-01", "2018-08-31")
    assert "GOMb0.04/reanalysis/2018/3z" in url
    assert "var=u" in url and "var=v" in url
    assert "west=272.800" in url and "east=280.000" in url
    assert "vertCoord=0" in url

    path = tmp_path / "hycom.nc"
    with netcdf_file(path, "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "hours since 2000-01-01 00:00:00"
        # 2018-08-01 and 2018-08-02
        origin = pd.Timestamp("2000-01-01")
        time[:] = [
            (pd.Timestamp("2018-08-01") - origin).total_seconds() / 3600,
            (pd.Timestamp("2018-08-02") - origin).total_seconds() / 3600,
        ]
        lat = ds.createVariable("lat", "f4", ("lat",))
        lat[:] = [25.0, 26.0]
        lon = ds.createVariable("lon", "f4", ("lon",))
        lon[:] = [273.0, 274.0]
        u = ds.createVariable("u", "f4", ("time", "lat", "lon"))
        v = ds.createVariable("v", "f4", ("time", "lat", "lon"))
        u[:] = np.full((2, 2, 2), 0.15, dtype="f4")
        v[:] = np.full((2, 2, 2), -0.05, dtype="f4")

    frame = florida_sts._hycom_netcdf_to_frame(path.read_bytes())
    assert len(frame) == 8
    assert frame["date"].min() == pd.Timestamp("2018-08-01")
    assert frame["date"].max() == pd.Timestamp("2018-08-02")
    assert frame["longitude"].min() == -87.0
    assert frame["longitude"].max() == -86.0
    assert frame["u_ms"].notna().all() and frame["v_ms"].notna().all()


def test_hycom_fetch_bisects_timeout_and_keeps_partial_rows(monkeypatch):
    import globalhab_demo.florida_sts as florida_sts

    calls = []

    def fake_chunk(start_date, end_date, lat_min, lat_max, lon_min, lon_max,
                   horiz_stride, time_stride, timeout=55):
        calls.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
        if pd.Timestamp(start_date) < pd.Timestamp(end_date):
            raise TimeoutError("synthetic timeout")
        day = pd.Timestamp(start_date).floor("D")
        return pd.DataFrame({
            "date": [day], "latitude": [27.5], "longitude": [-84.0],
            "u_ms": [0.1], "v_ms": [0.0],
        })

    monkeypatch.setattr(florida_sts, "_fetch_hycom_gom_chunk", fake_chunk)
    frame = florida_sts.fetch_hycom_gom_currents(
        "2018-08-01", "2018-08-04", chunk_days=4,
        lat_min=27.0, lat_max=28.0, lon_min=-85.0, lon_max=-83.0,
    )
    assert frame["date"].nunique() == 4
    assert len(calls) > 4  # initial block plus recursive subdivisions
    assert frame.attrs["requested_start"] == pd.Timestamp("2018-08-01")
    assert frame.attrs["requested_end"] == pd.Timestamp("2018-08-04")
