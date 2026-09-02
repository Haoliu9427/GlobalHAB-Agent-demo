import json
import pandas as pd

from globalhab_demo import florida_sts


def test_coastwatch_json_parser_and_chunk_fetch(monkeypatch):
    def fake_download(url: str, timeout: int = 60):
        # Return two finite grid rows for every requested chunk.
        payload = {
            "table": {
                "columnNames": ["time", "latitude", "longitude", "u_current", "v_current"],
                "columnTypes": ["String", "float", "float", "double", "double"],
                "columnUnits": ["UTC", "degrees_north", "degrees_east", "m/s", "m/s"],
                "rows": [
                    ["2018-08-01T00:00:00Z", 25.125, -85.125, 0.15, -0.04],
                    ["2018-08-02T00:00:00Z", 25.125, -85.125, 0.12, -0.03],
                ],
            }
        }
        return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(florida_sts, "_download_bytes", fake_download)
    out = florida_sts.fetch_coastwatch_currents("2018-08-01", "2018-08-02", spatial_stride=2)
    assert len(out) == 2
    assert out["date"].min() == pd.Timestamp("2018-08-01")
    assert out["u_ms"].notna().all()
    assert out["v_ms"].notna().all()


def test_coastwatch_live_fetch_raises_clear_error_when_all_chunks_fail(monkeypatch):
    def fake_download(url: str, timeout: int = 60):
        raise RuntimeError("synthetic network failure")

    monkeypatch.setattr(florida_sts, "_download_bytes", fake_download)
    try:
        florida_sts.fetch_coastwatch_currents("2018-08-01", "2018-08-10")
    except RuntimeError as exc:
        text = str(exc)
        assert "NOAA CoastWatch流场读取失败" in text
        assert "synthetic network failure" in text
    else:
        raise AssertionError("expected RuntimeError")


def test_coastwatch_falls_back_to_csv_when_json_fails(monkeypatch):
    calls = []
    def fake_download(url: str, timeout: int = 60):
        calls.append(url)
        if '.json?' in url:
            raise RuntimeError('json unavailable')
        return (
            'time,latitude,longitude,u_current,v_current\n'
            '2018-08-01T00:00:00Z,25.125,-85.125,0.15,-0.04\n'
        ).encode('utf-8')

    monkeypatch.setattr(florida_sts, '_download_bytes', fake_download)
    out = florida_sts.fetch_coastwatch_currents('2018-08-01', '2018-08-01')
    assert len(out) == 1
    assert any('.json?' in u for u in calls)
    assert any('.csv?' in u for u in calls)
