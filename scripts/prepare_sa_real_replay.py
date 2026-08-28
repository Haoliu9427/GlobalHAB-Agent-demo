"""Prepare the bundled South Australia real-event replay dataset.

Inputs
------
1. Murray et al. qPCR workbook from Zenodo record 20227730 (CC BY 4.0).
2. NOAA/NCEI OISST v2.1 daily SST through ERDDAP (public-domain US data).

The script keeps the original qPCR workbook untouched, cleans reported values,
matches each sample to the nearest valid 0.25-degree ocean cell, and computes
1991-2020 calendar-day mean/p90 thresholds with an 11-day window. It does not
construct supervised negative labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ZENODO_RECORD = "https://doi.org/10.5281/zenodo.20227730"
PAPER_DOI = "https://doi.org/10.1038/s41559-026-03115-0"
OISST_PRODUCT = "NOAA/NCEI Optimum Interpolation SST v2.1"
ERDDAP = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    "ncdcOisst21Agg.csv"
)
BASELINE_CHUNKS = (
    ("1991-01-01", "1999-12-31"),
    ("2000-01-01", "2008-12-31"),
    ("2009-01-01", "2017-12-31"),
    ("2018-01-01", "2020-12-31"),
)
SPECIES_COLUMNS = {
    "K. brevisulcata (cell/L)": "k_brevisulcata_cells_l",
    "K. mikimotoi (cell/L)": "k_mikimotoi_cells_l",
    "K. papilionacea (cell/L)": "k_papilionacea_cells_l",
    "K. longicanalis (cell/L)": "k_longicanalis_cells_l",
    "K. brevis (cell/L)": "k_brevis_cells_l",
    "K. cristata (cell/L)": "k_cristata_cells_l",
    "K. hui (cell/L)": "k_hui_cells_l",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _erddap(query: str, retries: int = 5) -> tuple[pd.DataFrame, str]:
    # ERDDAP expects the projection constraint itself to be URL encoded.
    url = f"{ERDDAP}?{quote(query, safe=':,.-')}"
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "GlobalHAB-Agent/3.2"})
            with urlopen(request, timeout=120) as response:
                text = response.read().decode("utf-8")
            return pd.read_csv(io.StringIO(text), skiprows=[1]), url
        except Exception as exc:  # pragma: no cover - network retry path
            error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"OISST request failed: {url}") from error


def _grid_center(value: pd.Series) -> pd.Series:
    return np.round((value - 0.125) / 0.25) * 0.25 + 0.125


def clean_qpcr(workbook: Path) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name="Final_qPCR_data_integrated")
    raw.columns = [column.replace("\xa0", "").strip() for column in raw.columns]
    output = pd.DataFrame({
        "source_row": np.arange(1, len(raw) + 1),
        "site_number": raw["Site"],
        "sample_date": pd.to_datetime(raw["Date"]),
        "location": raw["Location"].astype(str).str.strip(),
        "sample_number": raw["Sample number"].astype(str).str.strip(),
        "sampler": raw["Sampler"].astype(str).str.strip(),
        "depth": raw["Depth"].astype(str).str.strip(),
        "latitude": -pd.to_numeric(raw["Latitude (°S)"], errors="coerce"),
        "longitude": pd.to_numeric(raw["Longitude (°E)"], errors="coerce"),
    })
    for source, destination in SPECIES_COLUMNS.items():
        reported = raw[source].astype(str).str.strip()
        output[destination] = pd.to_numeric(reported, errors="coerce").fillna(0.0)
        output[destination.replace("_cells_l", "_reported_not_detected")] = (
            reported.str.casefold().eq("not detected")
        )
    abundance_columns = list(SPECIES_COLUMNS.values())
    output["karenia_total_cells_l"] = output[abundance_columns].sum(axis=1)
    output["karenia_species_detected"] = output[abundance_columns].gt(0).sum(axis=1)
    output["k_cristata_detected"] = output["k_cristata_cells_l"].gt(0)
    output["k_cristata_share"] = np.divide(
        output["k_cristata_cells_l"], output["karenia_total_cells_l"],
        out=np.zeros(len(output), dtype=float), where=output["karenia_total_cells_l"].gt(0),
    )
    output["nearest_grid_latitude"] = _grid_center(output["latitude"])
    output["nearest_grid_longitude"] = _grid_center(output["longitude"])
    output["source_record"] = ZENODO_RECORD
    output["source_license"] = "CC BY 4.0"
    return output


def _event_rectangle() -> tuple[pd.DataFrame, str]:
    query = (
        "sst[(2024-09-01T12:00:00Z):1:(2025-09-30T12:00:00Z)]"
        "[(0.0):1:(0.0)][(-35.875):1:(-33.625)][(135.625):1:(139.125)]"
    )
    frame, url = _erddap(query)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return frame.dropna(subset=["sst"]), url


def assign_valid_ocean_cells(qpcr: pd.DataFrame, event_grid: pd.DataFrame) -> pd.DataFrame:
    output = qpcr.copy()
    available = event_grid[["latitude", "longitude"]].drop_duplicates().to_numpy()
    assignments = []
    for row in output.itertuples():
        distance = (
            (available[:, 0] - row.latitude) ** 2
            + ((available[:, 1] - row.longitude) * np.cos(np.deg2rad(row.latitude))) ** 2
        )
        assignments.append(available[int(np.argmin(distance))])
    assigned = np.asarray(assignments)
    output["oisst_grid_latitude"] = assigned[:, 0]
    output["oisst_grid_longitude"] = assigned[:, 1]
    return output


def _baseline_for_cell(latitude: float, longitude: float) -> tuple[pd.DataFrame, list[str]]:
    """Fetch the 1991-2020 baseline in bounded requests.

    ERDDAP servers commonly reject or time out on a single 30-year request.
    Nine-year-or-shorter chunks keep the adapter reproducible without changing
    the resulting climatology.
    """
    parts: list[pd.DataFrame] = []
    urls: list[str] = []
    for start, stop in BASELINE_CHUNKS:
        query = (
            f"sst[({start}T12:00:00Z):1:({stop}T12:00:00Z)]"
            f"[(0.0):1:(0.0)][({latitude}):1:({latitude})]"
            f"[({longitude}):1:({longitude})]"
        )
        frame, url = _erddap(query)
        parts.append(frame)
        urls.append(url)
    baseline = pd.concat(parts, ignore_index=True)
    baseline["time"] = (
        pd.to_datetime(baseline["time"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    return baseline.dropna(subset=["sst"]), urls


def _climate_day(date: pd.Series) -> np.ndarray:
    day = date.dt.dayofyear.to_numpy()
    after_february = date.dt.month.to_numpy() > 2
    leap = date.dt.is_leap_year.to_numpy()
    day = day - (leap & after_february)
    day[(date.dt.month.to_numpy() == 2) & (date.dt.day.to_numpy() == 29)] = 59
    return day.astype(int)


def climatology(baseline: pd.DataFrame) -> pd.DataFrame:
    work = baseline.copy()
    work["climate_day"] = _climate_day(work["time"])
    rows = []
    for day in range(1, 366):
        circular = np.minimum((work["climate_day"] - day).abs(), 365 - (work["climate_day"] - day).abs())
        values = work.loc[circular.le(5), "sst"].dropna().to_numpy()
        rows.append({
            "climate_day": day,
            "climatological_mean_sst_c": float(np.mean(values)),
            "climatological_p90_sst_c": float(np.quantile(values, 0.90)),
            "baseline_values": int(len(values)),
        })
    return pd.DataFrame(rows)


def prepare(root: Path, qpcr_only: bool = False) -> None:
    raw_dir = root / "data" / "real_case" / "raw"
    derived_dir = root / "data" / "real_case" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    workbook = raw_dir / "Figure2_Final_qPCR_data_integrated.xlsx"
    qpcr = clean_qpcr(workbook)
    if qpcr_only:
        qpcr_path = derived_dir / "sa_qpcr_observations.csv"
        qpcr.to_csv(qpcr_path, index=False)
        provenance = {
            "version": "1.0-qpcr",
            "created_by": "scripts/prepare_sa_real_replay.py --qpcr-only",
            "qpcr": {
                "paper": PAPER_DOI,
                "record": ZENODO_RECORD,
                "license": "CC BY 4.0",
                "raw_file": workbook.name,
                "raw_sha256": _sha256(workbook),
                "rows": int(len(qpcr)),
                "sampling_dates": int(qpcr["sample_date"].nunique()),
                "locations": int(qpcr["location"].nunique()),
                "derived_sha256": _sha256(qpcr_path),
            },
            "sst": {
                "status": "not_bundled_in_qpcr_only_build",
                "expected_product": OISST_PRODUCT,
                "adapter": "run this script without --qpcr-only when ERDDAP is available",
            },
            "label_boundary": (
                "Reported not-detected values are retained as assay results only. "
                "They are not interpreted as complete ecological negative labels."
            ),
        }
        (derived_dir / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(provenance, ensure_ascii=False, indent=2))
        return
    event_grid, event_url = _event_rectangle()
    qpcr = assign_valid_ocean_cells(qpcr, event_grid)
    cells = list(qpcr[["oisst_grid_latitude", "oisst_grid_longitude"]].drop_duplicates().itertuples(index=False, name=None))

    baselines: dict[tuple[float, float], tuple[pd.DataFrame, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_baseline_for_cell, float(latitude), float(longitude)): (float(latitude), float(longitude))
            for latitude, longitude in cells
        }
        for future in as_completed(futures):
            baselines[futures[future]] = future.result()

    daily_parts = []
    query_urls = [event_url]
    for latitude, longitude in cells:
        baseline, urls = baselines[(float(latitude), float(longitude))]
        query_urls.extend(urls)
        climate = climatology(baseline)
        daily = event_grid[
            event_grid["latitude"].eq(latitude) & event_grid["longitude"].eq(longitude)
        ].copy()
        daily["climate_day"] = _climate_day(daily["time"])
        daily = daily.merge(climate, on="climate_day", how="left", validate="many_to_one")
        daily["is_mhw"] = daily["sst"].gt(daily["climatological_p90_sst_c"])
        daily["mhw_intensity_c"] = np.where(
            daily["is_mhw"], daily["sst"] - daily["climatological_mean_sst_c"], 0.0
        )
        daily_parts.append(daily)
    oisst = pd.concat(daily_parts, ignore_index=True).rename(columns={
        "time": "date", "latitude": "oisst_grid_latitude",
        "longitude": "oisst_grid_longitude", "sst": "sst_c",
    })
    oisst = oisst.sort_values(["date", "oisst_grid_latitude", "oisst_grid_longitude"])

    qpcr = qpcr.merge(
        oisst[[
            "date", "oisst_grid_latitude", "oisst_grid_longitude", "sst_c",
            "climatological_mean_sst_c", "climatological_p90_sst_c", "is_mhw",
            "mhw_intensity_c",
        ]],
        left_on=["sample_date", "oisst_grid_latitude", "oisst_grid_longitude"],
        right_on=["date", "oisst_grid_latitude", "oisst_grid_longitude"],
        how="left", validate="many_to_one",
    ).drop(columns="date")

    qpcr_path = derived_dir / "sa_qpcr_observations.csv"
    oisst_path = derived_dir / "sa_oisst_daily.csv"
    qpcr.to_csv(qpcr_path, index=False)
    oisst.to_csv(oisst_path, index=False)
    provenance = {
        "version": "1.0",
        "created_by": "scripts/prepare_sa_real_replay.py",
        "qpcr": {
            "paper": PAPER_DOI,
            "record": ZENODO_RECORD,
            "license": "CC BY 4.0",
            "raw_file": workbook.name,
            "raw_sha256": _sha256(workbook),
            "rows": int(len(qpcr)),
            "sampling_dates": int(qpcr["sample_date"].nunique()),
            "locations": int(qpcr["location"].nunique()),
        },
        "sst": {
            "product": OISST_PRODUCT,
            "provider": "NOAA/NCEI",
            "spatial_resolution": "0.25 degree",
            "temporal_resolution": "daily",
            "baseline": "1991-2020",
            "threshold": "calendar-day p90 from +/-5-day pooled window",
            "queries": query_urls,
            "derived_rows": int(len(oisst)),
        },
        "outputs": {
            qpcr_path.name: _sha256(qpcr_path),
            oisst_path.name: _sha256(oisst_path),
        },
        "label_boundary": (
            "Reported not-detected values are retained as assay results only. "
            "They are not interpreted as complete ecological negative labels."
        ),
    }
    (derived_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--qpcr-only", action="store_true")
    args = parser.parse_args()
    prepare(Path(args.root).resolve(), qpcr_only=args.qpcr_only)


if __name__ == "__main__":
    main()
