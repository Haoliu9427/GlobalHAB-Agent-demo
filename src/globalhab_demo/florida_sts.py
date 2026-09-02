"""Florida/Gulf of Mexico real-data STS retrospective and field-forward adapters.

Stage 1 uses verified Karenia brevis observations (NOAA HABSOS) together with a
continuous surface-current field.  The bundled live adapter targets NOAA's ArcGIS
HABSOS service and the public NOAA CoastWatch ERDDAP geostrophic-current product.
Users may instead upload a current CSV exported from HYCOM, Copernicus Marine,
HF-radar or another source.

Stage 2 accepts field observations/current files from future cruises or farm/station
partnerships.  A training-period lag is selected first and then evaluated on a later
forward block.  The flow calculation is deliberately first-order and is presented as
flow-constrained evidence, not a Lagrangian particle-tracking or causal proof.
"""

from __future__ import annotations

from io import BytesIO, StringIO
import json
from math import cos, exp, radians
from typing import Iterable
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


HABSOS_QUERY_URL = (
    "https://gis.ncdc.noaa.gov/arcgis/rest/services/"
    "ms/HABSOS_CellCounts/MapServer/0/query"
)
COASTWATCH_DATASET = "noaacwBLENDEDNRTcurrentsDaily"
COASTWATCH_ERDDAP = f"https://coastwatch.noaa.gov/erddap/griddap/{COASTWATCH_DATASET}.csvp"
DEFAULT_LAGS = (3, 7, 14, 21, 30)
LIVE_ADAPTER_REVISION = "2026-09-02-r2"

PUBLIC_SOURCE_CATALOG = pd.DataFrame([
    {
        "source": "NOAA HABSOS",
        "role": "Karenia brevis cell counts + associated field observations",
        "access": "ArcGIS REST / NCEI archive",
        "status": "live adapter",
    },
    {
        "source": "NOAA CoastWatch surface currents",
        "role": "daily geostrophic u/v current field",
        "access": "ERDDAP",
        "status": "live adapter; upload fallback",
    },
    {
        "source": "HYCOM GOM reanalysis",
        "role": "1/25° Gulf current/temperature/salinity reanalysis",
        "access": "THREDDS/NCSS/OPeNDAP",
        "status": "supported through exported CSV upload",
    },
    {
        "source": "Copernicus Marine",
        "role": "global analysis/reanalysis u/v + ocean environment",
        "access": "Marine Data Store",
        "status": "supported through exported CSV upload",
    },
    {
        "source": "GCOOS / IOOS HF radar",
        "role": "high-frequency coastal surface currents",
        "access": "ERDDAP / NCEI archive",
        "status": "supported through exported CSV upload",
    },
])


def _download_bytes(url: str, timeout: int = 45) -> bytes:
    request = Request(url, headers={"User-Agent": "GlobalHAB-Agent/4.1 research prototype"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed public endpoints
        return response.read()


def fetch_habsos(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    state_id: str = "FL",
    max_records: int = 12000,
) -> pd.DataFrame:
    """Fetch HABSOS cell-count records using the public ArcGIS REST service."""
    start = pd.Timestamp(start_date, tz="UTC") if pd.Timestamp(start_date).tzinfo is None else pd.Timestamp(start_date)
    end = pd.Timestamp(end_date, tz="UTC") if pd.Timestamp(end_date).tzinfo is None else pd.Timestamp(end_date)
    # HABSOS Cell Counts is a static feature layer rather than a time-enabled
    # layer.  Therefore the ArcGIS ``time=`` parameter is not relied on here;
    # SAMPLE_DATE is constrained explicitly in the SQL WHERE clause and the
    # returned frame is filtered once more locally below.
    start_day = start.tz_convert("UTC").strftime("%Y-%m-%d")
    end_exclusive = (end.tz_convert("UTC").floor("D") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # ArcGIS standardized SQL supports DATE literals for esriFieldTypeDate.
    # The selected window is day-based, so DATE avoids unnecessary timezone/time-of-day ambiguity.
    date_where = (
        f"SAMPLE_DATE >= DATE '{start_day}' AND "
        f"SAMPLE_DATE < DATE '{end_exclusive}'"
    )
    fields = [
        "OBJECTID", "LONGITUDE", "LATITUDE", "DESCRIPTION", "STATE_ID",
        "SAMPLE_DATE", "SAMPLE_DEPTH", "GENUS", "SPECIES", "CATEGORY",
        "CELLCOUNT", "CELLCOUNT_UNIT", "CELLCOUNT_QA", "SALINITY",
        "WATER_TEMP", "WIND_DIR", "WIND_SPEED", "QA_COMMENT",
    ]
    rows: list[dict[str, object]] = []
    offset = 0
    page_size = 2000
    while offset < max_records:
        params = {
            "where": f"STATE_ID='{state_id}' AND {date_where}",
            "outFields": ",".join(fields),
            "returnGeometry": "false",
            "orderByFields": "SAMPLE_DATE ASC",
            "resultOffset": offset,
            "resultRecordCount": min(page_size, max_records - offset),
            "f": "json",
        }
        payload = json.loads(_download_bytes(f"{HABSOS_QUERY_URL}?{urlencode(params)}").decode("utf-8"))
        if "error" in payload:
            raise RuntimeError(f"HABSOS service error: {payload['error']}")
        features = payload.get("features", [])
        rows.extend([feature.get("attributes", {}) for feature in features])
        if len(features) < page_size:
            break
        offset += len(features)
    result = normalize_habsos(pd.DataFrame(rows))
    # Defensive post-filter: public services can ignore unsupported temporal
    # parameters or return cached rows.  Never let observations outside the
    # user-selected window enter the validation.
    if not result.empty:
        start_local = pd.Timestamp(start).tz_convert(None).floor("D")
        end_local = pd.Timestamp(end).tz_convert(None).floor("D")
        result = result[result["date"].between(start_local, end_local)].copy()
        if "state_id" in result.columns:
            state_mask = result["state_id"].astype(str).str.upper().eq(str(state_id).upper())
            if state_mask.any():
                result = result[state_mask].copy()
    result = result.sort_values("date").reset_index(drop=True)
    # Never return historical rows outside the requested window. If the public
    # service changes query behaviour, an empty frame is safer than stale evidence.
    if not result.empty:
        if result["date"].min() < pd.Timestamp(start).tz_convert(None).floor("D") or result["date"].max() > pd.Timestamp(end).tz_convert(None).floor("D"):
            raise RuntimeError("HABSOS returned observations outside the requested date window")
    return result


def build_coastwatch_url(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    lat_min: float = 24.125,
    lat_max: float = 30.875,
    lon_min: float = -86.875,
    lon_max: float = -80.125,
    spatial_stride: int = 2,
) -> str:
    start = pd.Timestamp(start_date).strftime("%Y-%m-%dT00:00:00Z")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%dT00:00:00Z")
    cube = (
        f"[({start}):1:({end})]"
        f"[({lat_min:.3f}):{int(spatial_stride)}:({lat_max:.3f})]"
        f"[({lon_min:.3f}):{int(spatial_stride)}:({lon_max:.3f})]"
    )
    query = f"u_current{cube},v_current{cube}"
    return f"{COASTWATCH_ERDDAP}?{quote(query, safe='[]():,=.-TZ_')}"


def fetch_coastwatch_currents(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    spatial_stride: int = 2,
) -> pd.DataFrame:
    """Fetch a Florida west-shelf daily surface-current subset from NOAA ERDDAP."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be on/after start_date")
    if (end - start).days > 240:
        raise ValueError("live current fetch is limited to 240 days per request")
    raw = _download_bytes(build_coastwatch_url(start, end, spatial_stride=spatial_stride), timeout=60)
    return normalize_current_frame(pd.read_csv(BytesIO(raw)))


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    for alias in aliases:
        key = alias.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    for key, original in normalized.items():
        for alias in aliases:
            if key.startswith(alias.strip().lower().replace(" ", "_")):
                return original
    return None


def normalize_habsos(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "date", "latitude", "longitude", "cell_count", "category", "salinity",
            "water_temp_c", "sample_depth", "state_id", "genus", "species",
        ])
    mapping = {
        "date": ["sample_date", "date", "eventdate", "timestamp"],
        "latitude": ["latitude", "lat", "decimallatitude"],
        "longitude": ["longitude", "lon", "decimallongitude"],
        "cell_count": ["cellcount", "cell_count", "organismquantity", "cells_l"],
        "category": ["category", "description"],
        "salinity": ["salinity"],
        "water_temp_c": ["water_temp", "water_temperature", "water_temp_c", "temperature"],
        "sample_depth": ["sample_depth", "depth"],
        "state_id": ["state_id", "state"],
        "genus": ["genus"],
        "species": ["species", "scientificname"],
    }
    out = pd.DataFrame(index=frame.index)
    for target, aliases in mapping.items():
        column = _find_column(frame, aliases)
        out[target] = frame[column] if column is not None else np.nan
    # ArcGIS dates are Unix milliseconds; CSV/DwC dates are strings.
    numeric_date = pd.to_numeric(out["date"], errors="coerce")
    parsed_numeric = pd.to_datetime(numeric_date, unit="ms", errors="coerce", utc=True)
    parsed_text = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["date"] = parsed_numeric.fillna(parsed_text).dt.tz_convert(None).dt.floor("D")
    for c in ["latitude", "longitude", "cell_count", "salinity", "water_temp_c", "sample_depth"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["cell_count"] = out["cell_count"].clip(lower=0)
    out = out.dropna(subset=["date", "latitude", "longitude", "cell_count"]).copy()
    # Florida/Gulf workflow focuses on K. brevis and excludes obvious east-coast records.
    genus = out["genus"].astype(str).str.lower()
    species = out["species"].astype(str).str.lower()
    taxon_mask = genus.str.contains("karenia", na=False) | species.str.contains("brevis", na=False)
    if taxon_mask.any():
        out = out[taxon_mask]
    return out.sort_values("date").reset_index(drop=True)


def normalize_current_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "date": ["time", "date", "timestamp", "sample_date"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon"],
        "u_ms": ["u_current", "u", "uo", "eastward_sea_water_velocity"],
        "v_ms": ["v_current", "v", "vo", "northward_sea_water_velocity"],
    }
    out = pd.DataFrame(index=frame.index)
    for target, aliases in mapping.items():
        column = _find_column(frame, aliases)
        if column is None:
            raise ValueError(f"current file missing required column for {target}: {aliases}")
        out[target] = frame[column]
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")
    for c in ["latitude", "longitude", "u_ms", "v_ms"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().copy()
    # Plausibility gate prevents common unit mistakes (e.g. cm/s supplied as m/s).
    out = out[(out["u_ms"].abs() <= 5.0) & (out["v_ms"].abs() <= 5.0)]
    return out.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)


def normalize_field_observations(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_habsos(frame)
    # Preserve optional field-study identifiers and toxin/environment columns.
    aliases = {
        "station_id": ["station_id", "station", "site_id", "site"],
        "toxin_value": ["toxin_value", "toxin", "brevetoxin", "toxin_concentration"],
        "toxin_unit": ["toxin_unit", "toxin_units"],
        "dissolved_oxygen_mg_l": ["dissolved_oxygen_mg_l", "do_mg_l", "dissolved_oxygen"],
        "nitrate_mmol_m3": ["nitrate_mmol_m3", "nitrate", "no3"],
        "phosphate_mmol_m3": ["phosphate_mmol_m3", "phosphate", "po4"],
        "silicate_mmol_m3": ["silicate_mmol_m3", "silicate", "sio4"],
        "chlorophyll": ["chlorophyll", "chlorophyll_a", "chla"],
    }
    original = frame.reset_index(drop=True)
    # Align optional fields by date/lat/lon when normalize_habsos filtered rows.
    key = pd.DataFrame(index=original.index)
    date_col = _find_column(original, ["sample_date", "date", "eventdate", "timestamp"])
    lat_col = _find_column(original, ["latitude", "lat", "decimallatitude"])
    lon_col = _find_column(original, ["longitude", "lon", "decimallongitude"])
    if date_col and lat_col and lon_col:
        key["date"] = pd.to_datetime(original[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")
        key["latitude"] = pd.to_numeric(original[lat_col], errors="coerce")
        key["longitude"] = pd.to_numeric(original[lon_col], errors="coerce")
        for target, names in aliases.items():
            col = _find_column(original, names)
            key[target] = original[col] if col is not None else np.nan
        out = out.merge(key, on=["date", "latitude", "longitude"], how="left")
    return out


def attach_nearest_current(observations: pd.DataFrame, currents: pd.DataFrame) -> pd.DataFrame:
    work = observations.copy()
    work["u_ms"] = np.nan
    work["v_ms"] = np.nan
    work["current_match_km"] = np.nan
    current_groups = {date: group for date, group in currents.groupby("date")}
    for date, indices in work.groupby("date").groups.items():
        grid = current_groups.get(pd.Timestamp(date))
        if grid is None or grid.empty:
            continue
        obs = work.loc[list(indices), ["latitude", "longitude"]].to_numpy(float)
        pts = grid[["latitude", "longitude"]].to_numpy(float)
        lat0 = np.deg2rad(obs[:, 0:1])
        dx = (pts[None, :, 1] - obs[:, None, 1]) * 111.0 * np.cos(lat0)
        dy = (pts[None, :, 0] - obs[:, None, 0]) * 111.0
        dist = np.sqrt(dx * dx + dy * dy)
        nearest = np.argmin(dist, axis=1)
        rows = grid.iloc[nearest]
        work.loc[list(indices), "u_ms"] = rows["u_ms"].to_numpy()
        work.loc[list(indices), "v_ms"] = rows["v_ms"].to_numpy()
        work.loc[list(indices), "current_match_km"] = dist[np.arange(len(nearest)), nearest]
    return work


def _pair_score(source: pd.Series, target: pd.Series, lag_days: int, reverse: bool = False) -> tuple[float, float, float]:
    mean_lat = 0.5 * (float(source.latitude) + float(target.latitude))
    dx = (float(target.longitude) - float(source.longitude)) * 111.0 * cos(radians(mean_lat))
    dy = (float(target.latitude) - float(source.latitude)) * 111.0
    distance = sqrt_safe(dx * dx + dy * dy)
    factor = -1.0 if reverse else 1.0
    predicted_dx = factor * float(source.u_ms) * lag_days * 86400.0 / 1000.0
    predicted_dy = factor * float(source.v_ms) * lag_days * 86400.0 / 1000.0
    error = sqrt_safe((dx - predicted_dx) ** 2 + (dy - predicted_dy) ** 2)
    return float(exp(-error / 90.0)), float(error), float(distance)


def sqrt_safe(value: float) -> float:
    return float(np.sqrt(max(0.0, value)))


def build_flow_matched_pairs(
    observations: pd.DataFrame,
    currents: pd.DataFrame,
    lag_days: int,
    event_threshold: float = 100_000.0,
    date_tolerance_days: int = 2,
    max_source_distance_km: float = 450.0,
) -> pd.DataFrame:
    obs = attach_nearest_current(observations, currents)
    obs = obs.dropna(subset=["u_ms", "v_ms"]).copy()
    if obs.empty:
        return pd.DataFrame()
    by_date = {pd.Timestamp(date): group for date, group in obs.groupby("date")}
    rows: list[dict[str, object]] = []
    for target in obs.itertuples(index=False):
        target_date = pd.Timestamp(target.date)
        candidate_parts = []
        for delta in range(-date_tolerance_days, date_tolerance_days + 1):
            source_date = target_date - pd.Timedelta(days=lag_days + delta)
            if source_date in by_date:
                candidate_parts.append(by_date[source_date])
        if not candidate_parts:
            continue
        sources = pd.concat(candidate_parts, ignore_index=True)
        best_flow = None
        best_reverse = None
        best_no_flow = None
        for source in sources.itertuples(index=False):
            flow_weight, flow_error, distance = _pair_score(source, target, lag_days, reverse=False)
            reverse_weight, reverse_error, _ = _pair_score(source, target, lag_days, reverse=True)
            if distance > max_source_distance_km:
                continue
            log_cells = float(np.log1p(max(0.0, float(source.cell_count))))
            no_flow_weight = float(exp(-distance / 150.0))
            flow_value = log_cells * flow_weight
            reverse_value = log_cells * reverse_weight
            no_flow_value = log_cells * no_flow_weight
            candidate = (flow_value, source, flow_error, distance)
            reverse_candidate = (reverse_value, source, reverse_error, distance)
            no_flow_candidate = (no_flow_value, source, distance)
            if best_flow is None or candidate[0] > best_flow[0]:
                best_flow = candidate
            if best_reverse is None or reverse_candidate[0] > best_reverse[0]:
                best_reverse = reverse_candidate
            if best_no_flow is None or no_flow_candidate[0] > best_no_flow[0]:
                best_no_flow = no_flow_candidate
        if best_flow is None:
            continue
        source = best_flow[1]
        rows.append({
            "target_date": target_date,
            "target_latitude": float(target.latitude),
            "target_longitude": float(target.longitude),
            "target_cell_count": float(target.cell_count),
            "target_event": int(float(target.cell_count) >= event_threshold),
            "lag_days": int(lag_days),
            "source_date": pd.Timestamp(source.date),
            "source_latitude": float(source.latitude),
            "source_longitude": float(source.longitude),
            "source_cell_count": float(source.cell_count),
            "flow_signal": float(best_flow[0]),
            "reverse_signal": float(best_reverse[0]) if best_reverse else np.nan,
            "no_flow_signal": float(best_no_flow[0]) if best_no_flow else np.nan,
            "flow_path_error_km": float(best_flow[2]),
            "source_target_distance_km": float(best_flow[3]),
            "source_u_ms": float(source.u_ms),
            "source_v_ms": float(source.v_ms),
            "source_current_speed_ms": sqrt_safe(float(source.u_ms) ** 2 + float(source.v_ms) ** 2),
        })
    return pd.DataFrame(rows)


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def summarize_lag_pairs(pairs: pd.DataFrame) -> dict[str, float]:
    if pairs.empty:
        return {"samples": 0, "events": 0, "flow_ap": np.nan, "reverse_ap": np.nan, "no_flow_ap": np.nan}
    y = pairs["target_event"].to_numpy(int)
    flow_ap = _safe_ap(y, pairs["flow_signal"].to_numpy(float))
    reverse_ap = _safe_ap(y, pairs["reverse_signal"].to_numpy(float))
    no_flow_ap = _safe_ap(y, pairs["no_flow_signal"].to_numpy(float))
    return {
        "samples": int(len(pairs)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "flow_ap": flow_ap,
        "reverse_ap": reverse_ap,
        "no_flow_ap": no_flow_ap,
        "flow_vs_no_flow": float(flow_ap - no_flow_ap) if np.isfinite(flow_ap) and np.isfinite(no_flow_ap) else np.nan,
        "flow_vs_reverse": float(flow_ap - reverse_ap) if np.isfinite(flow_ap) and np.isfinite(reverse_ap) else np.nan,
        "median_path_error_km": float(pairs["flow_path_error_km"].median()),
    }


def run_retrospective_sts(
    observations: pd.DataFrame,
    currents: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    event_threshold: float = 100_000.0,
) -> dict[str, object]:
    observation_quality = field_quality_gate(observations, currents, event_threshold=event_threshold, strict_forward=False)
    lag_rows = []
    pairs_by_lag: dict[int, pd.DataFrame] = {}
    for lag in lags:
        pairs = build_flow_matched_pairs(observations, currents, int(lag), event_threshold)
        pairs_by_lag[int(lag)] = pairs
        lag_rows.append({"lag_days": int(lag), **summarize_lag_pairs(pairs)})
    summary = pd.DataFrame(lag_rows)
    valid = summary.dropna(subset=["flow_ap"])
    best_lag = int(valid.sort_values(["flow_ap", "flow_vs_reverse"], ascending=False).iloc[0]["lag_days"]) if not valid.empty else None
    return {
        "lag_summary": summary,
        "best_lag": best_lag,
        "best_pairs": pairs_by_lag.get(best_lag, pd.DataFrame()) if best_lag is not None else pd.DataFrame(),
        "quality": observation_quality,
        "boundary": (
            "Flow-constrained retrospective association using first-order current displacement. "
            "It is not full particle tracking and does not establish causal transport."
        ),
    }


def field_quality_gate(
    observations: pd.DataFrame,
    currents: pd.DataFrame,
    event_threshold: float = 100_000.0,
    strict_forward: bool = True,
) -> dict[str, object]:
    reasons = []
    obs = observations.dropna(subset=["date", "latitude", "longitude", "cell_count"]).copy()
    unique_dates = int(obs["date"].nunique()) if not obs.empty else 0
    event_count = int((obs["cell_count"] >= event_threshold).sum()) if not obs.empty else 0
    location_count = int(obs[["latitude", "longitude"]].round(3).drop_duplicates().shape[0]) if not obs.empty else 0
    if len(obs) < (120 if strict_forward else 60):
        reasons.append("HAB/field observations are too few for the requested validation mode")
    if unique_dates < (25 if strict_forward else 12):
        reasons.append("sampling dates are insufficient")
    if location_count < 3:
        reasons.append("at least three spatial locations are required")
    if event_count < 5:
        reasons.append("too few event observations at the selected cell-count threshold")
    current_series = pd.to_datetime(currents.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna().dt.floor("D")
    current_dates = set(current_series)
    obs_dates = set(pd.to_datetime(obs["date"], errors="coerce").dropna().dt.floor("D")) if len(obs) else set()
    overlap_dates = obs_dates & current_dates
    overlap = float(len(overlap_dates) / len(obs_dates)) if obs_dates else 0.0
    if overlap < 0.45:
        reasons.append("current-field temporal coverage is below 45% of observation dates")
    date_span = int((obs["date"].max() - obs["date"].min()).days) if len(obs) else 0
    if strict_forward and date_span < 60:
        reasons.append("time span is shorter than 60 days")
    return {
        "status": "ready" if not reasons else "defer",
        "observations": int(len(obs)),
        "sampling_dates": unique_dates,
        "locations": location_count,
        "events": event_count,
        "current_date_overlap": overlap,
        "overlap_dates": int(len(overlap_dates)),
        "current_dates": int(len(current_dates)),
        "observation_start": obs["date"].min() if len(obs) else pd.NaT,
        "observation_end": obs["date"].max() if len(obs) else pd.NaT,
        "current_start": current_series.min() if len(current_series) else pd.NaT,
        "current_end": current_series.max() if len(current_series) else pd.NaT,
        "date_span_days": date_span,
        "reasons": reasons,
    }


def run_forward_field_validation(
    observations: pd.DataFrame,
    currents: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    event_threshold: float = 100_000.0,
    test_fraction: float = 0.30,
) -> dict[str, object]:
    quality = field_quality_gate(observations, currents, event_threshold=event_threshold, strict_forward=True)
    if quality["status"] != "ready":
        return {"quality": quality, "status": "defer"}
    dates = np.sort(observations["date"].dropna().unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1.0 - test_fraction))])
    train_rows = []
    all_pairs: dict[int, pd.DataFrame] = {}
    for lag in lags:
        pairs = build_flow_matched_pairs(observations, currents, int(lag), event_threshold)
        all_pairs[int(lag)] = pairs
        train = pairs[pairs["target_date"].lt(cut)] if not pairs.empty else pairs
        train_rows.append({"lag_days": int(lag), **summarize_lag_pairs(train)})
    train_summary = pd.DataFrame(train_rows)
    valid = train_summary.dropna(subset=["flow_ap"])
    if valid.empty:
        quality["reasons"] = [*quality["reasons"], "training block has no evaluable lag"]
        quality["status"] = "defer"
        return {"quality": quality, "status": "defer", "training_lags": train_summary}
    selected_lag = int(valid.sort_values(["flow_ap", "flow_vs_reverse"], ascending=False).iloc[0]["lag_days"])
    test_pairs = all_pairs[selected_lag]
    test_pairs = test_pairs[test_pairs["target_date"].ge(cut)].copy()
    test_summary = summarize_lag_pairs(test_pairs)
    if test_summary["samples"] < 10 or test_summary["events"] < 2:
        quality["reasons"] = [*quality["reasons"], "forward test block has insufficient paired events"]
        quality["status"] = "defer"
        return {
            "quality": quality, "status": "defer", "training_lags": train_summary,
            "selected_lag": selected_lag, "cut_date": cut,
        }
    return {
        "quality": quality,
        "status": "evaluated",
        "training_lags": train_summary,
        "selected_lag": selected_lag,
        "cut_date": cut,
        "test_pairs": test_pairs,
        "test_summary": test_summary,
        "boundary": (
            "Lag selection uses only the earlier block; the later block is evaluated once. "
            "The flow match remains a first-order current constraint, not operational particle tracking."
        ),
    }


def project_next_sampling_candidates(
    observations: pd.DataFrame,
    currents: pd.DataFrame,
    horizon_days: int = 7,
    top_n: int = 12,
) -> pd.DataFrame:
    if observations.empty or currents.empty:
        return pd.DataFrame()
    latest_date = pd.Timestamp(observations["date"].max())
    recent = observations[observations["date"].ge(latest_date - pd.Timedelta(days=2))].copy()
    recent = attach_nearest_current(recent, currents)
    recent = recent.dropna(subset=["u_ms", "v_ms"])
    if recent.empty:
        return pd.DataFrame()
    km_east = recent["u_ms"] * horizon_days * 86400.0 / 1000.0
    km_north = recent["v_ms"] * horizon_days * 86400.0 / 1000.0
    lat_rad = np.deg2rad(recent["latitude"].to_numpy(float))
    recent["projected_latitude"] = recent["latitude"] + km_north / 111.0
    recent["projected_longitude"] = recent["longitude"] + km_east / (111.0 * np.maximum(np.cos(lat_rad), 0.2))
    recent["current_speed_ms"] = np.sqrt(recent["u_ms"] ** 2 + recent["v_ms"] ** 2)
    recent["sampling_priority"] = np.log1p(recent["cell_count"]) * (1.0 + recent["current_speed_ms"])
    columns = [
        "date", "latitude", "longitude", "cell_count", "projected_latitude",
        "projected_longitude", "u_ms", "v_ms", "current_speed_ms", "sampling_priority",
    ]
    return recent.sort_values("sampling_priority", ascending=False)[columns].head(top_n).reset_index(drop=True)


def field_observation_template() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": "2026-08-01", "station_id": "UPSTREAM_A", "latitude": 27.50, "longitude": -84.00,
            "cell_count": 120000, "toxin_value": "", "toxin_unit": "", "water_temp_c": 30.1,
            "salinity": 35.2, "dissolved_oxygen_mg_l": 5.8, "nitrate_mmol_m3": 1.2,
            "phosphate_mmol_m3": 0.15, "silicate_mmol_m3": 2.8, "chlorophyll": 3.1,
        },
        {
            "date": "2026-08-02", "station_id": "MID_B", "latitude": 27.65, "longitude": -83.70,
            "cell_count": 25000, "toxin_value": "", "toxin_unit": "", "water_temp_c": 30.0,
            "salinity": 35.1, "dissolved_oxygen_mg_l": 5.6, "nitrate_mmol_m3": 1.1,
            "phosphate_mmol_m3": 0.14, "silicate_mmol_m3": 2.6, "chlorophyll": 2.7,
        },
    ])


def field_current_template() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2026-08-01", "latitude": 27.50, "longitude": -84.00, "u_ms": 0.18, "v_ms": 0.05},
        {"date": "2026-08-02", "latitude": 27.65, "longitude": -83.70, "u_ms": 0.16, "v_ms": 0.04},
    ])
