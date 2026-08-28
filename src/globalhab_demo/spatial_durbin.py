"""Competition-equivalent spatial Durbin impact decomposition.

The implementation fits an inspectable linear-probability spatial panel over an
abstract row-standardised directed W matrix. It reports association-scale
direct, indirect and total impacts; it does not make a causal claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import REGIONS, UPSTREAM


EFFECT_NAMES = (
    "multiscale_anomaly_score_lag14", "nutrient_context", "circulation_residence_proxy"
)


def build_weight_matrix() -> tuple[np.ndarray, list[str]]:
    regions = list(REGIONS)
    index = {region: position for position, region in enumerate(regions)}
    weights = np.zeros((len(regions), len(regions)), dtype=float)
    for target, source in UPSTREAM.items():
        weights[index[target], index[source]] = 1.0
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
    return weights, regions


def _prepare_panel(
    frame: pd.DataFrame,
    anomaly_daily: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp], list[str]]:
    merged = frame.merge(
        anomaly_daily[["date", "region", "multiscale_anomaly_score"]],
        on=["date", "region"], how="left", validate="one_to_one",
    )
    merged = merged.sort_values(["region", "date"])
    merged["multiscale_anomaly_score_lag14"] = (
        merged.groupby("region")["multiscale_anomaly_score"].shift(14).fillna(0)
    )
    nutrient_raw = (
        0.50 * merged["nitrate_mmol_m3"]
        + 2.30 * merged["phosphate_mmol_m3"]
        + 0.16 * merged["silicate_mmol_m3"]
    )
    merged["nutrient_context"] = nutrient_raw
    for column in EFFECT_NAMES:
        scale = float(merged[column].std(ddof=0))
        merged[column] = (merged[column] - merged[column].mean()) / max(scale, 1e-8)

    dates = sorted(pd.Timestamp(value) for value in merged["date"].unique())
    regions = list(REGIONS)
    y = np.stack([
        merged[merged["date"].eq(date)].set_index("region").loc[regions, "hab_event"].to_numpy()
        for date in dates
    ])
    x = np.stack([
        merged[merged["date"].eq(date)].set_index("region").loc[regions, list(EFFECT_NAMES)].to_numpy()
        for date in dates
    ])
    day = np.array([date.dayofyear for date in dates])
    season = np.column_stack([
        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)
    ])
    weights, _ = build_weight_matrix()
    return y.astype(float), x.astype(float), season, weights, dates, regions


def _design(x: np.ndarray, season: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_dates, n_regions, _ = x.shape
    wx = np.einsum("ij,tjp->tip", weights, x)
    region_dummy = np.tile(np.eye(n_regions)[:, 1:], (n_dates, 1))
    season_rows = np.repeat(season, n_regions, axis=0)
    return np.column_stack([
        np.ones(n_dates * n_regions),
        x.reshape(-1, x.shape[2]),
        wx.reshape(-1, wx.shape[2]),
        region_dummy,
        season_rows,
    ])


def _fit_at_rho(
    y: np.ndarray,
    x: np.ndarray,
    season: np.ndarray,
    weights: np.ndarray,
    rho: float,
) -> dict[str, object]:
    wy = np.einsum("ij,tj->ti", weights, y)
    target = (y - rho * wy).reshape(-1)
    design = _design(x, season, weights)
    penalty = np.eye(design.shape[1]) * 1e-6
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    residual = target - design @ coefficients
    rss = float(np.dot(residual, residual))
    n = len(target)
    bic = float(n * np.log(max(rss / n, 1e-12)) + len(coefficients) * np.log(n))
    prediction = rho * wy.reshape(-1) + design @ coefficients
    total = y.reshape(-1)
    pseudo_r2 = float(1.0 - np.sum((total - prediction) ** 2) / np.sum((total - total.mean()) ** 2))
    return {
        "rho": float(rho), "coefficients": coefficients, "rss": rss,
        "bic": bic, "pseudo_r2": pseudo_r2, "design_condition_number": float(np.linalg.cond(design)),
    }


def _impacts(coefficients: np.ndarray, rho: float, weights: np.ndarray) -> np.ndarray:
    p = len(EFFECT_NAMES)
    beta = coefficients[1:1 + p]
    theta = coefficients[1 + p:1 + 2 * p]
    multiplier = np.linalg.inv(np.eye(len(weights)) - rho * weights)
    rows = []
    for variable_index in range(p):
        surface = multiplier @ (
            beta[variable_index] * np.eye(len(weights)) + theta[variable_index] * weights
        )
        direct = float(np.mean(np.diag(surface)))
        total = float(np.mean(surface.sum(axis=1)))
        indirect = total - direct
        rows.append([direct, indirect, total])
    return np.asarray(rows)


def estimate_spatial_durbin_impacts(
    frame: pd.DataFrame,
    anomaly_daily: pd.DataFrame,
    seed: int = 42,
    bootstrap_repeats: int = 39,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Fit rho by BIC grid search and decompose SDM multiplier impacts."""
    y, x, season, weights, dates, regions = _prepare_panel(frame, anomaly_daily)
    grid = np.linspace(-0.65, 0.65, 27)
    fits = [_fit_at_rho(y, x, season, weights, float(rho)) for rho in grid]
    best = min(fits, key=lambda item: float(item["bic"]))
    point = _impacts(best["coefficients"], float(best["rho"]), weights)

    rng = np.random.default_rng(seed + 912)
    bootstrap = []
    block_length = min(21, max(7, len(dates) // 20))
    for _ in range(bootstrap_repeats):
        selected: list[int] = []
        while len(selected) < len(dates):
            start = int(rng.integers(0, max(1, len(dates) - block_length + 1)))
            selected.extend(range(start, min(start + block_length, len(dates))))
        selected = selected[:len(dates)]
        fit = _fit_at_rho(
            y[selected], x[selected], season[selected], weights, float(best["rho"])
        )
        bootstrap.append(_impacts(fit["coefficients"], float(best["rho"]), weights))
    samples = np.stack(bootstrap)
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)

    rows = []
    for index, variable in enumerate(EFFECT_NAMES):
        for effect_index, effect_type in enumerate(("direct", "indirect", "total")):
            rows.append({
                "variable": variable,
                "effect_type": effect_type,
                "effect_per_1sd": float(point[index, effect_index]),
                "ci90_lower": float(lower[index, effect_index]),
                "ci90_upper": float(upper[index, effect_index]),
                "interpretation": "association-scale change in synthetic HAB probability per 1 SD",
            })
    effects = pd.DataFrame(rows)
    diagnostics = {
        "rho": float(best["rho"]),
        "bic": float(best["bic"]),
        "pseudo_r2": float(best["pseudo_r2"]),
        "design_condition_number": float(best["design_condition_number"]),
        "observations": int(y.size),
        "dates": int(len(dates)),
        "regions": regions,
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_block_days": int(block_length),
        "weight_matrix_definition": "row-standardised abstract directed upstream graph",
        "spatial_exposure_lag_days": 14,
        "lag_choice_basis": "Agent/TE-CTE synthetic signal recovery; fixed before SDM effect display",
        "outcome_model": "linear-probability SDM; exploratory association, not causal effect",
    }
    weights_frame = pd.DataFrame(weights, index=regions, columns=regions)
    weights_frame.index.name = "target_region"
    return effects, diagnostics, weights_frame.reset_index()
