"""Discrete TE/CTE network with permutation and direction controls.

The estimator is intentionally compact and inspectable. It measures conditional
mutual information in bits and uses circular-shift permutations to preserve
source-series autocorrelation while breaking the proposed lag alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import REGIONS, UPSTREAM
from .experiment import REVERSED_UPSTREAM


def _conditional_mutual_information(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    table = pd.DataFrame({"x": x.astype(int), "y": y.astype(int), "z": z.astype(int)})
    n = len(table)
    if n == 0:
        return 0.0
    xyz = table.value_counts(["x", "y", "z"]).rename("n_xyz").reset_index()
    xz = table.value_counts(["x", "z"]).rename("n_xz")
    yz = table.value_counts(["y", "z"]).rename("n_yz")
    zc = table.value_counts("z").rename("n_z")
    value = 0.0
    for row in xyz.itertuples(index=False):
        n_xyz = float(row.n_xyz)
        n_xz = float(xz.loc[(row.x, row.z)])
        n_yz = float(yz.loc[(row.y, row.z)])
        n_z = float(zc.loc[row.z])
        value += (n_xyz / n) * np.log2((n_xyz * n_z) / (n_xz * n_yz))
    return float(max(0.0, value))


def _source_states(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0]
    if len(positive) < 3:
        return (values > 0).astype(int)
    threshold = float(np.median(positive))
    return np.where(values <= 0, 0, np.where(values <= threshold, 1, 2)).astype(int)


def _nutrient_states(group: pd.DataFrame) -> np.ndarray:
    nutrient = (
        0.50 * group["nitrate_mmol_m3"].rank(pct=True)
        + 0.30 * group["phosphate_mmol_m3"].rank(pct=True)
        + 0.20 * group["silicate_mmol_m3"].rank(pct=True)
    )
    return np.clip((nutrient.to_numpy() * 3).astype(int), 0, 2)


def _edge_information(
    source_values: np.ndarray,
    target: pd.DataFrame,
    lag_days: int,
) -> tuple[float, float]:
    x = _source_states(source_values[:-lag_days])
    y = target["hab_event"].to_numpy()[lag_days:].astype(int)
    y_previous = target["hab_event"].shift(1).fillna(0).to_numpy()[lag_days:].astype(int)
    nutrients = _nutrient_states(target)[lag_days:]
    te = _conditional_mutual_information(x, y, y_previous)
    condition = y_previous * 3 + nutrients
    cte = _conditional_mutual_information(x, y, condition)
    return te, cte


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(n, dtype=float)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def estimate_te_cte_network(
    frame: pd.DataFrame,
    lags: tuple[int, ...] = (3, 7, 14, 21, 30, 45),
    permutations: int = 79,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate directed TE and nutrient-conditioned TE for every abstract edge."""
    if permutations < 19:
        raise ValueError("at least 19 permutations are required")
    region_frames = {
        region: group.sort_values("date").reset_index(drop=True)
        for region, group in frame.groupby("region")
    }
    rng = np.random.default_rng(seed + 430)
    rows: list[dict[str, object]] = []
    for target_region in REGIONS:
        target = region_frames[target_region]
        forward_source_region = UPSTREAM[target_region]
        reverse_source_region = REVERSED_UPSTREAM[target_region]
        forward_values = region_frames[forward_source_region]["mhw_intensity_c"].to_numpy()
        reverse_values = region_frames[reverse_source_region]["mhw_intensity_c"].to_numpy()
        for lag in lags:
            te, cte = _edge_information(forward_values, target, lag)
            reverse_te, reverse_cte = _edge_information(reverse_values, target, lag)
            permuted = []
            minimum_shift = max(lags) + 5
            maximum_shift = len(forward_values) - minimum_shift
            for _ in range(permutations):
                shift = int(rng.integers(minimum_shift, maximum_shift))
                shifted = np.roll(forward_values, shift)
                _, value = _edge_information(shifted, target, lag)
                permuted.append(value)
            p_value = (1.0 + sum(value >= cte for value in permuted)) / (permutations + 1.0)
            rows.append({
                "source_region": forward_source_region,
                "target_region": target_region,
                "lag_days": int(lag),
                "te_bits": te,
                "cte_bits": cte,
                "reverse_source_region": reverse_source_region,
                "reverse_te_bits": reverse_te,
                "reverse_cte_bits": reverse_cte,
                "net_directionality_bits": cte - reverse_cte,
                "permutation_p": float(p_value),
                "permutations": int(permutations),
                "conditioning": "target t-1 state + target nutrient tertile",
            })
    result = pd.DataFrame(rows)
    result["fdr_q"] = _bh_fdr(result["permutation_p"].to_numpy())
    result["significant_fdr_0_10"] = result["fdr_q"].le(0.10)
    return result.sort_values(
        ["cte_bits", "net_directionality_bits"], ascending=False, ignore_index=True
    )


def summarise_te_cte_by_lag(network: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the edge-level network for concise UI and discovery cards."""
    summary = network.groupby("lag_days", as_index=False).agg(
        mean_te_bits=("te_bits", "mean"),
        mean_cte_bits=("cte_bits", "mean"),
        mean_reverse_cte_bits=("reverse_cte_bits", "mean"),
        mean_net_directionality_bits=("net_directionality_bits", "mean"),
        minimum_p=("permutation_p", "min"),
        significant_edges=("significant_fdr_0_10", "sum"),
    )
    return summary.sort_values("lag_days", ignore_index=True)

