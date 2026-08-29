"""Scientifically structured synthetic data for software verification only."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIONS = (
    "Synthetic_Region_A",
    "Synthetic_Region_B",
    "Synthetic_Region_C",
    "Synthetic_Region_D",
)

# This abstract directed graph is used only to test whether the agent can
# recover a displaced response. It does not describe any real ocean current.
UPSTREAM = {
    "Synthetic_Region_A": "Synthetic_Region_D",
    "Synthetic_Region_B": "Synthetic_Region_A",
    "Synthetic_Region_C": "Synthetic_Region_B",
    "Synthetic_Region_D": "Synthetic_Region_C",
}


def _ar1(
    rng: np.random.Generator,
    n: int,
    persistence: float = 0.82,
    scale: float = 1.0,
) -> np.ndarray:
    values = np.zeros(n)
    noise = rng.normal(scale=scale, size=n)
    for index in range(1, n):
        values[index] = persistence * values[index - 1] + noise[index]
    return values


def _heat_pulses(rng: np.random.Generator, n: int, count: int = 24) -> np.ndarray:
    """Create smooth warm anomalies that occasionally exceed a p90 threshold."""
    time = np.arange(n)
    pulses = np.zeros(n)
    centers = rng.choice(np.arange(25, n - 25), size=count, replace=False)
    for center in centers:
        duration = rng.uniform(3.0, 7.0)
        amplitude = rng.uniform(1.8, 3.4)
        pulses += amplitude * np.exp(-0.5 * ((time - center) / duration) ** 2)
    return pulses


def _bounded_proxy(microplastic_concentration: np.ndarray) -> np.ndarray:
    """Map MPs to a bounded circulation-residence/convergence proxy.

    MPs are not treated as a biological driver or a measurement of current
    velocity/direction. The monotonic transform is used only to weight the
    synthetic transmission pathway.
    """
    log_mp = np.log1p(microplastic_concentration)
    return 1.0 / (1.0 + np.exp(-(log_mp - 2.4)))


def generate_demo_data(days: int = 720, seed: int = 42) -> pd.DataFrame:
    """Generate a multi-region series with a hidden 14-day displaced response.

    MHW intensity is zero outside MHW days. On MHW days it equals SST minus the
    seasonal climatological mean, after SST exceeds the synthetic p90 threshold.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    day = np.arange(days)
    records: list[pd.DataFrame] = []
    mhw_by_region: dict[str, np.ndarray] = {}

    for index, region in enumerate(REGIONS):
        phase = index * 0.42
        base_temperature = 16.0 + index * 1.3
        climatological_mean = (
            base_temperature
            + 4.2 * np.sin(2 * np.pi * day / 365.25 + phase)
            + 0.35 * np.sin(4 * np.pi * day / 365.25)
        )
        climatological_p90 = climatological_mean + 1.35
        weather = 0.24 * _ar1(rng, days, persistence=0.88, scale=0.55)
        sst = climatological_mean + weather + _heat_pulses(rng, days)
        is_mhw = sst > climatological_p90
        mhw_intensity = np.where(is_mhw, sst - climatological_mean, 0.0)

        nitrate = np.clip(
            3.0
            + 0.65 * _ar1(rng, days, persistence=0.76, scale=0.45)
            + 0.75 * np.cos(2 * np.pi * day / 365.25 + phase),
            0.05,
            None,
        )
        phosphate = np.clip(
            0.42
            + 0.09 * _ar1(rng, days, persistence=0.72, scale=0.45)
            + 0.08 * np.cos(2 * np.pi * day / 365.25 + phase + 0.35),
            0.01,
            None,
        )
        silicate = np.clip(
            4.8
            + 0.85 * _ar1(rng, days, persistence=0.78, scale=0.55)
            + 1.15 * np.cos(2 * np.pi * day / 365.25 + phase - 0.25),
            0.05,
            None,
        )

        circulation_state = 0.8 * _ar1(rng, days, persistence=0.91, scale=0.30)
        circulation_state += 0.35 * np.sin(2 * np.pi * day / 180 + phase)
        microplastic_concentration = np.exp(
            2.35 + 0.72 * circulation_state + rng.normal(scale=0.22, size=days)
        )
        circulation_proxy = _bounded_proxy(microplastic_concentration)

        mhw_by_region[region] = mhw_intensity
        records.append(pd.DataFrame({
            "date": dates,
            "region": region,
            "sst_c": sst,
            "climatological_mean_sst_c": climatological_mean,
            "climatological_p90_sst_c": climatological_p90,
            "is_mhw": is_mhw.astype(int),
            "mhw_intensity_c": mhw_intensity,
            "nitrate_mmol_m3": nitrate,
            "phosphate_mmol_m3": phosphate,
            "silicate_mmol_m3": silicate,
            "microplastic_concentration_items_m3": microplastic_concentration,
            "circulation_residence_proxy": circulation_proxy,
        }))

    frame = pd.concat(records, ignore_index=True)
    events = np.zeros(len(frame), dtype=int)
    probabilities = np.zeros(len(frame), dtype=float)
    hidden_lag = 14
    label_rng = np.random.default_rng(seed + 1984)

    for region in REGIONS:
        mask = frame["region"].eq(region).to_numpy()
        rows = frame.loc[mask]
        upstream_mhw = mhw_by_region[UPSTREAM[region]]
        transmitted_mhw = np.r_[np.zeros(hidden_lag), upstream_mhw[:-hidden_lag]]
        route_gate = rows["circulation_residence_proxy"].to_numpy()
        transmitted_signal = transmitted_mhw * (0.55 + 0.90 * route_gate)

        nitrate_z = (rows["nitrate_mmol_m3"].to_numpy() - 3.0) / 0.9
        phosphate_z = (rows["phosphate_mmol_m3"].to_numpy() - 0.42) / 0.13
        silicate_z = (rows["silicate_mmol_m3"].to_numpy() - 4.8) / 1.35
        nutrient_context = 0.50 * nitrate_z + 0.30 * phosphate_z + 0.20 * silicate_z

        # The label depends on the displaced MHW signal and nutrient context.
        # MPs affect only the transmission gate, not the biological term.
        logit = (
            -3.80
            + 1.40 * transmitted_signal
            + 0.50 * nutrient_context
            + 0.15 * transmitted_signal * nutrient_context
            + label_rng.normal(scale=0.35, size=days)
        )
        probability = 1.0 / (1.0 + np.exp(-logit))
        probabilities[mask] = probability
        events[mask] = label_rng.binomial(1, probability)

    frame["hab_event"] = events
    frame["hidden_probability"] = probabilities
    return frame.sort_values(["date", "region"], ignore_index=True)
