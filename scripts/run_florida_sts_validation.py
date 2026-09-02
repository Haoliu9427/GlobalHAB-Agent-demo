"""Run Florida/Gulf flow-constrained STS retrospective validation.

Use --online for NOAA HABSOS plus a live Gulf current source. HYCOM GOMb0.04
reanalysis is the default for retrospective validation; NOAA CoastWatch remains
available as an alternative. CSV inputs are also supported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from globalhab_demo.florida_sts import (
    DEFAULT_LAGS,
    fetch_coastwatch_currents,
    fetch_hycom_gom_currents,
    fetch_habsos,
    normalize_current_frame,
    normalize_habsos,
    run_retrospective_sts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-08-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--threshold", type=float, default=100_000.0)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--current-source", choices=["hycom", "coastwatch"], default="hycom")
    parser.add_argument("--habsos-csv")
    parser.add_argument("--current-csv")
    args = parser.parse_args()

    if args.online:
        obs = fetch_habsos(args.start, args.end)
        if args.current_source == "hycom":
            current = fetch_hycom_gom_currents(args.start, args.end)
        else:
            current = fetch_coastwatch_currents(args.start, args.end)
    else:
        if not args.habsos_csv or not args.current_csv:
            parser.error("use --online or provide --habsos-csv and --current-csv")
        obs = normalize_habsos(pd.read_csv(args.habsos_csv))
        current = normalize_current_frame(pd.read_csv(args.current_csv))

    # Keep the Gulf/Florida domain used by the web workflow.
    obs = obs[obs["longitude"].between(-88.5, -80.0) & obs["latitude"].between(24.0, 31.2)].copy()
    result = run_retrospective_sts(obs, current, DEFAULT_LAGS, args.threshold)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    result["lag_summary"].to_csv(out / "florida_sts_lag_summary.csv", index=False)
    result["best_pairs"].to_csv(out / "florida_sts_best_pairs.csv", index=False)
    print(result["lag_summary"].to_string(index=False))
    print(f"\nBest lag: {result['best_lag']} days")
    print(result["boundary"])


if __name__ == "__main__":
    main()
