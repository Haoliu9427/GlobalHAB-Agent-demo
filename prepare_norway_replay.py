"""Prepare the public Norwegian harmful-algae monitoring replay.

The source table is distributed with the probabilistic-model repository linked
to Silva et al. (2025), Communications Earth & Environment, under CC BY 4.0.
This adapter normalizes names and creates transparent event flags using the
paper's 200 cells L-1 operational definition for D. acuta and the
A. tamarense complex. It does not invent missing samples or negative labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ARTICLE = "https://doi.org/10.1038/s43247-025-02421-y"
ZENODO = "https://doi.org/10.5281/zenodo.10958487"
SOURCE_ARCHIVE_MD5 = "502fc865c6324cf7526230aaceab6979"
EVENT_THRESHOLD_CELLS_L = 200.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root: Path) -> dict[str, object]:
    raw_path = root / "data" / "real_case_norway" / "raw" / "norway_hab_observations.csv"
    derived_dir = root / "data" / "real_case_norway" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    output = raw.rename(columns={
        "Date": "sample_date",
        "Region": "region",
        "A. sp.": "alexandrium_sp_cells_l",
        "A. tamarense": "a_tamarense_cells_l",
        "D. acuta": "d_acuta_cells_l",
        "D. acuminata": "d_acuminata_cells_l",
        "D. norvegica": "d_norvegica_cells_l",
        "Pn. sp.": "pseudo_nitzschia_sp_cells_l",
        "P. reticulatum": "p_reticulatum_cells_l",
        "A. spinosum": "a_spinosum_cells_l",
        "sst": "sst_c",
        "par": "par_e_m2_d",
        "mld": "mixed_layer_depth_m",
        "sss": "sea_surface_salinity_psu",
    }).copy()
    output["a_tamarense_hab_event"] = output["a_tamarense_cells_l"].gt(
        EVENT_THRESHOLD_CELLS_L
    )
    output["d_acuta_hab_event"] = output["d_acuta_cells_l"].gt(
        EVENT_THRESHOLD_CELLS_L
    )
    output["target_hab_event"] = (
        output["a_tamarense_hab_event"] | output["d_acuta_hab_event"]
    )
    output["target_peak_cells_l"] = output[[
        "a_tamarense_cells_l", "d_acuta_cells_l"
    ]].max(axis=1)
    output["source_record"] = ZENODO
    output["source_license"] = "CC BY 4.0"
    output = output.sort_values(["sample_date", "region"], ignore_index=True)

    derived_path = derived_dir / "norway_hab_observations.csv"
    output.to_csv(derived_path, index=False)
    provenance = {
        "version": "1.0-observed-replay",
        "article": ARTICLE,
        "record": ZENODO,
        "license": "CC BY 4.0",
        "source_archive_md5": SOURCE_ARCHIVE_MD5,
        "raw_file": raw_path.name,
        "raw_sha256": sha256(raw_path),
        "derived_file": derived_path.name,
        "derived_sha256": sha256(derived_path),
        "observations": int(len(output)),
        "sampling_dates": int(output["sample_date"].nunique()),
        "regions": int(output["region"].nunique()),
        "date_range": [
            output["sample_date"].min().date().isoformat(),
            output["sample_date"].max().date().isoformat(),
        ],
        "event_definition": {
            "taxa": ["A. tamarense complex", "D. acuta"],
            "threshold_cells_l": EVENT_THRESHOLD_CELLS_L,
            "operator": ">",
            "basis": "Silva et al. (2025) study definition",
        },
        "boundary": (
            "Rows are monitoring observations. Event flags reproduce the paper's "
            "study definition and are not universal harvesting or health thresholds."
        ),
    }
    (derived_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(prepare(project_root), ensure_ascii=False, indent=2))
