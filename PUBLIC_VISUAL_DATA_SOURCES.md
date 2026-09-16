# Public visual data sources for optional domain adaptation

The release keeps third-party raw images outside the normal Git repository. It stores source metadata, adapters/lightweight heads, model cards and hashes instead.

## Algal Blooms Sweden - 2023

- Record: https://zenodo.org/records/10599927
- DOI: 10.5281/zenodo.10599927
- Published: 2024-01-31
- Record description: 60 Baltic Sea surface bloom observations with time/location metadata.
- Evidence boundary: the record states the blooms were verified by Swedish information centres, but the photographed blooms were not taxonomically annotated by microscopy or genetic analysis.
- Intended GlobalHAB-Agent role: optional positive-domain embedding adapter / representation calibration only. It is not used as a species or toxin label set.

Fetch locally with:

```bash
python scripts/fetch_public_visual_data.py
```

Then build aggregate embedding adapters with:

```bash
python scripts/build_public_visual_adapter.py --backbone efficientnet
python scripts/build_public_visual_adapter.py --backbone convnext
python scripts/build_public_visual_adapter.py --backbone dinov2
```

Review the upstream record and licence/terms before redistribution or production use.
