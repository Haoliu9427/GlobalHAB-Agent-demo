# Third-party data attribution

## South Australia Karenia qPCR observations

- Citation: Murray, S. A., Bolch, C. J. S., Brett, S. et al. (2026), *A catastrophic marine mortality event caused by a complex algal bloom including the brevetoxin producer Karenia cristata*, Nature Ecology & Evolution. https://doi.org/10.1038/s41559-026-03115-0
- Data record: https://doi.org/10.5281/zenodo.20227730
- Record license: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Bundled source files:
  - `data/real_case/raw/Figure2_Final_qPCR_data_integrated.xlsx`
  - `data/real_case/raw/DataDescription_README.txt`
- Derived file: `data/real_case/derived/sa_qpcr_observations.csv`

Transformations are implemented in `scripts/prepare_sa_real_replay.py`: column names are normalized, southern latitudes are converted to negative decimal degrees, reported numeric abundances are parsed, and each `Not detected` value is represented by both a numeric zero and a separate boolean status. No synthetic rows or supervised negative labels are added.

The original workbook is preserved unchanged. Source and derived SHA-256 values are recorded in `data/real_case/derived/provenance.json`.

## Norwegian toxic-algae monitoring observations

- Citation: Silva, E., Counillon, F., Brajard, J. et al. (2025), *Warming and freshening coastal waters impact harmful algal bloom frequency in high latitudes*, Communications Earth & Environment. https://doi.org/10.1038/s43247-025-02421-y
- Data and model record: https://doi.org/10.5281/zenodo.10958487
- Record license: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Bundled source table: `data/real_case_norway/raw/norway_hab_observations.csv`
- Derived table: `data/real_case_norway/derived/norway_hab_observations.csv`

`scripts/prepare_norway_replay.py` normalizes the original field names and reproduces the study definition of an event observation (`A. tamarense` complex or `D. acuta` above 200 cells L⁻¹). It does not interpolate missing monitoring effort, create synthetic observations, or treat the study definition as a universal harvesting threshold. Source archive MD5 and source/derived SHA-256 values are retained in `data/real_case_norway/derived/provenance.json`.

## NOAA OISST adapter

`scripts/prepare_sa_real_replay.py` contains an optional adapter for the NOAA/NCEI OISST v2.1 daily 0.25-degree product. The default offline package does not bundle the multi-decadal NOAA archive. When run with network access, the adapter requests the required subset and calculates a 1991–2020 calendar-day mean and p90 threshold using a ±5-day pooled window.

Historical subsets are requested from the NOAA CoastWatch ERDDAP dataset `ncdcOisst21Agg` (1981–present): https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg . The 1991–2020 baseline is split into bounded time chunks to reduce timeout risk; every generated request URL is retained in the provenance record.

The repository MIT license applies to project code only. It does not replace the licenses or attribution requirements of third-party data.

## Florida/Gulf public real-data adapters

### NOAA HABSOS

The Florida/Gulf retrospective workflow can request public Harmful Algal BloomS Observing System (HABSOS) records from the NOAA/NCEI ArcGIS REST service. The adapter requests observation coordinates, sample dates, taxonomic fields, cell count and available environmental fields, then normalizes them at runtime. These live records are not relicensed under the repository MIT license and are not bundled as a fabricated fixed performance result.

- NOAA/NCEI product page: https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system
- ArcGIS service used by the adapter: https://gis.ncdc.noaa.gov/arcgis/rest/services/ms/HABSOS_CellCounts/MapServer/0

### NOAA CoastWatch sea-surface geostrophic currents

The built-in live current adapter targets NOAA CoastWatch ERDDAP dataset `noaacwBLENDEDNRTcurrentsDaily`, which exposes daily gridded eastward and northward surface geostrophic current components. Source data remain subject to the NOAA dataset's own attribution/disclaimer terms.

- Dataset information: https://coastwatch.noaa.gov/erddap/info/noaacwBLENDEDNRTcurrentsDaily/index.html

### Alternative uploaded current sources

The workflow accepts normalized CSV exports from sources such as HYCOM Gulf of Mexico reanalysis, Copernicus Marine products and IOOS/GCOOS HF-radar. The repository does not redistribute those full archives. Users remain responsible for source-specific licenses, attribution and access conditions.

## Future field data

Files uploaded through the field-forward interface remain user-supplied research data. The repository template files contain only illustrative schema rows and do not grant rights over future cruise, station, farm, toxin or biological-response observations.
