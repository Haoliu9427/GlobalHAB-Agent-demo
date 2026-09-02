# GlobalHAB-Agent v4.1 change notes

## 1. Equal-budget Bayesian experiment design

The original constrained Agent remains available and unchanged as a transparent reference. A new policy benchmark compares it with Bayesian Expected Improvement, Bayesian Information Gain, Thompson Sampling and Random selection on the same 24-action landscape and the same experiment budget. Acquisition rules cannot see the hidden 14-day truth. The truth is used only after each trajectory for recovery scoring.

## 2. Florida/Gulf public-data STS retrospective validation

A new real-evidence case connects NOAA HABSOS Karenia brevis observations with an observed current field. The built-in live current adapter uses NOAA CoastWatch daily surface geostrophic u/v; uploaded HYCOM, Copernicus or HF-radar CSVs are also supported. Candidate lags are compared against spatial-only and reversed-flow controls.

No Florida performance score is hard-coded or bundled as if it were a completed real validation. Results are computed only after public data are fetched or user data are uploaded.

## 3. Future field forward-validation interface

Two CSV templates define the minimum cruise/station contract. The interface runs a quality gate, chooses lag only in the earlier block, evaluates the later block once, and returns a defer state when evidence is insufficient. Optional toxin, DO, nutrients, chlorophyll and aquaculture-response fields are retained for later expansion.

## 4. Reproducibility

Added:
- `src/globalhab_demo/bayesian_design.py`
- `src/globalhab_demo/florida_sts.py`
- `scripts/run_agent_policy_benchmark.py`
- `scripts/run_florida_sts_validation.py`
- `scripts/run_field_forward_validation.py`
- `data/field_validation/field_observations_template.csv`
- `data/field_validation/field_currents_template.csv`
- focused policy and field-validation tests.

## 5. Fast release smoke and submission compliance

The GitHub `release-smoke-test` now runs a focused ~tens-of-seconds release suite rather than the full heavy audit. Submission-facing Agent-policy notes, minimum reproduction commands and requirement mapping are included under `prompts/`, `docs/MINIMAL_REPRODUCTION.md` and `SUBMISSION_MANIFEST.md`. The full offline test suite remains available separately.
