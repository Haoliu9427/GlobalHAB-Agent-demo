# Version 4.1 — Bayesian exploration + real-flow STS validation

GlobalHAB-Agent keeps the six-module workflow and all current-run dynamic KPI behavior. The exploration environment now includes an equal-budget experiment-design audit, and the real-evidence module implements a two-stage path from public Florida/Gulf retrospective validation to future cruise/farm forward validation.

## Agent experiment design

The same 24 route × lag × model actions and the same experiment budget can now be explored with five policies:

- the existing transparent constrained heuristic;
- Bayesian Expected Improvement;
- Bayesian Information Gain;
- Thompson Sampling;
- equal-budget Random selection.

Bayesian policies observe only the utilities of experiments already selected. The pre-registered synthetic 14-day truth is not provided to acquisition functions and is used only after a trajectory ends to score recovery efficiency. This is an experiment-selection audit, not a claim that Bayesian policies are superior on real oceans.

## Two-stage real STS validation

### Stage 1 — Florida/Gulf retrospective validation

The web interface can read NOAA HABSOS Karenia brevis observations and a NOAA CoastWatch daily surface-geostrophic current field, or accept current CSVs exported from HYCOM, Copernicus Marine or HF-radar products. Candidate lags are evaluated by asking whether a prior high-abundance observation is more consistent with the observed current direction than with spatial-only or reversed-flow controls.

This is a flow-constrained retrospective association. The live CoastWatch adapter uses a first-order current displacement and is not a full Lagrangian particle-tracking forecast.

### Stage 2 — field forward validation interface

Future cruise, monitoring-station or aquaculture-partner data can be uploaded through a fixed schema. The system checks observation count, sampling dates, spatial support, events and current-field temporal coverage. When the quality gate passes, an earlier block selects the lag and a later block is evaluated once. The same interface then produces flow-vs-control metrics and first-order next-sampling candidates. Insufficient data are deferred rather than forced through the model.

## Reproduction

- `python scripts/run_agent_policy_benchmark.py`
- `python scripts/run_florida_sts_validation.py --online`
- `python scripts/run_field_forward_validation.py --observations <csv> --currents <csv>`

Field templates are under `data/field_validation/`.
