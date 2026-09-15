# Version 4.1

## Experiment selection

- Added Bayesian Expected Improvement, Bayesian Information Gain and Thompson Sampling alongside the constrained heuristic and Random policy.
- All policies use the same 24 candidate experiments and the same budget.
- The synthetic 14-day reference is excluded from acquisition functions and used only after a trajectory is complete.

## Florida/Gulf retrospective analysis

- Added NOAA HABSOS `Karenia brevis` input.
- Added live HYCOM-TSIS GOMb0.04 Gulf reanalysis as the default retrospective current source; NOAA CoastWatch remains an online alternative, with CSV adapters for HYCOM/Copernicus/HF-radar products.
- Added flow-constrained, no-flow and reverse-flow lag comparisons.

## Field forward validation

- Added station-observation and current-field CSV templates.
- Added data-quality checks, training-period lag selection, later-period forward evaluation and next-sampling projections.
- Insufficient data return `DEFER`.


## LLM result interpretation workspace

- Added a third top-level workspace: `大模型结果解读`.
- Reads registered project results, real-training results, mainland validation, the latest in-session user analysis, or uploaded result files.
- Supports research, defense, paper-results and management-summary modes.
- Uses the existing user/server Chat Completions compatible configuration; credentials remain session-only.
- The LLM receives a bounded result summary only after explicit consent and cannot modify upstream metrics or predictions.

## Reproduction

```bash
python scripts/run_agent_policy_benchmark.py
python scripts/run_florida_sts_validation.py --online
python scripts/run_field_forward_validation.py --observations <csv> --currents <csv>
```
