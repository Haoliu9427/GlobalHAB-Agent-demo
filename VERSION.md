# Version 4.1

## Experiment selection

- Added Bayesian Expected Improvement, Bayesian Information Gain and Thompson Sampling alongside the constrained heuristic and Random policy.
- All policies use the same 24 candidate experiments and the same budget.
- The synthetic 14-day reference is excluded from acquisition functions and used only after a trajectory is complete.

## Florida/Gulf retrospective analysis

- Added NOAA HABSOS `Karenia brevis` input.
- Added NOAA CoastWatch surface-current input and CSV adapters for HYCOM, Copernicus Marine and HF-radar products.
- Added flow-constrained, no-flow and reverse-flow lag comparisons.

## Field forward validation

- Added station-observation and current-field CSV templates.
- Added data-quality checks, training-period lag selection, later-period forward evaluation and next-sampling projections.
- Insufficient data return `DEFER`.

## Reproduction

```bash
python scripts/run_agent_policy_benchmark.py
python scripts/run_florida_sts_validation.py --online
python scripts/run_field_forward_validation.py --observations <csv> --currents <csv>
```
