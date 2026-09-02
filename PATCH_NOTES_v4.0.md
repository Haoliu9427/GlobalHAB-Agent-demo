# GlobalHAB-Agent v4.0 change notes

## Competition-facing redesign

The six existing modules remain. v4.0 makes the project readable to non-specialist judges through four linked ideas: special data, science-structured model improvement, a major cross-region ecological question, and practical monitoring/aquaculture value.

## Dynamic-result rule

No existing current-run KPI was replaced with a fixed information card. New quality KPIs are calculated from the active synthetic run. New model-comparison KPIs appear only after the comparison is run for the active configuration; if the configuration changes, old model results are not shown as current results. Registered evidence is explicitly labelled and kept separate.

## New model comparison

The original Agent search remains unchanged. An independent same-holdout comparison evaluates Logistic, Random Forest, STS-Interaction GLM and STS-Gated TCN. All use the same held-out region and forward test rows. Training hyperparameters are chosen inside the training era, and stochastic models are checked across five seeds.

## Data fusion and quality

The evidence module now exposes the environment → transport → biological hazard → aquaculture vulnerability chain and current-run completeness/continuity/anomaly/event-support diagnostics.

## Communication layer

A capability comparison explains how the system differs from station-only prediction and generic spatiotemporal ML. A practical-use table frames monitoring, aquaculture, fisheries and future insurance/risk-management potential without claiming operational readiness.

## Model comparison panel refinement

The Exploration & Validation page now keeps the science-structured model comparison in one visible panel. Logistic, Random Forest and STS-Interaction GLM are evaluated dynamically on the current blocked holdout; STS-Gated TCN is added through the on-demand four-model, five-seed comparison. The registered lightweight-TCN audit remains a separate supplemental audit rather than a current-run result.
