# Version 4.0 — science-value-model-fusion

GlobalHAB-Agent v4.0 keeps the six-module scientific workflow and the existing dynamic KPI behavior, while making four competition-facing strengths explicit: special multi-source data, science-structured model improvement, the cross-region HAB propagation question, and practical monitoring/aquaculture value.

## What is preserved

- Risk-scenario and cage-fish controls remain instant dynamic calculations.
- Sidebar exploration settings still require an explicit recalculation and preserve the last valid run until the new run finishes.
- Dynamic anomaly, router, TE/CTE, Durbin, Logistic and Random-Forest results remain bound to the current active run.
- Registered audit evidence remains clearly separated from current-run results.
- The 24-candidate / limited-budget Agent search is unchanged.

## Data fusion and quality gate

The evidence page now exposes a four-layer fusion logic: environment shock, transport/residence background, biological hazard observations, and aquaculture vulnerability. Current-run quality indicators report completeness, daily continuity, MHW-day share, multiscale anomaly-day share and event rate. Data that fail continuity, class-balance or evidence-support requirements are degraded or deferred instead of being forced through every model.

## Model comparison and improvement

A new on-demand current-configuration comparison uses the same held-out region and forward test rows for all compared models and repeats the stochastic models across five seeds. Hyperparameters are selected only inside the outer training era.

- Logistic: transparent baseline candidate model.
- Random Forest: nonlinear classical benchmark.
- STS-Interaction GLM: explicitly represents transported MHW signal, nutrient context and their interaction.
- STS-Gated TCN: dual temporal branches for local/upstream histories with a transport-driven learnable gate and science-structured residual features.

The deeper model is not assumed to win. Negative results are displayed rather than hidden; the purpose is to test whether scientific structure or additional capacity contributes under the same leakage-safe split.

## Scientific and application narrative

The interface now states the central question in non-specialist language: can an environmental shock move across regions and appear days or weeks later as downstream ecological risk, and under limited monitoring resources where should the next sample be taken? A capability comparison and application table connect the research outputs to monitoring agencies, aquaculture operators, fisheries and future risk-management use without claiming operational deployment.

## Boundary

Synthetic results verify software and hypothesis-recovery behavior. South Australia is a real-event replay. Norway provides a separate retrospective forward benchmark. The biological-response sandbox is not calibrated mortality prediction, and no page provides automatic farm operation or regulatory instructions.
