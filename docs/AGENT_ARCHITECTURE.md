# GlobalHAB-Agent architecture

## What makes the project an Agent

GlobalHAB-Agent is not defined by the presence of an LLM. The agentic loop is the auditable scientific workflow that selects an admissible method/action, runs an experiment, receives quantitative feedback, updates the next action, and preserves the evidence trail under a fixed budget.

The existing `HypothesisAgent` operates on a `route × lag × model` action catalogue. The larger product-level Agent adds method routing, ST/STS scientific reasoning, real-event validation, Case management, field evidence, biological-response scenarios, and an interpretation layer.

## Data flow

1. **Input** — environmental time series, SST/MHW, nutrients, transport/current information, HAB or qPCR observations, latitude/longitude/time, field images, aquaculture context, user CSV, registered CSV/JSON model outputs.
2. **Perception and quality control** — multiscale anomaly detection, sample/data support checks, field-image quality assessment, temporal and spatial availability diagnostics.
3. **Adaptive method routing** — `router.py` decides which scientific branches are admissible for the available evidence.
4. **ST / STS scientific reasoning** — ST covers Shock identification and Transmission direction/lag; STS adds Spillover decomposition. Current implementations include multiscale anomalies, blocked prediction, TE/CTE and Spatial Durbin effects.
5. **Hypothesis Agent** — selects the next `route × lag × model` experiment within a budget and receives AP/calibration/control feedback.
6. **Evidence grounding** — South Australia replay, Norway forward validation, user-data analysis and the Case queue connect candidates to independent evidence.
7. **Field and biological evidence** — field-image screening, laboratory evidence and the biological-response sandbox add operational context without overwriting upstream results.
8. **Interpretation** — the LLM workspace reads registered summaries and Cases. It does not recompute or alter numerical evidence.
9. **Feedback** — confirmed field evidence and user-approved images can enter the Case ledger/training library for later model updates.

## Compatibility boundary

The system is intentionally branchable. A source does not need every variable. Missing current/transport information disables or limits transmission interpretation rather than fabricating it. Non-continuous event observations are treated as replay/evidence rather than converted into an unjustified supervised forecast. The current user workbench is CSV-first; Florida/HYCOM adapters also parse small NetCDF responses internally.

## Outputs

Outputs include anomaly/event tables, routing diagnostics, TE/CTE networks, spatial direct/indirect effects, candidate route/lag/model records, Agent logs, blocked validation metrics, real-event replay/forward-validation summaries, Case/evidence status, field-screening records, biological-response scenario comparisons, and structured interpretation text.
