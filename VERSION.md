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


## Field visual screening workspace

- Added a fourth top-level workspace: `现场影像甄别 / Field Visual Screening`.
- Supports Streamlit camera capture or JPG/PNG upload, local image-quality checks, transparent colour/texture screening, and field metadata.
- The default backend is explicitly a heuristic visual baseline, not a trained HAB species/toxin classifier.
- Outputs a follow-up screening priority (`low / moderate / high / defer`) rather than an HAB probability.
- Stores the structured result in-session and can hand it to the LLM interpretation workspace; raw photos are not sent to the remote LLM by default.
- Added `VisualScreeningBackend` as a replaceable interface for a future calibrated project-specific vision model.


## Adaptive visual routing

- Upgraded the field visual workspace with quality-first adaptive routing.
- Added optional DINOv2, ConvNeXt-Tiny and EfficientNet-B0 frozen feature extractors.
- Added safe NumPy linear heads that fuse image embeddings with 12 field-metadata features.
- Added entropy, top-1/top-2 margin, multi-branch disagreement and OOD-aware DEFER.
- Deep branches activate only when both encoder assets and project-trained heads are available; otherwise the workspace falls back to the transparent heuristic baseline.
- Added `requirements-vision.txt`, training manifest template, lightweight-head training script and `VISION_ROUTER_GUIDE.md`.
