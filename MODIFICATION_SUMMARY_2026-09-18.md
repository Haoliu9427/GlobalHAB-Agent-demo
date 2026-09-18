# GlobalHAB-Agent enhancement summary — 2026-09-18

This package is derived from the user-supplied final Streamlit deployment package. The scientific calculations and registered validation outputs were not rewritten.

## 1. Interface refinement

- Reduced decorative gradients, shadows, hover lift effects, oversized rounding and repeated all-caps English labels.
- Kept the existing navy/teal scientific visual identity while making cards and navigation flatter and more data-first.
- Reworded workspace hero copy to describe the actual task rather than use product-marketing language.

## 2. New Homepage / project overview

- Added `项目总览` as the first workspace.
- The Homepage reads existing registered outputs and session state rather than recomputing results.
- It summarizes the current method candidate, Norway forward-validation result, Case queue/evidence count, field-image library, user-data status and LLM interpretation status.
- Four linked cards jump directly to Research & Validation, Own Data Analysis, Field Image Screening and LLM Result Interpretation.

## 3. Agent architecture and method framework

- Added an in-app architecture diagram showing input → perception/QC → Adaptive Router → ST/STS → Hypothesis Agent → evidence/output → feedback.
- Defined ST as Shock + Transmission and STS as ST + Spillover, mapped to the existing multiscale anomaly, TE/CTE, blocked prediction and Spatial Durbin implementations.
- Added reusable PPT/document figures:
  - `assets/GlobalHAB-Agent_method_framework.png`
  - `assets/GlobalHAB-Agent_method_framework.svg`
- Added `docs/AGENT_ARCHITECTURE.md` with input, output, compatibility and data-flow definitions.

## 4. Map / mainland-network compatibility

- Preserved OpenStreetMap as an option.
- Added `内置矢量底图（无需外部瓦片）`, which uses Plotly geographic vectors and therefore does not request OSM tiles.
- The new map backend switch only changes presentation. Stored scientific coordinates and numerical results are not changed.
- Added `docs/MOBILE_ANDROID_AND_MAPS.md` for AMap/Android planning.

## 5. Android / APK path

- Added `mobile_android/README.md` describing two routes:
  - fast demo APK: Android WebView/Capacitor shell loading the deployed Streamlit app;
  - production app: mobile client + Python API backend, with AMap Android SDK on mainland-China deployments.
- A signed APK is not included because this environment does not contain the Android SDK/build tools and no application signing credentials or AMap key were supplied.

## Validation performed here

- Python byte-code compilation passed for modified Python modules.
- Both map rendering branches were instantiated with a small coordinate frame: `scattergeo` for the no-tile backend and `scattermap/open-street-map` for OSM.
- The method-framework SVG was rendered to PNG and visually inspected.
- The repository's Streamlit UI test could not be executed in this runtime because Streamlit is not installed and outbound package installation is unavailable. The original package already includes its own validation scripts for execution in the deployment environment.
