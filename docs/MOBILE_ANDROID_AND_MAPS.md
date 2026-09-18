# Android / APK and mainland-China map plan

## Recommended split

Do not replace the stable Streamlit competition build with an Android-only codebase. Keep the Python scientific engine as the source of truth and add a mobile client layer.

### Fast demonstration APK

Use an Android WebView shell (or Capacitor) that loads the deployed, mobile-responsive Streamlit URL. This reuses the current application and can produce an installable APK, but it still requires network access to the Streamlit server and does not make Python calculations run locally on the phone.

### Production mobile application

Split the Python computation into a service API (FastAPI/ASGI is a natural fit) and build the mobile UI in Flutter, Kotlin/Jetpack Compose, or a web-native shell. The mobile client owns camera, local caching, push notifications and the native map. The server owns scientific computation, model versions, Case records and auditable outputs.

## Map provider

For mainland-China deployment, use a provider selected by deployment region rather than forcing one basemap everywhere:

- **AMap/Gaode Android SDK** for the native Android map in mainland China.
- **AMap JavaScript API 2.0** for a web/Streamlit China-specific map component when a web map is required.
- Keep a non-tile Plotly geographic fallback for global scientific result panels so the core evidence view does not depend on one external tile service.

AMap requires an application key. Android keys are bound to the package name and signing SHA1. New JS API keys also use a security key; production deployments should protect that security key behind a proxy rather than expose it in browser code.

## Coordinate rule

Scientific source data should remain stored in WGS84. When drawing mainland-China overlays on AMap, convert only the display coordinates to the coordinate convention required by the provider. Never overwrite the scientific source coordinates in stored results.

## Mobile-specific UX

The new Homepage should become the phone landing screen. On small screens, show the four workspaces as stacked cards, keep the map full width, move dense research tables into drill-down pages, and make field-image capture / Case review the primary mobile workflow.
