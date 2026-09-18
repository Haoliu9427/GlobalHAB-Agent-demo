# GlobalHAB-Agent Android packaging starter

This folder documents the recommended APK route without changing the Python scientific engine.

## Route A — demonstration APK (fastest)

Create a minimal Android WebView shell whose start URL is the deployed GlobalHAB-Agent Streamlit app. Android officially supports embedding first-party web content with `WebView`. This is suitable for a competition/demo installation because it reuses the existing application and keeps one codebase.

Important: this APK is a **client shell**, not an offline copy of the Python app. The Streamlit server must remain reachable.

Suggested start URL for the current project deployment:

`https://globalhab-agent-demo.streamlit.app/`

Required Android permission: `android.permission.INTERNET`.

Recommended WebView settings: JavaScript enabled, DOM storage enabled, mixed content disabled, production web debugging disabled, external origins opened with a browser/custom tab unless explicitly allow-listed.

## Route B — production app (recommended long term)

1. Extract scientific operations from Streamlit callbacks into an ASGI/FastAPI service.
2. Keep `globalhab_demo` as the shared scientific package.
3. Build Android UI with Kotlin/Jetpack Compose or Flutter.
4. Use AMap Android SDK for the native map on mainland-China deployments.
5. Keep WGS84 in the scientific database; transform only display coordinates for the selected map provider.
6. Camera capture, cached Cases and notifications live on the phone; numerical analysis and model/version audit stay on the server.

## Why an APK is not included here

A signed APK requires an Android SDK/build-tools installation, a package id, signing configuration and (for AMap) an application key bound to the package/SHA1. Those deployment credentials are intentionally not embedded in the scientific source package.
