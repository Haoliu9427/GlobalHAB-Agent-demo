@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow REAL_DATA_GUIDE.md to create .venv and install requirements.txt.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
