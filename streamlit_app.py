"""Alternative Streamlit entry point.

Use either ``app.py`` or ``streamlit_app.py``. Both execute the same application.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
