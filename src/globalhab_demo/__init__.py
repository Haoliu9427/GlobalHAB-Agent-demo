"""GlobalHAB-Agent competition demo package.

The initializer intentionally stays dependency-light. Application and CLI entry
points import feature modules directly, which keeps Streamlit hot redeploys from
reusing a stale aggregate import surface.
"""

__version__ = "4.1"

__all__ = ["__version__"]
