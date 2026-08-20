"""Console launcher for the Streamlit RS-Agent application."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise RuntimeError(
            'Streamlit is not installed; run pip install -e ".[web]"'
        ) from exc
    app_path = Path(__file__).resolve().parents[1] / "web" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
