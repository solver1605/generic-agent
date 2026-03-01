"""Compatibility wrapper for local Streamlit execution.

Preferred entrypoint: `generic-agent-ui`.
"""
from __future__ import annotations

import sys
from pathlib import Path


try:
    from emergent_planner.ui import main
except ModuleNotFoundError:
    # Local checkout fallback when package is not installed.
    repo_src = Path(__file__).resolve().parent / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    from emergent_planner.ui import main


if __name__ == "__main__":
    main()
