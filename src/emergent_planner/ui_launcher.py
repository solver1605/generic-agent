"""Package entrypoint that launches the Streamlit UI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ui_script_path() -> str:
    # Streamlit expects a script path rather than an import path.
    return str((Path(__file__).resolve().parent / "ui.py").resolve())


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch GenericAgent Streamlit UI.",
        epilog="Pass additional Streamlit arguments after '--'. Example: generic-agent-ui -- --server.port 8502",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("GENERIC_AGENT_CONFIG", "agent_config.yaml"),
        help="Path to agent config YAML (default: agent_config.yaml).",
    )
    parser.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to `streamlit run` (prefix with '--').",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _make_parser().parse_args(argv)
    forward = list(args.streamlit_args or [])
    if forward and forward[0] == "--":
        forward = forward[1:]

    env = dict(os.environ)
    env["GENERIC_AGENT_CONFIG"] = str(args.config)

    cmd = [sys.executable, "-m", "streamlit", "run", _ui_script_path(), *forward]
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
