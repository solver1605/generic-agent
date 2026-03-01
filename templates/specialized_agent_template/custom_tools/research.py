"""Example custom tool factory for profile-based tool loading."""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def summarize_topic(topic: str) -> str:
    """Return a deterministic scaffold summary for a research topic."""
    t = (topic or "").strip()
    if not t:
        return "Topic is empty."
    return f"Summary scaffold for: {t}"


def build_tools():
    return [summarize_topic]
