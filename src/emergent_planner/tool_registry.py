"""
Helpers for introspecting and selecting tools by name.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def tool_name(tool_obj: Any) -> str:
    if hasattr(tool_obj, "name"):
        return str(getattr(tool_obj, "name"))
    if hasattr(tool_obj, "__name__"):
        return str(getattr(tool_obj, "__name__"))
    return str(tool_obj)


def tool_catalog(tools: Iterable[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in tools:
        out.append(
            {
                "name": tool_name(t),
                "description": (getattr(t, "description", "") or "").strip(),
            }
        )
    return out


def select_tools(tools: Iterable[Any], enabled_names: Iterable[str]) -> List[Any]:
    enabled = set(enabled_names)
    selected: List[Any] = []
    for t in tools:
        if tool_name(t) in enabled:
            selected.append(t)
    return selected
