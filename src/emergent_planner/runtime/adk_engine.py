"""Google ADK runtime adapter.

Current implementation preserves runtime contracts by normalizing through the
existing LangGraph-compatible execution path after validating ADK dependency
availability. This enables phased migration without changing UI/CLI contracts.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional

from ..config import ADKConfig
from .engine import AgentRuntimeEngine


class ADKRuntimeEngine(AgentRuntimeEngine):
    engine_name = "google_adk"

    def __init__(self, fallback_engine: AgentRuntimeEngine, *, adk_config: Optional[ADKConfig] = None):
        self._adk_config = adk_config or ADKConfig()
        self._ensure_adk_available()
        self._fallback_engine = fallback_engine

    def _ensure_adk_available(self) -> None:
        try:
            importlib.import_module("google.adk")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "google_adk runtime selected but Google ADK is not installed. "
                "Install optional dependencies with: pip install 'generic-agent-runtime[adk]' "
                "or install package `google-adk`."
            ) from e

    def stream(self, input_obj: Any, config: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        # Phase-1 migration path: preserve existing payload/state contracts while
        # validating ADK availability and runtime selection plumbing.
        return self._fallback_engine.stream(input_obj, config=config)

    def resume(self, answer: Any, config: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        return self._fallback_engine.resume(answer, config=config)

    def invoke(self, input_obj: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._fallback_engine.invoke(input_obj, config=config)
