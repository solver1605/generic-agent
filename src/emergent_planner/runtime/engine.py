"""Runtime engine interface for supervisor and worker execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional


class AgentRuntimeEngine(ABC):
    """Abstract execution interface for agent runtimes."""

    engine_name: str = "unknown"

    @abstractmethod
    def stream(self, input_obj: Any, config: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        """Yield full state snapshots for each runtime step."""

    @abstractmethod
    def resume(self, answer: Any, config: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        """Resume an interrupted run using user-provided answer."""

    @abstractmethod
    def invoke(self, input_obj: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run to completion and return the final state snapshot."""
