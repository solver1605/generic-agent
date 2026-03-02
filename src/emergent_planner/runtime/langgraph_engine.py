"""LangGraph runtime adapter implementing AgentRuntimeEngine."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from langgraph.types import Command

from .engine import AgentRuntimeEngine


class LangGraphRuntimeEngine(AgentRuntimeEngine):
    engine_name = "langgraph"

    def __init__(self, app: Any):
        self.app = app

    def stream(self, input_obj: Any, config: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        return self.app.stream(input_obj, config=config or {}, stream_mode="values")

    def resume(self, answer: Any, config: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        return self.stream(Command(resume=answer), config=config)

    def invoke(self, input_obj: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        final: Dict[str, Any] = {}
        for st in self.stream(input_obj, config=config):
            final = st
        return final
