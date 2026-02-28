"""
Types for sub-agent orchestration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SubAgentTask(BaseModel):
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=1)
    constraints: List[str] = Field(default_factory=list)
    expected_output: str = Field(..., min_length=1)
    can_run_parallel: bool = True
    tool_overrides: Optional[List[str]] = None


class SubAgentExecutionConfig(BaseModel):
    max_workers: int = 4
    max_worker_turns: int = 8
    max_wall_time_s: float = 45.0
    max_retries: int = 1


@dataclass
class SubAgentError:
    task_id: str
    code: str
    message: str
    retryable: bool = False
    attempts: int = 1
    task_prompt: str = ""
    turn_traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubAgentResult:
    task_id: str
    title: str
    status: str
    task_prompt: str
    output: str
    summary: str
    worker_run_id: str
    attempts: int
    turns_used: int
    turn_traces: List[Dict[str, Any]] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    artifact_path: Optional[str] = None
    timings_ms: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubAgentRunRecord:
    request_id: str
    parent_run_id: str
    status: str
    summary: str
    results: List[SubAgentResult] = field(default_factory=list)
    errors: List[SubAgentError] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
            "errors": [e.to_dict() for e in self.errors],
            "stats": self.stats,
        }
