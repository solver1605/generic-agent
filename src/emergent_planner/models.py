"""
Data models for the Emergent Planner agent framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal


# ---------------------------------------------------------------------------
# State reducers
# ---------------------------------------------------------------------------

def file_reducer(left, right):
    """Merge two file-system dicts: right values override left."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


# ---------------------------------------------------------------------------
# Simple result / value objects
# ---------------------------------------------------------------------------

@dataclass
class PythonReplResult:
    """Structured result returned by the python_repl tool."""
    stdout: str
    result_repr: str
    error: str
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------

@dataclass
class PromptCard:
    name: str
    text: str
    tags: set = field(default_factory=set)
    priority: int = 50  # lower = earlier


@dataclass
class PromptLibrary:
    cards: List[PromptCard]

    def select(self, selector: Callable[[PromptCard], bool]) -> List[PromptCard]:
        return sorted([c for c in self.cards if selector(c)], key=lambda c: c.priority)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

@dataclass
class SkillMeta:
    name: str
    description: str
    path: Path
    meta: Dict[str, Any]
    body: Optional[str] = None  # loaded on demand only


# ---------------------------------------------------------------------------
# Debug / recording types
# ---------------------------------------------------------------------------

@dataclass
class StepSnapshot:
    step_idx: int
    node: str
    update: Dict[str, Any]                      # per-node diff/update (stream_mode="updates")
    state_view: Dict[str, Any] = field(default_factory=dict)  # selected state keys


@dataclass
class Step:
    idx: int
    state: Dict[str, Any]          # full snapshot (values stream)
    diff: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    reason: Literal["plan_created", "plan_changed", "clarification", "risky_action"] = "plan_changed"
    question: str = Field(..., description="What to ask the user.")
    kind: Literal["confirm", "clarify", "pick_one", "pick_many"] = "confirm"
    choices: Optional[List[str]] = None
    context: Optional[str] = Field(None, description="Plan snippet / diff / summary to show.")
    default: Optional[str] = Field(None, description="Default answer if UI supports quick submit.")


# ---------------------------------------------------------------------------
# Agent State (LangGraph TypedDict)
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    history: List[AnyMessage]          # ground truth conversation
    messages: List[AnyMessage]         # curated messages passed to LLM
    memory: Dict[str, Any]             # summaries / episodic notes
    runtime: Dict[str, Any]            # turn flags, last tool, errors, subagent_runs/results/stats
    skills: List[SkillMeta]            # discovered skills
    file_system: Annotated[Dict[str, Any], file_reducer]
    telemetry: List[Dict[str, Any]]    # per-node observability records
