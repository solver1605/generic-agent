"""
Parent-to-worker context packing.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from .types import SubAgentTask


def _recent_non_tool_history(history: List[Any], limit: int = 6) -> List[Any]:
    out: List[Any] = []
    for m in reversed(history or []):
        if isinstance(m, (HumanMessage, AIMessage)):
            out.append(m)
        if len(out) >= limit:
            break
    out.reverse()
    return out


def _first_user_goal(history: List[Any]) -> str:
    for m in history or []:
        if isinstance(m, HumanMessage):
            return str(getattr(m, "content", "") or "")
    return ""


def build_worker_initial_state(
    parent_state: Dict[str, Any],
    task: SubAgentTask,
    *,
    worker_run_id: str,
    parent_run_id: str,
    request_id: str,
    subagent_depth: int,
    tool_names: List[str],
) -> Dict[str, Any]:
    hist = list(parent_state.get("history", []) or [])
    memory = dict(parent_state.get("memory", {}) or {})
    skills = list(parent_state.get("skills", []) or [])
    parent_runtime = dict(parent_state.get("runtime", {}) or {})

    goal = _first_user_goal(hist)
    summary = memory.get("summary", "")
    plan = memory.get("plan", "")

    task_brief_lines = [
        f"Task ID: {task.id}",
        f"Title: {task.title}",
        f"Objective: {task.objective}",
        f"Expected output: {task.expected_output}",
    ]
    if task.constraints:
        task_brief_lines.append("Constraints:")
        for c in task.constraints:
            task_brief_lines.append(f"- {c}")
    if goal:
        task_brief_lines.append(f"Parent user goal: {goal[:1200]}")
    if summary:
        task_brief_lines.append(f"Parent summary: {summary[:2000]}")
    if plan:
        task_brief_lines.append(f"Current plan excerpt: {plan[:1200]}")

    worker_prompt = "\n".join(task_brief_lines)

    history = _recent_non_tool_history(hist, limit=4)
    history.append(HumanMessage(content=worker_prompt))

    runtime = {
        "run_id": worker_run_id,
        "turn_index": 0,
        "after_tool": False,
        "subagent_mode": True,
        "subagent_depth": subagent_depth,
        "subagent_request_id": request_id,
        "subagent_task_id": task.id,
        "subagent_task_brief": worker_prompt,
        "parent_run_id": parent_run_id,
        "enabled_tool_names": tool_names,
        "disable_hitl": True,
        "disable_subagent_spawn": True,
        "model_card_id": parent_runtime.get("model_card_id"),
        "model_name": parent_runtime.get("model_name"),
        "thinking_budget": parent_runtime.get("thinking_budget"),
    }

    # Keep memory scoped and compact.
    worker_memory = {}
    if summary:
        worker_memory["summary"] = summary[:4000]
    if plan:
        worker_memory["plan"] = plan[:2000]

    return {
        "history": history,
        "memory": worker_memory,
        "runtime": runtime,
        "skills": skills,
    }
