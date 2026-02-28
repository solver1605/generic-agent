"""
Tool selection policy for worker sub-agents.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from ..config import SubAgentToolPolicyConfig
from .types import SubAgentTask


def infer_task_type(task: SubAgentTask) -> str:
    text = f"{task.title} {task.objective}".lower()
    if any(k in text for k in ["research", "investigate", "compare", "survey"]):
        return "research"
    if any(k in text for k in ["analyze", "analysis", "evaluate", "compute"]):
        return "analysis"
    if any(k in text for k in ["write", "draft", "summarize", "report"]):
        return "writing"
    return "default"


def resolve_worker_tool_names(
    task: SubAgentTask,
    *,
    supervisor_enabled: List[str],
    policy: SubAgentToolPolicyConfig,
) -> List[str]:
    enabled = set(supervisor_enabled)
    deny = set(policy.denylist or [])

    task_type = infer_task_type(task)
    allowed_by_type = list((policy.allow_by_task_type or {}).get(task_type, []) or [])
    if not allowed_by_type:
        allowed_by_type = list((policy.allow_by_task_type or {}).get("default", []) or [])

    allowed = set(allowed_by_type)
    if task.tool_overrides:
        for t in task.tool_overrides:
            if t in (policy.permitted_overrides or []):
                allowed.add(t)

    selected = [t for t in supervisor_enabled if t in enabled and t in allowed and t not in deny]
    return selected
