"""
Worker execution helpers for sub-agents.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage

from ..config import ModelCard, build_llm_from_model_card
from ..graph import build_app
from ..policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy
from ..prompts import make_default_prompt_lib
from ..utils import normalize_content
from .context import build_worker_initial_state
from .types import SubAgentError, SubAgentTask


@dataclass
class WorkerRunSuccess:
    output: str
    summary: str
    turns_used: int
    worker_run_id: str
    timings_ms: Dict[str, int]


@dataclass
class WorkerRunFailure:
    error: SubAgentError
    worker_run_id: str
    turns_used: int
    timings_ms: Dict[str, int]


WorkerRunOutcome = Tuple[Optional[WorkerRunSuccess], Optional[WorkerRunFailure]]


def _extract_final_ai_text(state: Dict[str, Any]) -> str:
    hist = list(state.get("history", []) or [])
    for m in reversed(hist):
        if isinstance(m, AIMessage):
            return normalize_content(getattr(m, "content", "") or "")
    return ""


def _short_summary(text: str, max_chars: int = 600) -> str:
    t = (text or "").strip()
    return t[:max_chars] + ("..." if len(t) > max_chars else "")


def run_worker_task_once(
    *,
    task: SubAgentTask,
    parent_state: Dict[str, Any],
    parent_run_id: str,
    request_id: str,
    task_index: int,
    model_card: ModelCard,
    worker_tools: List[Any],
    max_worker_turns: int,
    max_wall_time_s: float,
    google_api_key: str,
) -> WorkerRunOutcome:
    t0 = time.perf_counter()
    worker_run_id = f"{parent_run_id}:{request_id}:{task.id}:{task_index}:{uuid.uuid4().hex[:8]}"

    if not google_api_key:
        err = SubAgentError(task_id=task.id, code="missing_api_key", message="GOOGLE_API_KEY not set.", retryable=False)
        return None, WorkerRunFailure(err, worker_run_id, 0, {"total": int((time.perf_counter() - t0) * 1000)})

    try:
        llm = build_llm_from_model_card(model_card, google_api_key=google_api_key)
        llm_with_tools = llm.bind_tools(worker_tools)
        app = build_app(
            llm=llm_with_tools,
            prompt_lib=make_default_prompt_lib(),
            budget_policy=BudgetPolicy(),
            tool_log_policy=ToolLogPolicy(),
            summary_policy=SummaryPolicy(),
            tools=worker_tools,
        )
    except Exception as e:
        err = SubAgentError(task_id=task.id, code="worker_init_failed", message=f"{type(e).__name__}: {e}", retryable=False)
        return None, WorkerRunFailure(err, worker_run_id, 0, {"total": int((time.perf_counter() - t0) * 1000)})

    init_state = build_worker_initial_state(
        parent_state,
        task,
        worker_run_id=worker_run_id,
        parent_run_id=parent_run_id,
        request_id=request_id,
        subagent_depth=int((parent_state.get("runtime", {}) or {}).get("subagent_depth", 0)) + 1,
        tool_names=[getattr(t, "name", getattr(t, "__name__", str(t))) for t in worker_tools],
    )

    config = {
        "configurable": {"thread_id": worker_run_id},
        "recursion_limit": max(50, int(max_worker_turns) * 8),
    }

    last_state = init_state
    turn_cap_hit = False
    timeout_hit = False

    try:
        for st in app.stream(init_state, config=config, stream_mode="values"):
            last_state = st
            runtime = st.get("runtime", {}) or {}
            turns = int(runtime.get("turn_index", 0))
            if turns >= int(max_worker_turns):
                turn_cap_hit = True
                break
            if (time.perf_counter() - t0) > float(max_wall_time_s):
                timeout_hit = True
                break

            if st.get("__interrupt__"):
                err = SubAgentError(
                    task_id=task.id,
                    code="needs_clarification",
                    message="Worker requested clarification. Supervisor must handle user interaction.",
                    retryable=False,
                )
                return None, WorkerRunFailure(err, worker_run_id, turns, {"total": int((time.perf_counter() - t0) * 1000)})

    except Exception as e:
        err = SubAgentError(task_id=task.id, code="worker_execution_failed", message=f"{type(e).__name__}: {e}", retryable=True)
        turns = int((last_state.get("runtime", {}) or {}).get("turn_index", 0))
        return None, WorkerRunFailure(err, worker_run_id, turns, {"total": int((time.perf_counter() - t0) * 1000)})

    turns_used = int((last_state.get("runtime", {}) or {}).get("turn_index", 0))
    if timeout_hit:
        err = SubAgentError(task_id=task.id, code="worker_timeout", message="Worker exceeded wall-time budget.", retryable=True)
        return None, WorkerRunFailure(err, worker_run_id, turns_used, {"total": int((time.perf_counter() - t0) * 1000)})
    if turn_cap_hit:
        err = SubAgentError(task_id=task.id, code="worker_turn_cap", message="Worker exceeded turn budget.", retryable=True)
        return None, WorkerRunFailure(err, worker_run_id, turns_used, {"total": int((time.perf_counter() - t0) * 1000)})

    output = _extract_final_ai_text(last_state)
    if not output.strip():
        output = "Worker completed without a textual final answer."

    return (
        WorkerRunSuccess(
            output=output,
            summary=_short_summary(output),
            turns_used=turns_used,
            worker_run_id=worker_run_id,
            timings_ms={"total": int((time.perf_counter() - t0) * 1000)},
        ),
        None,
    )
