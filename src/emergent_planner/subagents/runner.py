"""
Worker execution helpers for sub-agents.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, ToolMessage

from ..config import ModelCard, build_llm_from_model_card
from ..graph import build_app
from ..policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy
from ..prompts import make_default_prompt_lib
from ..utils import extract_tool_calls, normalize_content
from .context import build_worker_initial_state
from .types import SubAgentError, SubAgentTask


@dataclass
class WorkerRunSuccess:
    task_prompt: str
    output: str
    summary: str
    turns_used: int
    worker_run_id: str
    timings_ms: Dict[str, int]
    turn_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkerRunFailure:
    error: SubAgentError
    worker_run_id: str
    turns_used: int
    timings_ms: Dict[str, int]
    task_prompt: str = ""
    turn_traces: List[Dict[str, Any]] = field(default_factory=list)


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


def _serialize_prompt_messages(messages: List[Any], *, max_chars: int, max_msgs: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in list(messages or [])[:max_msgs]:
        role = m.__class__.__name__.replace("Message", "").lower()
        txt = normalize_content(getattr(m, "content", ""))
        out.append(
            {
                "role": role,
                "content": txt[:max_chars] + ("...[truncated]..." if len(txt) > max_chars else ""),
            }
        )
    return out


def _finalize_turn_traces(traces_by_turn: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k in sorted(traces_by_turn.keys()):
        item = traces_by_turn[k]
        if not (item.get("prompt_messages") or item.get("tool_calls") or item.get("tool_outputs")):
            continue
        out.append(item)
    return out


def run_worker_task_once(
    *,
    task: SubAgentTask,
    parent_state: Dict[str, Any],
    parent_run_id: str,
    request_id: str,
    task_index: int,
    model_card: ModelCard,
    worker_tools: List[Any],
    budget_policy: BudgetPolicy,
    tool_log_policy: ToolLogPolicy,
    summary_policy: SummaryPolicy,
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
            budget_policy=budget_policy,
            tool_log_policy=tool_log_policy,
            summary_policy=summary_policy,
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
    task_prompt = str((init_state.get("runtime", {}) or {}).get("subagent_task_brief", ""))
    turn_cap_hit = False
    timeout_hit = False
    prev_hist_len = len(list(init_state.get("history", []) or []))
    traces_by_turn: Dict[int, Dict[str, Any]] = {}
    prompt_preview_chars = max(300, int(budget_policy.max_tool_snippet_chars))
    tool_preview_chars = max(300, int(tool_log_policy.max_inline_chars))
    max_prompt_msgs = max(8, int(budget_policy.max_skills_top_k) + 6)

    try:
        for st in app.stream(init_state, config=config, stream_mode="values"):
            last_state = st
            runtime = st.get("runtime", {}) or {}
            turns = int(runtime.get("turn_index", 0))
            trace = traces_by_turn.setdefault(
                turns,
                {
                    "turn_index": turns,
                    "prompt_messages": [],
                    "tool_calls": [],
                    "tool_outputs": [],
                },
            )
            prompt_msgs = list(st.get("messages", []) or [])
            if prompt_msgs:
                trace["prompt_messages"] = _serialize_prompt_messages(
                    prompt_msgs,
                    max_chars=prompt_preview_chars,
                    max_msgs=max_prompt_msgs,
                )

            hist = list(st.get("history", []) or [])
            if prev_hist_len < 0:
                prev_hist_len = 0
            new_msgs = hist[prev_hist_len:]
            prev_hist_len = len(hist)
            for nm in new_msgs:
                if isinstance(nm, AIMessage):
                    for c in extract_tool_calls(nm):
                        args = c.get("args")
                        if isinstance(args, str):
                            args_str = args[:prompt_preview_chars]
                            args = args_str + ("...[truncated]..." if len(args) > prompt_preview_chars else "")
                        trace["tool_calls"].append(
                            {
                                "id": c.get("id"),
                                "name": c.get("name"),
                                "args": args,
                            }
                        )
                elif isinstance(nm, ToolMessage):
                    txt = normalize_content(getattr(nm, "content", ""))
                    trace["tool_outputs"].append(
                        {
                            "tool_call_id": getattr(nm, "tool_call_id", None),
                            "content": txt[:tool_preview_chars] + ("...[truncated]..." if len(txt) > tool_preview_chars else ""),
                            "content_len": len(txt),
                        }
                    )

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
                return None, WorkerRunFailure(
                    err,
                    worker_run_id,
                    turns,
                    {"total": int((time.perf_counter() - t0) * 1000)},
                    task_prompt=task_prompt,
                    turn_traces=_finalize_turn_traces(traces_by_turn),
                )

    except Exception as e:
        err = SubAgentError(task_id=task.id, code="worker_execution_failed", message=f"{type(e).__name__}: {e}", retryable=True)
        turns = int((last_state.get("runtime", {}) or {}).get("turn_index", 0))
        return None, WorkerRunFailure(
            err,
            worker_run_id,
            turns,
            {"total": int((time.perf_counter() - t0) * 1000)},
            task_prompt=task_prompt,
            turn_traces=_finalize_turn_traces(traces_by_turn),
        )

    turns_used = int((last_state.get("runtime", {}) or {}).get("turn_index", 0))
    if timeout_hit:
        err = SubAgentError(task_id=task.id, code="worker_timeout", message="Worker exceeded wall-time budget.", retryable=True)
        return None, WorkerRunFailure(
            err,
            worker_run_id,
            turns_used,
            {"total": int((time.perf_counter() - t0) * 1000)},
            task_prompt=task_prompt,
            turn_traces=_finalize_turn_traces(traces_by_turn),
        )
    if turn_cap_hit:
        err = SubAgentError(task_id=task.id, code="worker_turn_cap", message="Worker exceeded turn budget.", retryable=True)
        return None, WorkerRunFailure(
            err,
            worker_run_id,
            turns_used,
            {"total": int((time.perf_counter() - t0) * 1000)},
            task_prompt=task_prompt,
            turn_traces=_finalize_turn_traces(traces_by_turn),
        )

    output = _extract_final_ai_text(last_state)
    if not output.strip():
        output = "Worker completed without a textual final answer."

    return (
        WorkerRunSuccess(
            task_prompt=task_prompt,
            turn_traces=_finalize_turn_traces(traces_by_turn),
            output=output,
            summary=_short_summary(output),
            turns_used=turns_used,
            worker_run_id=worker_run_id,
            timings_ms={"total": int((time.perf_counter() - t0) * 1000)},
        ),
        None,
    )
