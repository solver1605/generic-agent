"""
End-to-end sub-agent orchestration.
"""
from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..config import (
    AgentConfig,
    ModelCard,
    default_agent_config,
    load_agent_config,
    resolve_runtime_policies,
)
from ..runtime.factory import resolve_runtime_engine
from ..tool_registry import select_tools, tool_name
from .artifacts import persist_task_artifact
from .policy import resolve_worker_tool_names
from .runner import run_worker_task_once
from .types import (
    SubAgentError,
    SubAgentExecutionConfig,
    SubAgentResult,
    SubAgentRunRecord,
    SubAgentTask,
)


def _resolve_execution(
    cfg: AgentConfig,
    parent_runtime: Dict[str, Any],
    execution: SubAgentExecutionConfig,
) -> SubAgentExecutionConfig:
    scfg = getattr(cfg, "subagents", default_agent_config().subagents)
    max_workers = int(parent_runtime.get("subagent_max_workers", execution.max_workers or scfg.max_workers_default))
    max_worker_turns = int(
        parent_runtime.get("subagent_max_worker_turns", execution.max_worker_turns or scfg.max_worker_turns_default)
    )
    max_wall = float(parent_runtime.get("subagent_max_wall_time_s", execution.max_wall_time_s or scfg.max_wall_time_s_default))
    max_retries = int(execution.max_retries if execution.max_retries is not None else scfg.max_retries_default)

    return SubAgentExecutionConfig(
        max_workers=max(1, min(max_workers, scfg.max_workers_limit)),
        max_worker_turns=max(1, min(max_worker_turns, scfg.max_worker_turns_limit)),
        max_wall_time_s=max(1.0, min(max_wall, scfg.max_wall_time_s_limit)),
        max_retries=max(0, min(max_retries, scfg.max_retries_limit)),
    )


def _resolve_model_card(cfg: AgentConfig, parent_runtime: Dict[str, Any]) -> ModelCard:
    model_card_id = parent_runtime.get("model_card_id")
    card = cfg.get_model_card(model_card_id)

    model_name = parent_runtime.get("model_name")
    thinking = parent_runtime.get("thinking_budget")

    if model_name:
        card = replace(card, model_name=str(model_name))
    if thinking is not None:
        card = replace(card, thinking_budget=int(thinking))

    return card


def _build_summary(results: List[SubAgentResult], errors: List[SubAgentError]) -> str:
    lines = []
    if results:
        lines.append("Completed tasks:")
        for r in results[:8]:
            lines.append(f"- {r.task_id}: {r.summary[:160]}")
    if errors:
        lines.append("Failed tasks:")
        for e in errors[:8]:
            lines.append(f"- {e.task_id}: {e.code} ({e.message[:160]})")
    if not lines:
        return "No sub-agent work was executed."
    return "\n".join(lines)


def run_subagents(
    *,
    tasks: List[SubAgentTask],
    execution: SubAgentExecutionConfig,
    parent_state: Dict[str, Any],
    all_tools: Iterable[Any],
    config_path: Path = Path("agent_config.yaml"),
) -> SubAgentRunRecord:
    cfg = load_agent_config(config_path)
    scfg = getattr(cfg, "subagents", default_agent_config().subagents)
    parent_runtime = dict(parent_state.get("runtime", {}) or {})
    parent_run_id = str(parent_runtime.get("run_id", "default_run"))
    request_id = f"subreq_{uuid.uuid4().hex[:10]}"

    if not scfg.enabled or not bool(parent_runtime.get("subagent_enabled", True)):
        err = SubAgentError(task_id="*", code="subagents_disabled", message="Sub-agents are disabled by config.")
        return SubAgentRunRecord(
            request_id=request_id,
            parent_run_id=parent_run_id,
            status="failed",
            summary="Sub-agent dispatch blocked: disabled by policy.",
            errors=[err],
            stats={"tasks_requested": len(tasks), "tasks_executed": 0},
        )

    if bool(parent_runtime.get("disable_subagent_spawn", False)) or int(parent_runtime.get("subagent_depth", 0)) >= 1:
        err = SubAgentError(
            task_id="*",
            code="subagent_recursion_blocked",
            message="Sub-agent recursion is disabled. Only supervisor can spawn sub-agents.",
            retryable=False,
        )
        return SubAgentRunRecord(
            request_id=request_id,
            parent_run_id=parent_run_id,
            status="failed",
            summary="Sub-agent dispatch blocked: recursion policy.",
            errors=[err],
            stats={"tasks_requested": len(tasks), "tasks_executed": 0},
        )

    resolved_exec = _resolve_execution(cfg, parent_runtime, execution)
    resolved_runtime_engine = resolve_runtime_engine(
        cfg=cfg,
        profile_runtime_engine=None,
        explicit_runtime_engine=str(parent_runtime.get("runtime_engine", "")).strip() or None,
    )
    model_card = _resolve_model_card(cfg, parent_runtime)
    budget_policy, tool_log_policy, summary_policy, resolved_profile_id = resolve_runtime_policies(
        cfg,
        str(parent_runtime.get("policy_profile_id", "")).strip() or None,
    )
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")

    all_tools_list = list(all_tools)
    all_tool_names = [tool_name(t) for t in all_tools_list]
    supervisor_enabled = list(parent_runtime.get("enabled_tool_names") or all_tool_names)

    scheduled: List[Tuple[int, SubAgentTask, List[str]]] = []
    errors: List[SubAgentError] = []

    for idx, task in enumerate(tasks):
        names = resolve_worker_tool_names(
            task,
            supervisor_enabled=supervisor_enabled,
            policy=scfg.tool_policy,
        )
        if not names:
            errors.append(
                SubAgentError(
                    task_id=task.id,
                    code="no_allowed_tools",
                    message="No tools allowed for this task under current policy.",
                    retryable=False,
                )
            )
            continue
        scheduled.append((idx, task, names))

    start = time.perf_counter()
    deadline = start + float(resolved_exec.max_wall_time_s)

    indexed_results: List[Tuple[int, SubAgentResult]] = []

    def _run_with_retries(idx: int, task: SubAgentTask, names: List[str]) -> Tuple[int, SubAgentResult | None, SubAgentError | None]:
        attempts = 0
        tool_objs = select_tools(all_tools_list, names)
        while attempts <= resolved_exec.max_retries:
            attempts += 1
            now = time.perf_counter()
            if now > deadline:
                err = SubAgentError(
                    task_id=task.id,
                    code="global_timeout",
                    message="Sub-agent request exceeded max wall time.",
                    retryable=False,
                    attempts=attempts,
                )
                return idx, None, err

            succ, fail = run_worker_task_once(
                task=task,
                parent_state=parent_state,
                parent_run_id=parent_run_id,
                request_id=request_id,
                task_index=idx,
                model_card=model_card,
                worker_tools=tool_objs,
                budget_policy=budget_policy,
                tool_log_policy=tool_log_policy,
                summary_policy=summary_policy,
                max_worker_turns=resolved_exec.max_worker_turns,
                max_wall_time_s=max(1.0, deadline - now),
                google_api_key=google_api_key,
                runtime_engine=resolved_runtime_engine,
                cfg=cfg,
            )

            if succ is not None:
                result = SubAgentResult(
                    task_id=task.id,
                    title=task.title,
                    status="ok",
                    task_prompt=getattr(succ, "task_prompt", ""),
                    output=succ.output,
                    summary=succ.summary,
                    worker_run_id=succ.worker_run_id,
                    attempts=attempts,
                    turns_used=succ.turns_used,
                    turn_traces=list(getattr(succ, "turn_traces", []) or []),
                    tool_names=names,
                    timings_ms=succ.timings_ms,
                )
                return idx, result, None

            if fail is None:
                err = SubAgentError(task_id=task.id, code="unknown_failure", message="Unknown worker failure", retryable=False)
                return idx, None, err

            err = fail.error
            err.attempts = attempts
            err.task_prompt = str(getattr(fail, "task_prompt", "") or "")
            err.turn_traces = list(getattr(fail, "turn_traces", []) or [])
            if err.retryable and attempts <= resolved_exec.max_retries:
                continue
            return idx, None, err

        err = SubAgentError(task_id=task.id, code="retry_exhausted", message="Worker retries exhausted", retryable=False)
        return idx, None, err

    serial_items = [it for it in scheduled if not it[1].can_run_parallel]
    parallel_items = [it for it in scheduled if it[1].can_run_parallel]

    for idx, task, names in serial_items:
        i, result, err = _run_with_retries(idx, task, names)
        if result is not None:
            indexed_results.append((i, result))
        if err is not None:
            errors.append(err)

    if parallel_items:
        max_workers = max(1, min(resolved_exec.max_workers, len(parallel_items)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_with_retries, idx, task, names) for idx, task, names in parallel_items]
            processed = set()
            remaining = max(0.1, deadline - time.perf_counter())
            try:
                for fut in as_completed(futures, timeout=remaining):
                    processed.add(fut)
                    i, result, err = fut.result()
                    if result is not None:
                        indexed_results.append((i, result))
                    if err is not None:
                        errors.append(err)
            except TimeoutError:
                completed = {f for f in futures if f.done() and f not in processed}
                done_indices = set()
                for f in completed:
                    i, result, err = f.result()
                    done_indices.add(i)
                    if result is not None:
                        indexed_results.append((i, result))
                    if err is not None:
                        errors.append(err)
                for f in processed:
                    try:
                        i, _, _ = f.result()
                        done_indices.add(i)
                    except Exception:
                        # Ignore here; this path only marks already-processed futures.
                        pass
                for f in futures:
                    if not f.done():
                        f.cancel()
                pending = [task for idx, task, _ in parallel_items if idx not in done_indices]
                for task in pending:
                    errors.append(
                        SubAgentError(
                            task_id=task.id,
                            code="global_timeout",
                            message="Sub-agent request exceeded max wall time before task completion.",
                            retryable=False,
                        )
                    )

    indexed_results.sort(key=lambda x: x[0])
    results = [r for _, r in indexed_results]

    for r in results:
        payload = {
            "task_id": r.task_id,
            "title": r.title,
            "status": r.status,
            "summary": r.summary,
            "output": r.output,
            "attempts": r.attempts,
            "turns_used": r.turns_used,
            "task_prompt": r.task_prompt,
            "turn_traces": r.turn_traces,
            "tool_names": r.tool_names,
            "timings_ms": r.timings_ms,
        }
        path = persist_task_artifact(
            artifact_root=scfg.artifact_dir,
            parent_run_id=parent_run_id,
            request_id=request_id,
            task_id=r.task_id,
            payload=payload,
        )
        r.artifact_path = path.as_posix()

    for e in errors:
        payload = {
            "task_id": e.task_id,
            "code": e.code,
            "message": e.message,
            "retryable": e.retryable,
            "attempts": e.attempts,
            "task_prompt": e.task_prompt,
            "turn_traces": e.turn_traces,
        }
        persist_task_artifact(
            artifact_root=scfg.artifact_dir,
            parent_run_id=parent_run_id,
            request_id=request_id,
            task_id=f"{e.task_id}_error",
            payload=payload,
        )

    status = "success"
    if errors and results:
        status = "partial"
    elif errors and not results:
        status = "failed"

    unique_failed_task_ids = sorted({str(e.task_id) for e in errors})
    stats = {
        "tasks_requested": len(tasks),
        "tasks_scheduled": len(scheduled),
        "tasks_completed": len(results),
        "tasks_failed": len(unique_failed_task_ids),
        "error_entries": len(errors),
        "max_workers_used": min(resolved_exec.max_workers, len(parallel_items)) if parallel_items else 1,
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
        "policy_profile_id": resolved_profile_id,
        "runtime_engine": resolved_runtime_engine,
    }

    summary = _build_summary(results, errors)

    return SubAgentRunRecord(
        request_id=request_id,
        parent_run_id=parent_run_id,
        status=status,
        summary=summary,
        results=results,
        errors=errors,
        stats=stats,
    )
