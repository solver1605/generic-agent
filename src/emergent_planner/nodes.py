"""
LangGraph node functions and the instrument_node observability wrapper.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import ToolNode

from .context_manager import ContextManager
from .models import AgentState
from .policies import SummaryPolicy, ToolLogPolicy
from .utils import (
    _classify_error,
    _coarse_size,
    _fingerprint_prompt,
    _messages_to_compact_text,
    _now_ms,
    _safe_len,
    extract_tool_calls,
    get_history_from_state,
    get_prompt_messages_from_state,
    msg_tokens,
    normalize_content,
)


# ---------------------------------------------------------------------------
# Tool output persistence
# ---------------------------------------------------------------------------

def persist_tool_outputs_node(
    state: Dict[str, Any], policy: ToolLogPolicy
) -> Dict[str, Any]:
    history: List[BaseMessage] = state.get("history", [])
    runtime = state.get("runtime", {}) or {}
    run_id = runtime.get("run_id", "default_run")

    out_history: List[BaseMessage] = []
    changed = False

    base_dir = policy.artifacts_dir / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    for m in history:
        if isinstance(m, ToolMessage):
            content = m.content or ""
            if len(content) > policy.max_inline_chars:
                tool_call_id = m.tool_call_id or "unknown_tool_call"
                file_path = base_dir / f"{tool_call_id}.txt"
                file_path.write_text(content, encoding="utf-8")

                snippet = content[:policy.max_inline_chars] + "\n...[truncated]..."
                pointer = f"\n\nFull tool output saved at: {file_path.as_posix()}"
                out_history.append(
                    ToolMessage(content=snippet + pointer, tool_call_id=m.tool_call_id)
                )
                changed = True
            else:
                out_history.append(m)
        else:
            out_history.append(m)

    if not changed:
        return {}

    runtime = {**runtime, "after_tool": True, "tool_logs_persisted": True}
    return {"history": out_history, "runtime": runtime}


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

def summarize_node(
    state: Dict[str, Any], *, llm, policy: SummaryPolicy
) -> Dict[str, Any]:
    history: List[BaseMessage] = state.get("history", [])
    if len(history) <= policy.summarize_when_history_len_exceeds:
        return {}

    keep_n = policy.keep_last_n_messages
    old = history[:-keep_n]
    recent = history[-keep_n:]

    memory = state.get("memory", {}) or {}
    existing_summary = memory.get("summary", "")

    old_text = _messages_to_compact_text(old)

    prompt = [
        SystemMessage(content=(
            "You are a summarization engine for an agent runtime.\n"
            "Update the running summary so it preserves: user goals, constraints, decisions, "
            "tool outcomes, and open tasks.\n"
            "Be concise, structured, and factual. No fluff."
        )),
        HumanMessage(content=(
            f"Existing summary:\n{existing_summary}\n\n"
            f"New conversation chunk to merge:\n{old_text}\n\n"
            "Return an updated summary with headings:\n"
            "- Goals\n- Decisions\n- Tool outcomes\n- Open tasks\n- Constraints\n"
        )),
    ]

    resp = llm.invoke(prompt)
    updated_summary = resp.content

    memory = {**memory, "summary": updated_summary}
    runtime = {**(state.get("runtime", {}) or {}), "summarized": True}

    return {"memory": memory, "history": recent, "runtime": runtime}


# ---------------------------------------------------------------------------
# Skill activation
# ---------------------------------------------------------------------------

def _parse_tool_json_payload(m: ToolMessage) -> Optional[Dict[str, Any]]:
    """
    Parse tool payloads robustly across provider/tool content shapes.
    """
    raw = normalize_content(getattr(m, "content", None)).strip()
    if not raw:
        return None

    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                candidates.append(inner)
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1].strip())

    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def activate_skill_from_tool_result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the most recent ToolMessage contains a load_skill JSON payload, inject the
    skill body into runtime so the ContextManager can include it in the next prompt.
    """
    hist = state.get("history", [])
    if not hist:
        return {}

    for m in reversed(hist):
        if not isinstance(m, ToolMessage):
            continue
        payload = _parse_tool_json_payload(m)
        if payload is None:
            continue

        if isinstance(payload, dict) and payload.get("body") and payload.get("name"):
            runtime = state.get("runtime", {}) or {}
            runtime = {
                **runtime,
                "active_skill_name": payload["name"],
                "active_skill_body": payload["body"],
                "active_skill_meta": payload.get("meta", {}),
            }
            return {"runtime": runtime}

    return {}


def activate_subagent_from_tool_result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect spawn_subagents tool output and merge deterministic records into runtime:
      - runtime["subagent_runs"] append-only by request_id
      - runtime["subagent_results"] task-id map (latest wins)
      - runtime["subagent_stats"] latest stats object
    """
    hist = state.get("history", [])
    if not hist:
        return {}

    payload = None
    for m in reversed(hist):
        if not isinstance(m, ToolMessage):
            continue
        obj = _parse_tool_json_payload(m)
        if obj is None:
            continue
        if isinstance(obj, dict) and obj.get("__tool") == "spawn_subagents":
            payload = obj
            break

    if not payload:
        return {}

    request_id = str(payload.get("request_id", "")).strip()
    if not request_id:
        return {}

    runtime = dict(state.get("runtime", {}) or {})
    runs = list(runtime.get("subagent_runs", []) or [])
    if not any(str(r.get("request_id", "")) == request_id for r in runs if isinstance(r, dict)):
        runs.append(
            {
                "request_id": request_id,
                "status": payload.get("status", "unknown"),
                "summary": payload.get("summary", ""),
                "results_count": len(payload.get("results", []) or []),
                "errors_count": len(payload.get("errors", []) or []),
                "stats": payload.get("stats", {}),
            }
        )

    results_map = dict(runtime.get("subagent_results", {}) or {})
    for r in payload.get("results", []) or []:
        if isinstance(r, dict) and r.get("task_id"):
            results_map[str(r["task_id"])] = r
    errors_map = dict(runtime.get("subagent_errors", {}) or {})
    for e in payload.get("errors", []) or []:
        if isinstance(e, dict) and e.get("task_id"):
            errors_map[str(e["task_id"])] = e

    runtime["subagent_runs"] = runs
    runtime["subagent_results"] = results_map
    runtime["subagent_errors"] = errors_map
    runtime["subagent_stats"] = payload.get("stats", {})
    runtime["last_subagent_request_id"] = request_id
    return {"runtime": runtime}


# ---------------------------------------------------------------------------
# Core processing nodes
# ---------------------------------------------------------------------------

def context_node(state: AgentState, ctx_mgr: ContextManager) -> Dict[str, Any]:
    runtime = state.get("runtime", {}) or {}
    if runtime.get("subagent_mode") and runtime.get("subagent_task_brief"):
        messages = ctx_mgr.compose_for_subagent(state, str(runtime.get("subagent_task_brief", "")))
    else:
        messages = ctx_mgr.compose(state)
    return {"messages": messages}


def llm_node(state: AgentState, llm) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    ai = llm.invoke(msgs)
    hist = state.get("history", [])
    runtime = state.get("runtime", {}) or {}
    runtime = {
        **runtime,
        "turn_index": runtime.get("turn_index", 0) + 1,
        "after_tool": False,
    }
    return {"history": hist + [ai], "runtime": runtime}


def tools_node(state: AgentState, tool_node_impl: ToolNode) -> Dict[str, Any]:
    """
    ToolNode expects {"messages": ...}. We feed it the history
    then append only the delta tool messages.
    """
    hist = state.get("history", [])
    res = tool_node_impl.invoke({"messages": hist})
    updated = res["messages"]
    runtime = state.get("runtime", {}) or {}
    runtime = {**runtime, "after_tool": True}
    return {"history": hist + updated, "runtime": runtime}


def persist_prompt_artifact_node(state: AgentState) -> Dict[str, Any]:
    runtime = state.get("runtime", {}) or {}
    turn = runtime.get("turn_index", 0)

    pm = get_prompt_messages_from_state(state)
    if not pm:
        return {}

    lines = []
    for m in pm:
        role = m.__class__.__name__.replace("Message", "").lower()
        lines.append(f"{role}:\n{normalize_content(getattr(m, 'content', None))}\n")

    prompt_text = "\n".join(lines)

    arts = list(runtime.get("prompt_artifacts", []) or [])
    arts.append({"turn": turn, "prompt_text": prompt_text})
    runtime = {**runtime, "prompt_artifacts": arts}

    return {"runtime": runtime}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def has_tool_calls(state: Dict[str, Any]) -> str:
    hist = state.get("history", [])
    if not hist:
        return "end"
    last = hist[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def should_summarize(state: Dict[str, Any], policy: SummaryPolicy) -> str:
    hist = state.get("history", [])
    return "summarize" if len(hist) > policy.summarize_when_history_len_exceeds else "skip"


def should_pause(state: Dict[str, Any]) -> str:
    runtime = state.get("runtime", {}) or {}
    if runtime.get("waiting_for_user"):
        return "pause"
    return "continue"


# ---------------------------------------------------------------------------
# instrument_node — observability wrapper
# ---------------------------------------------------------------------------

def instrument_node(
    name: str,
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    *,
    history_key: str = "history",
    prompt_key: str = "messages",
    capture_state_sizes: bool = True,
    capture_tool_calls: bool = True,
    capture_prompt_fingerprint: bool = True,
    capture_prompt_preview: bool = False,
    prompt_preview_chars: int = 1200,
    swallow_exceptions: bool = False,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Wrap a LangGraph node function with production-grade observability.

    Logs into state["telemetry"]:
      - run_id / trace_id / turn_index / step_id
      - node timing (start/end/elapsed_ms)
      - status ("ok" | "error") + error classification
      - updated keys returned by node
      - message counts + approximate token counts
      - tool calls observed in latest AI messages (optional)
      - prompt fingerprint (optional)

    Also updates runtime on error:
      runtime["last_error"], runtime["last_error_traceback"],
      runtime["last_failed_node"], runtime["after_tool"]=False
    """

    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        # correlation ids
        runtime_in = state.get("runtime", {}) or {}
        trace_id = runtime_in.get("trace_id") or runtime_in.get("run_id") or str(uuid.uuid4())
        run_id = runtime_in.get("run_id") or trace_id
        turn = int(runtime_in.get("turn_index", 0))
        step_id = str(uuid.uuid4())

        t0 = time.perf_counter()
        start_ms = _now_ms()

        out: Dict[str, Any] = {}
        status = "ok"
        err_text = ""
        err_type = ""
        err_class = ""
        tb = ""

        try:
            out = fn(state) or {}
            return out
        except Exception as e:
            status = "error"
            err_type = type(e).__name__
            err_class = _classify_error(e)
            tb = traceback.format_exc()
            err_text = f"{err_type}: {e}"

            runtime = dict(runtime_in)
            runtime["last_error"] = err_text
            runtime["last_error_traceback"] = tb
            runtime["last_failed_node"] = name
            runtime["after_tool"] = False

            out = dict(out) if out else {}
            out["runtime"] = runtime

            if not swallow_exceptions:
                raise
            return out
        finally:
            end_ms = _now_ms()
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            hist = state.get(history_key, []) or state.get("messages", []) or []
            pm = state.get(prompt_key, []) or []

            history_len = _safe_len(hist)
            prompt_len = _safe_len(pm)

            history_tokens = 0
            prompt_tokens = 0
            try:
                history_tokens = sum(msg_tokens(m) for m in hist[-20:])
            except Exception:
                pass
            try:
                prompt_tokens = sum(msg_tokens(m) for m in pm)
            except Exception:
                pass

            tool_calls: List[Dict[str, Any]] = []
            if capture_tool_calls:
                try:
                    for m in reversed(hist[-10:]):
                        if m.__class__.__name__ == "AIMessage":
                            tool_calls = extract_tool_calls(m)
                            if tool_calls:
                                break
                except Exception:
                    tool_calls = []

            prompt_fingerprint = ""
            prompt_preview = ""
            if capture_prompt_fingerprint:
                try:
                    prompt_fingerprint = _fingerprint_prompt(pm)
                except Exception:
                    pass
            if capture_prompt_preview and pm:
                try:
                    tail = pm[-6:]
                    chunks = []
                    for m in tail:
                        role = m.__class__.__name__
                        txt = normalize_content(getattr(m, "content", None))
                        chunks.append(f"{role}:\n{txt[:300]}")
                    prompt_preview = "\n\n".join(chunks)[:prompt_preview_chars]
                except Exception:
                    pass

            entry: Dict[str, Any] = {
                "trace_id": trace_id,
                "run_id": run_id,
                "turn_index": turn,
                "step_id": step_id,
                "node": name,
                "status": status,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "elapsed_ms": elapsed_ms,
                "updates": sorted(list((out or {}).keys())),
                "counts": {
                    "history_len": history_len,
                    "prompt_len": prompt_len,
                },
                "tokens_approx": {
                    "history_recent_window": history_tokens,
                    "prompt_total": prompt_tokens,
                },
            }

            if capture_state_sizes:
                entry["state_sizes"] = {
                    history_key: _coarse_size(hist),
                    prompt_key: _coarse_size(pm),
                    "runtime": _coarse_size(runtime_in),
                }

            if capture_tool_calls:
                entry["tool_calls"] = tool_calls

            if capture_prompt_fingerprint:
                entry["prompt_fingerprint"] = prompt_fingerprint
            if capture_prompt_preview:
                entry["prompt_preview"] = prompt_preview

            if status == "error":
                entry["error"] = {
                    "type": err_type,
                    "class": err_class,
                    "message": err_text,
                    "traceback": tb,
                }

            telemetry = list(state.get("telemetry", []) or [])
            telemetry.append(entry)

            if out is None:
                out = {}
            out["telemetry"] = telemetry

    return wrapped
