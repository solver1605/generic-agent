"""
Streamlit UI for Emergent Planner.

Provides two modes:
- User View: chat-first interaction with HITL interrupt handling.
- Debug View: step/state inspection similar to notebook debug UI.

Run:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from src.emergent_planner import (
    DEFAULT_TOOLS,
    BudgetPolicy,
    SummaryPolicy,
    ToolLogPolicy,
    build_app,
    discover_skills,
    make_default_prompt_lib,
)
from src.emergent_planner.config import build_llm_from_model_card, load_agent_config
from src.emergent_planner.tool_registry import select_tools, tool_catalog, tool_name
from src.emergent_planner.utils import (
    _diff_states,
    _shallow_snapshot,
    extract_tool_calls,
    get_history_from_state,
    get_prompt_messages_from_state,
    msg_tokens,
    normalize_content,
)

load_dotenv()

SNAPSHOT_KEYS = [
    "history",
    "messages",
    "input_messages",
    "llm_input",
    "telemetry",
    "runtime",
    "memory",
    "__interrupt__",
]


def _normalize_interrupt_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize LangGraph interrupt payloads into a dict with:
    type, reason, question, choices, context, default.
    """
    if raw is None:
        return None

    # Common case: already a dict.
    if isinstance(raw, dict):
        return {
            "type": raw.get("type") or raw.get("kind") or "confirm",
            "reason": raw.get("reason", ""),
            "question": raw.get("question", "Please provide input to continue."),
            "choices": raw.get("choices"),
            "context": raw.get("context"),
            "default": raw.get("default"),
        }

    # Some runtimes surface a tuple/list containing Interrupt objects.
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return _normalize_interrupt_payload(raw[0])

    # LangGraph Interrupt object often has `.value`.
    value = getattr(raw, "value", None)
    if value is not None:
        return _normalize_interrupt_payload(value)

    # Fallback for opaque objects.
    return {
        "type": "confirm",
        "reason": "",
        "question": str(raw),
        "choices": None,
        "context": None,
        "default": None,
    }


def _extract_interrupt_payload(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    intr = state.get("__interrupt__")
    if not intr:
        return None
    if isinstance(intr, list) and len(intr) > 0:
        first = intr[0]
        if isinstance(first, dict) and "value" in first:
            return _normalize_interrupt_payload(first["value"])
        return _normalize_interrupt_payload(first)
    return _normalize_interrupt_payload(intr)


def _message_role(m: BaseMessage) -> str:
    return m.__class__.__name__.replace("Message", "").lower()


def _serialize_message(m: BaseMessage) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "role": _message_role(m),
        "content": normalize_content(getattr(m, "content", "")),
        "tokens_approx": msg_tokens(m),
    }
    calls = extract_tool_calls(m)
    if calls:
        data["tool_calls"] = calls
    tcid = getattr(m, "tool_call_id", None)
    if tcid:
        data["tool_call_id"] = tcid
    return data


def _resolve_subagent_cfg(cfg) -> Dict[str, Any]:
    scfg = getattr(cfg, "subagents", None)
    if scfg is None:
        # Backward-compatibility for older in-memory AgentConfig instances.
        return {"enabled": True, "max_workers_default": 4, "max_wall_time_s_default": 45.0}
    return {
        "enabled": bool(getattr(scfg, "enabled", True)),
        "max_workers_default": int(getattr(scfg, "max_workers_default", 4)),
        "max_wall_time_s_default": float(getattr(scfg, "max_wall_time_s_default", 45.0)),
    }


@st.cache_resource(show_spinner=False)
def _build_runtime(
    model_card_id: str,
    skills_root: str,
    model_name_override: str,
    thinking_budget_override: Optional[int],
    enabled_tool_names: tuple[str, ...],
) -> Dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")

    cfg = load_agent_config(Path("agent_config.yaml"))
    subcfg = _resolve_subagent_cfg(cfg)
    card = cfg.get_model_card(model_card_id=model_card_id)
    if model_name_override.strip():
        card = replace(card, model_name=model_name_override.strip())
    if thinking_budget_override is not None:
        card = replace(card, thinking_budget=thinking_budget_override)

    base_llm = build_llm_from_model_card(card, google_api_key=api_key)
    tools = select_tools(DEFAULT_TOOLS, enabled_tool_names)
    if not tools:
        raise ValueError("No tools are enabled. Enable at least one tool in the sidebar.")
    llm_with_tools = base_llm.bind_tools(tools)

    prompt_lib = make_default_prompt_lib()
    budget_policy = BudgetPolicy()
    tool_policy = ToolLogPolicy()
    summary_policy = SummaryPolicy()

    app = build_app(
        llm=llm_with_tools,
        prompt_lib=prompt_lib,
        skills_root=Path(skills_root),
        budget_policy=budget_policy,
        tool_log_policy=tool_policy,
        summary_policy=summary_policy,
        tools=tools,
    )

    skills = discover_skills(Path(skills_root))
    return {
        "app": app,
        "skills": skills,
        "model_card": card,
        "enabled_tool_names": [tool_name(t) for t in tools],
        "subagent_defaults": {
            "enabled": subcfg["enabled"],
            "max_workers": subcfg["max_workers_default"],
            "max_wall_time_s": subcfg["max_wall_time_s_default"],
        },
    }


def _init_session(runtime: Dict[str, Any]) -> None:
    if "run_id" in st.session_state:
        return

    run_id = str(uuid.uuid4())
    st.session_state.run_id = run_id
    st.session_state.thread_id = run_id
    st.session_state.steps = []
    st.session_state.last_snap = {}
    st.session_state.pending_interrupt = None
    st.session_state.current_state = {
        "history": [],
        "memory": {},
        "runtime": {
            "run_id": run_id,
            "turn_index": 0,
            "model_card_id": runtime["model_card"].id,
            "model_name": runtime["model_card"].model_name,
            "thinking_budget": runtime["model_card"].thinking_budget,
            "enabled_tool_names": runtime["enabled_tool_names"],
            "subagent_enabled": runtime["subagent_defaults"]["enabled"],
            "subagent_max_workers": runtime["subagent_defaults"]["max_workers"],
            "subagent_max_wall_time_s": runtime["subagent_defaults"]["max_wall_time_s"],
        },
        "skills": runtime["skills"],
    }


def _append_step(full_state: Dict[str, Any]) -> None:
    snap = _shallow_snapshot(full_state, SNAPSHOT_KEYS)
    prev = st.session_state.last_snap
    diff = _diff_states(prev, snap) if st.session_state.steps else {"note": "initial snapshot"}

    st.session_state.steps.append(
        {
            "idx": len(st.session_state.steps),
            "state": snap,
            "diff": diff,
        }
    )
    st.session_state.last_snap = snap


def _run_graph(input_obj: Any, runtime: Dict[str, Any]) -> None:
    app = runtime["app"]
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    interrupted = False
    final_state: Dict[str, Any] = st.session_state.current_state

    for full_state in app.stream(input_obj, config=config, stream_mode="values"):
        _append_step(full_state)
        final_state = full_state
        payload = _extract_interrupt_payload(full_state)
        if payload is not None:
            st.session_state.pending_interrupt = payload
            interrupted = True
            break

    st.session_state.current_state = final_state
    if not interrupted:
        st.session_state.pending_interrupt = None


def _apply_runtime_controls(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = dict(state.get("runtime", {}) or {})
    rt["subagent_enabled"] = bool(st.session_state.get("subagent_enabled_ui", rt.get("subagent_enabled", True)))
    rt["subagent_max_workers"] = int(st.session_state.get("subagent_max_workers_ui", rt.get("subagent_max_workers", 4)))
    rt["subagent_max_wall_time_s"] = float(
        st.session_state.get("subagent_max_wall_time_ui", rt.get("subagent_max_wall_time_s", 45.0))
    )
    if st.session_state.get("enabled_tool_names_ui"):
        rt["enabled_tool_names"] = list(st.session_state.get("enabled_tool_names_ui"))
    state = dict(state)
    state["runtime"] = rt
    return state


def _render_chat_history(show_tools: bool) -> None:
    hist = get_history_from_state(st.session_state.current_state)
    for m in hist:
        if isinstance(m, ToolMessage) and not show_tools:
            continue

        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.markdown(normalize_content(m.content))
            continue

        if isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                content = normalize_content(m.content)
                st.markdown(content if content.strip() else "_(tool-call-only response)_")
                calls = extract_tool_calls(m)
                if calls:
                    with st.expander("Tool calls", expanded=False):
                        st.json(calls)
            continue

        if isinstance(m, ToolMessage):
            with st.chat_message("assistant"):
                st.markdown("**Tool output**")
                st.code(normalize_content(m.content), language="text")
                if getattr(m, "tool_call_id", None):
                    st.caption(f"tool_call_id: {m.tool_call_id}")
            continue


def _render_subagent_runs(state: Dict[str, Any]) -> None:
    runtime = state.get("runtime", {}) or {}
    runs = list(runtime.get("subagent_runs", []) or [])
    if not runs:
        return

    st.markdown("### Sub-agent Runs")
    for run in reversed(runs[-8:]):
        if not isinstance(run, dict):
            continue
        request_id = run.get("request_id", "unknown")
        status = run.get("status", "unknown")
        results_count = run.get("results_count", 0)
        errors_count = run.get("errors_count", 0)
        label = f"{request_id} · status={status} · results={results_count} · errors={errors_count}"
        with st.expander(label, expanded=False):
            st.markdown(run.get("summary", "") or "_(no summary)_")
            st.json(run.get("stats", {}))


def _render_interrupt_card(runtime: Dict[str, Any]) -> None:
    payload = st.session_state.pending_interrupt
    if not payload:
        return

    kind = payload.get("type", "confirm")
    reason = payload.get("reason", "")
    question = payload.get("question", "Please provide input to continue.")
    context = payload.get("context", "") or ""
    choices = payload.get("choices", []) or []
    default = payload.get("default", "")

    st.warning("Action required to continue")
    with st.container(border=True):
        st.markdown("### Clarification Needed")
        if reason:
            st.caption(f"Reason: `{reason}`")
        st.markdown(f"**{question}**")
    if context:
        with st.expander("Context", expanded=False):
            st.markdown(context)

    with st.form("interrupt_form", clear_on_submit=False):
        if kind == "pick_one" and choices:
            answer = st.selectbox("Select one", options=choices, index=0)
        elif kind == "pick_many" and choices:
            answer = st.multiselect("Select one or more", options=choices)
        elif kind == "confirm":
            ans_default = str(default or "yes").lower()
            yes_default = ans_default in {"yes", "y", "true", "1"}
            answer = st.radio("Confirm", options=["yes", "no"], index=0 if yes_default else 1, horizontal=True)
        else:
            answer = st.text_input("Answer", value=str(default or ""))

        submitted = st.form_submit_button("Submit and Continue")

    if submitted:
        _run_graph(Command(resume=answer), runtime)
        st.rerun()


def _run_user_turn(prompt: str, runtime: Dict[str, Any]) -> None:
    cur_state = _apply_runtime_controls(st.session_state.current_state)
    hist = list(cur_state.get("history", []))
    hist.append(HumanMessage(content=prompt))

    next_state = {
        **cur_state,
        "history": hist,
    }

    _run_graph(next_state, runtime)


def _render_user_view(runtime: Dict[str, Any]) -> None:
    st.subheader("User View")
    show_tools = st.toggle("Show tool messages in chat", value=False)

    _render_chat_history(show_tools=show_tools)
    _render_subagent_runs(st.session_state.current_state)

    if st.session_state.pending_interrupt:
        _render_interrupt_card(runtime)
        return

    prompt = st.chat_input("Send a message")
    if prompt:
        _run_user_turn(prompt, runtime)
        st.rerun()


def _render_debug_view() -> None:
    st.subheader("Debug View")

    current = st.session_state.current_state
    steps = st.session_state.steps
    rt = current.get("runtime", {}) or {}
    hist = get_history_from_state(current)
    pm = get_prompt_messages_from_state(current)
    telemetry = current.get("telemetry", []) or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("History msgs", len(hist))
    c2.metric("Prompt msgs", len(pm))
    c3.metric("Steps", len(steps))
    c4.metric("Turn index", int(rt.get("turn_index", 0)))

    st.caption(f"run_id={st.session_state.run_id} | thread_id={st.session_state.thread_id}")

    if steps:
        idx = st.slider("Step", min_value=0, max_value=len(steps) - 1, value=len(steps) - 1)
        selected = steps[idx]
        snap = selected["state"]
        diff = selected["diff"]
    else:
        snap = current
        diff = {}

    tabs = st.tabs(["History", "Prompt", "Runtime", "Memory", "Sub-agents", "Telemetry", "Diff"])

    with tabs[0]:
        serial = [_serialize_message(m) for m in get_history_from_state(snap)]
        st.json(serial)

    with tabs[1]:
        serial = [_serialize_message(m) for m in get_prompt_messages_from_state(snap)]
        st.json(serial)

    with tabs[2]:
        st.json(snap.get("runtime", {}))

    with tabs[3]:
        st.json(snap.get("memory", {}))

    with tabs[4]:
        rt = snap.get("runtime", {}) or {}
        st.json(
            {
                "subagent_runs": rt.get("subagent_runs", []),
                "subagent_results": rt.get("subagent_results", {}),
                "subagent_stats": rt.get("subagent_stats", {}),
                "last_subagent_request_id": rt.get("last_subagent_request_id"),
            }
        )

    with tabs[5]:
        tel = snap.get("telemetry", []) or []
        st.json(tel[-25:])

    with tabs[6]:
        st.json(diff)


def _reset_session() -> None:
    for k in [
        "run_id",
        "thread_id",
        "steps",
        "last_snap",
        "pending_interrupt",
        "current_state",
        "enabled_tool_names_ui",
        "subagent_enabled_ui",
        "subagent_max_workers_ui",
        "subagent_max_wall_time_ui",
    ]:
        if k in st.session_state:
            del st.session_state[k]


def main() -> None:
    st.set_page_config(page_title="Emergent Planner UI", layout="wide")
    st.title("Emergent Planner")

    with st.sidebar:
        st.header("Configuration")
        cfg = load_agent_config(Path("agent_config.yaml"))
        subcfg = _resolve_subagent_cfg(cfg)
        card_ids = [c.id for c in cfg.model_cards]
        default_idx = card_ids.index(cfg.default_model_card) if cfg.default_model_card in card_ids else 0
        model_card_id = st.selectbox("Model card", options=card_ids, index=default_idx)

        selected = cfg.get_model_card(model_card_id)
        model_name = st.text_input("Model name override", value=selected.model_name)
        thinking_budget = st.number_input(
            "Thinking budget override",
            min_value=0,
            step=128,
            value=int(selected.thinking_budget or 0),
            help="Set 0 to disable extra reasoning budget for supported models.",
        )
        skills_root = st.text_input("Skills root", value=str(Path(".skills")))
        st.subheader("Available Tools")
        catalog = tool_catalog(DEFAULT_TOOLS)
        enabled_names: List[str] = []
        for entry in catalog:
            tname = entry["name"]
            checked = st.checkbox(tname, value=True, key=f"tool_enabled_{tname}")
            if checked:
                enabled_names.append(tname)
            desc = (entry["description"] or "").strip()
            if desc:
                st.caption(desc.splitlines()[0][:180])
        st.subheader("Sub-agent Controls")
        subagent_enabled_ui = st.checkbox(
            "Enable sub-agents",
            value=bool(st.session_state.get("subagent_enabled_ui", subcfg["enabled"])),
            key="subagent_enabled_ui",
        )
        st.number_input(
            "Sub-agent max workers",
            min_value=1,
            max_value=16,
            value=int(st.session_state.get("subagent_max_workers_ui", subcfg["max_workers_default"])),
            step=1,
            key="subagent_max_workers_ui",
        )
        st.number_input(
            "Sub-agent max wall time (s)",
            min_value=5.0,
            max_value=600.0,
            value=float(st.session_state.get("subagent_max_wall_time_ui", subcfg["max_wall_time_s_default"])),
            step=5.0,
            key="subagent_max_wall_time_ui",
        )
        st.caption("Set GOOGLE_API_KEY in .env or environment before running.")
        if st.button("Reset Session"):
            _reset_session()
            st.rerun()
        st.session_state["enabled_tool_names_ui"] = list(sorted(enabled_names))

    try:
        runtime = _build_runtime(
            model_card_id=model_card_id,
            skills_root=skills_root,
            model_name_override=model_name,
            thinking_budget_override=int(thinking_budget),
            enabled_tool_names=tuple(sorted(enabled_names)),
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.sidebar.caption(
        "Resolved: "
        f"{runtime['model_card'].id} | "
        f"{runtime['model_card'].provider} | "
        f"thinking_budget={runtime['model_card'].thinking_budget}"
    )
    st.sidebar.caption("Enabled tools: " + ", ".join(runtime["enabled_tool_names"]))
    st.sidebar.caption(
        "Sub-agents: "
        + ("enabled" if st.session_state.get("subagent_enabled_ui", runtime["subagent_defaults"]["enabled"]) else "disabled")
    )

    _init_session(runtime)

    user_tab, debug_tab = st.tabs(["User View", "Debug View"])
    with user_tab:
        _render_user_view(runtime)

    with debug_tab:
        _render_debug_view()


if __name__ == "__main__":
    main()
