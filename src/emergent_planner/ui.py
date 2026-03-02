"""
Streamlit UI for Emergent Planner.

Provides two modes:
- User View: chat-first interaction with HITL interrupt handling.
- Debug View: step/state inspection similar to notebook debug UI.

Run:
  generic-agent-ui
  streamlit run src/emergent_planner/ui.py
"""
from __future__ import annotations

import os
import uuid
import hashlib
import time
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from emergent_planner import (
    DEFAULT_TOOLS,
    build_runtime_app,
    build_prompt_lib_for_profile,
    discover_skills_in_roots,
)
from emergent_planner.config import (
    build_llm_from_model_card,
    load_agent_config,
    resolve_runtime_policies,
)
from emergent_planner.skills import find_project_root
from emergent_planner.tool_loader import build_tool_catalog, resolve_tools_for_profile
from emergent_planner.tool_registry import tool_catalog, tool_name
from emergent_planner.runtime.factory import resolve_runtime_engine
from emergent_planner.utils import (
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

TOOL_GROUPS: List[tuple[str, set[str]]] = [
    ("Core File Tools", {"read_file", "read_file_range", "write_file"}),
    ("Office Document Tools", {"write_excel_file", "create_pptx_deck"}),
    ("Reasoning and Search", {"python_repl", "search_web"}),
    ("Agent Orchestration", {"load_skill", "spawn_subagents", "verify_with_user"}),
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
        return {
            "enabled": True,
            "max_workers_default": 4,
            "max_workers_limit": 16,
            "max_worker_turns_default": 8,
            "max_worker_turns_limit": 64,
            "max_wall_time_s_default": 45.0,
            "max_wall_time_s_limit": 600.0,
        }
    return {
        "enabled": bool(getattr(scfg, "enabled", True)),
        "max_workers_default": int(getattr(scfg, "max_workers_default", 4)),
        "max_workers_limit": int(getattr(scfg, "max_workers_limit", 16)),
        "max_worker_turns_default": int(getattr(scfg, "max_worker_turns_default", 8)),
        "max_worker_turns_limit": int(getattr(scfg, "max_worker_turns_limit", 64)),
        "max_wall_time_s_default": float(getattr(scfg, "max_wall_time_s_default", 45.0)),
        "max_wall_time_s_limit": float(getattr(scfg, "max_wall_time_s_limit", 600.0)),
    }


def _normalize_skill_key(name: str) -> str:
    raw = (name or "").strip().lower()
    if not raw:
        return ""
    import re

    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _resolve_skills_roots(skills_roots: List[str]) -> List[Path]:
    out: List[Path] = []
    project_root = find_project_root(Path.cwd())
    for item in skills_roots:
        p = Path(item).expanduser()
        if p.is_absolute():
            out.append(p.resolve())
            continue
        cwd_resolved = (Path.cwd() / p).resolve()
        if cwd_resolved.exists():
            out.append(cwd_resolved)
        else:
            out.append((project_root / p).resolve())
    deduped: List[Path] = []
    seen = set()
    for p in out:
        k = p.as_posix()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(p)
    return deduped


def _skills_signature(skills_roots: List[Path]) -> str:
    files: List[Path] = []
    for root in skills_roots:
        if root.exists():
            files.extend([p for p in root.rglob("SKILL.md") if p.is_file()])
    if not files:
        return "missing"
    h = hashlib.sha1()
    for p in sorted(files):
        st = p.stat()
        h.update(p.as_posix().encode("utf-8"))
        h.update(str(st.st_mtime_ns).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


def _apply_deep_research_preset(
    profile_ids: List[str],
    available_tool_names: List[str],
    subcfg: Dict[str, Any],
) -> None:
    target = "deep_research" if "deep_research" in profile_ids else (profile_ids[0] if profile_ids else "balanced")
    st.session_state["policy_profile_id_ui"] = target
    st.session_state["subagent_enabled_ui"] = True
    st.session_state["subagent_max_workers_ui"] = min(6, int(subcfg.get("max_workers_limit", 16)))
    st.session_state["subagent_max_worker_turns_ui"] = min(14, int(subcfg.get("max_worker_turns_limit", 64)))
    st.session_state["subagent_max_wall_time_ui"] = min(180.0, float(subcfg.get("max_wall_time_s_limit", 600.0)))
    for tname in ["search_web", "spawn_subagents", "load_skill", "verify_with_user"]:
        if tname in set(available_tool_names):
            st.session_state[f"tool_enabled_{tname}"] = True


def _artifact_root() -> Path:
    return (find_project_root(Path.cwd()) / "artifacts").resolve()


def _list_artifacts(*, session_only: bool = True) -> List[Path]:
    root = _artifact_root()
    if not root.exists():
        return []
    # Recursive scan to include nested artifact outputs (e.g. artifacts/research/*).
    files = [p for p in root.rglob("*") if p.is_file()]
    if session_only:
        started_at = float(st.session_state.get("session_started_at", 0.0))
        if started_at > 0:
            files = [p for p in files if p.stat().st_mtime >= (started_at - 1.0)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _artifact_label(path: Path) -> str:
    root = _artifact_root()
    try:
        rel = path.resolve().relative_to(root)
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _format_age(ts: float) -> str:
    delta = max(0.0, time.time() - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _render_artifact_preview(path: Path, *, key_prefix: str) -> None:
    if not path.exists():
        st.warning(f"Artifact not found: {path}")
        return

    suffix = path.suffix.lower()
    size = path.stat().st_size
    st.caption(f"{_artifact_label(path)} · {size} bytes")

    image_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    text_ext = {".txt", ".md", ".json", ".log", ".csv", ".yaml", ".yml", ".toml", ".py", ".sql"}

    if suffix in image_ext:
        st.image(path.as_posix(), use_container_width=True)
        return

    if suffix == ".pdf":
        data = path.read_bytes()
        st.download_button(
            "Download PDF",
            data=data,
            file_name=path.name,
            key=f"{key_prefix}_pdf_dl",
        )
        st.info("PDF preview is not inline-rendered; use download.")
        return

    if suffix in text_ext or size <= 200_000:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".json":
            try:
                import json as _json

                st.json(_json.loads(raw))
                return
            except Exception:
                pass
        if suffix == ".md":
            st.markdown(raw)
            return
        st.code(raw[:12000], language="text")
        return

    data = path.read_bytes()
    st.download_button(
        "Download artifact",
        data=data,
        file_name=path.name,
        key=f"{key_prefix}_dl",
    )
    st.info("Binary artifact preview is not supported inline.")


def _render_artifacts_sidebar() -> None:
    st.sidebar.subheader("Session Artifacts")
    st.sidebar.caption("Scanning recursively under artifacts/**")
    session_only = st.sidebar.checkbox("Session only", value=False, key="artifact_session_only_ui")
    files = _list_artifacts(session_only=session_only)

    if not files:
        root = _artifact_root()
        if not root.exists():
            st.sidebar.caption(f"No artifacts directory at: {root}")
        else:
            st.sidebar.caption("No artifacts found for this filter.")
        return

    st.sidebar.caption(f"Found: {len(files)}")
    options = [p.as_posix() for p in files]
    selected = st.sidebar.selectbox(
        "Artifact",
        options=options,
        format_func=lambda s: _artifact_label(Path(s)),
        key="artifact_selected_path_ui",
    )
    if selected:
        sel_path = Path(selected)
        st.sidebar.caption(f"Updated: {_format_age(sel_path.stat().st_mtime)}")

    with st.sidebar.expander("All artifacts", expanded=False):
        for p in files:
            st.caption(f"- {_artifact_label(p)} ({_format_age(p.stat().st_mtime)})")

    if st.sidebar.toggle("Inline preview", value=False, key="artifact_sidebar_preview_ui") and selected:
        with st.sidebar.expander("Preview", expanded=True):
            digest = hashlib.sha1(selected.encode("utf-8")).hexdigest()[:10]
            _render_artifact_preview(Path(selected), key_prefix=f"sidebar_artifact_{digest}")


def _render_artifacts_view() -> None:
    st.subheader("Artifacts")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.caption("Recursive view: artifacts/**/*")
        session_only = st.checkbox("Session only", value=False, key="artifact_session_only_tab_ui")
        files = _list_artifacts(session_only=session_only)
        if not files:
            st.info("No artifacts found.")
            return

        options = [p.as_posix() for p in files]
        selected = st.selectbox(
            "Select artifact",
            options=options,
            format_func=lambda s: _artifact_label(Path(s)),
            key="artifact_selected_path_tab_ui",
        )
        st.caption(f"Total: {len(files)}")

    with c2:
        if not selected:
            st.info("Select an artifact to preview.")
            return
        digest = hashlib.sha1(selected.encode("utf-8")).hexdigest()[:10]
        _render_artifact_preview(Path(selected), key_prefix=f"tab_artifact_{digest}")


@st.cache_resource(show_spinner=False)
def _build_runtime(
    config_path: str,
    agent_profile_id: str,
    runtime_engine: str,
    model_card_id: str,
    policy_profile_id: str,
    skills_signature: str,
    model_name_override: str,
    thinking_budget_override: Optional[int],
    enabled_tool_names: tuple[str, ...],
) -> Dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")

    cfg_path = Path(config_path)
    cfg = load_agent_config(cfg_path)
    config_dir = cfg_path.resolve().parent
    subcfg = _resolve_subagent_cfg(cfg)
    agent_profile = cfg.get_agent_profile(agent_profile_id)
    card = cfg.get_model_card(model_card_id=model_card_id)
    if model_name_override.strip():
        card = replace(card, model_name=model_name_override.strip())
    if thinking_budget_override is not None:
        card = replace(card, thinking_budget=thinking_budget_override)

    all_tools = build_tool_catalog(cfg, DEFAULT_TOOLS)
    profile_tools = resolve_tools_for_profile(all_tools, agent_profile)
    available_tool_names = [tool_name(t) for t in profile_tools]
    unknown = sorted(set(enabled_tool_names) - set(available_tool_names))
    if unknown:
        raise ValueError(
            "Selected tools are not available for this profile: " + ", ".join(unknown)
        )
    tools = resolve_tools_for_profile(
        all_tools,
        agent_profile,
        extra_allow=list(enabled_tool_names),
    )

    base_llm = build_llm_from_model_card(card, google_api_key=api_key)
    if not tools:
        raise ValueError("No tools are enabled. Enable at least one tool in the sidebar.")
    llm_with_tools = base_llm.bind_tools(tools)

    prompt_lib = build_prompt_lib_for_profile(cfg, agent_profile, config_dir=config_dir)
    budget_policy, tool_policy, summary_policy, resolved_profile_id = resolve_runtime_policies(
        cfg,
        policy_profile_id,
    )
    profile_meta = cfg.get_policy_profile(resolved_profile_id)
    skill_roots = _resolve_skills_roots(list(agent_profile.skills.roots or [".skills"]))
    allowlist_norm = sorted(
        {
            _normalize_skill_key(x)
            for x in list(agent_profile.skills.allowlist or [])
            if _normalize_skill_key(x)
        }
    )
    denylist_norm = sorted(
        {
            _normalize_skill_key(x)
            for x in list(agent_profile.skills.denylist or [])
            if _normalize_skill_key(x)
        }
    )

    app = build_runtime_app(
        llm=llm_with_tools,
        prompt_lib=prompt_lib,
        skills_root=(skill_roots[0] if skill_roots else Path(".skills")),
        budget_policy=budget_policy,
        tool_log_policy=tool_policy,
        summary_policy=summary_policy,
        tools=tools,
        engine=runtime_engine,
        cfg=cfg,
    )

    skills = discover_skills_in_roots(skill_roots, strict_scope=True)
    if allowlist_norm:
        allowed = set(allowlist_norm)
        skills = [s for s in skills if _normalize_skill_key(getattr(s, "name", "")) in allowed]
    if denylist_norm:
        denied = set(denylist_norm)
        skills = [s for s in skills if _normalize_skill_key(getattr(s, "name", "")) not in denied]
    return {
        "app": app,
        "skills": skills,
        "agent_profile_id": agent_profile.id,
        "agent_profile_description": agent_profile.description,
        "runtime_engine": runtime_engine,
        "model_card": card,
        "enabled_tool_names": [tool_name(t) for t in tools],
        "available_tool_names": available_tool_names,
        "skills_roots_resolved": [p.as_posix() for p in skill_roots],
        "skills_allowlist_norm": allowlist_norm,
        "skills_denylist_norm": denylist_norm,
        "subagent_defaults": {
            "enabled": subcfg["enabled"],
            "max_workers": subcfg["max_workers_default"],
            "max_worker_turns": subcfg["max_worker_turns_default"],
            "max_wall_time_s": subcfg["max_wall_time_s_default"],
        },
        "policy_profile_id": resolved_profile_id,
        "policy_profile_description": profile_meta.description,
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
    st.session_state.session_started_at = time.time()
    st.session_state.current_state = {
        "history": [],
        "memory": {},
        "runtime": {
            "run_id": run_id,
            "turn_index": 0,
            "model_card_id": runtime["model_card"].id,
            "model_name": runtime["model_card"].model_name,
            "thinking_budget": runtime["model_card"].thinking_budget,
            "agent_profile_id": runtime["agent_profile_id"],
            "enabled_tool_names": runtime["enabled_tool_names"],
            "skills_roots_resolved": list(runtime.get("skills_roots_resolved", [])),
            "skills_allowlist_norm": list(runtime.get("skills_allowlist_norm", [])),
            "skills_denylist_norm": list(runtime.get("skills_denylist_norm", [])),
            "subagent_enabled": runtime["subagent_defaults"]["enabled"],
            "subagent_max_workers": runtime["subagent_defaults"]["max_workers"],
            "subagent_max_worker_turns": runtime["subagent_defaults"]["max_worker_turns"],
            "subagent_max_wall_time_s": runtime["subagent_defaults"]["max_wall_time_s"],
            "policy_profile_id": runtime["policy_profile_id"],
            "runtime_engine": runtime.get("runtime_engine", "langgraph"),
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


def _run_graph(input_obj: Any, runtime: Dict[str, Any], *, resume_answer: Any | None = None) -> None:
    runtime_engine = runtime["app"]
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    interrupted = False
    final_state: Dict[str, Any] = st.session_state.current_state

    if resume_answer is None:
        stream_iter = runtime_engine.stream(input_obj, config=config)
    else:
        stream_iter = runtime_engine.resume(resume_answer, config=config)

    for full_state in stream_iter:
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
    rt["subagent_max_worker_turns"] = int(
        st.session_state.get("subagent_max_worker_turns_ui", rt.get("subagent_max_worker_turns", 8))
    )
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


def _parse_json_payload(raw: Any) -> Optional[Dict[str, Any]]:
    text = normalize_content(raw).strip()
    if not text:
        return None
    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                candidates.append(inner)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1].strip())

    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _render_subagent_turn_traces(turn_traces: List[Dict[str, Any]]) -> None:
    traces = [t for t in (turn_traces or []) if isinstance(t, dict)]
    if not traces:
        st.info("No per-turn traces captured.")
        return
    for i, trace in enumerate(traces):
        turn_idx = trace.get("turn_index", i)
        with st.expander(f"Turn {turn_idx}", expanded=False):
            prompt_msgs = list(trace.get("prompt_messages", []) or [])
            tool_calls = list(trace.get("tool_calls", []) or [])
            tool_outputs = list(trace.get("tool_outputs", []) or [])

            st.markdown("**Prompt passed to worker LLM**")
            if prompt_msgs:
                st.json(prompt_msgs)
            else:
                st.caption("No prompt snapshot captured for this turn.")

            st.markdown("**Tool calls from worker LLM**")
            if tool_calls:
                st.json(tool_calls)
            else:
                st.caption("No tool calls in this turn.")

            st.markdown("**Tool outputs received**")
            if tool_outputs:
                st.json(tool_outputs)
            else:
                st.caption("No tool outputs in this turn.")


def _render_subagents_debug_tab(snap: Dict[str, Any]) -> None:
    rt = snap.get("runtime", {}) or {}
    runs = list(rt.get("subagent_runs", []) or [])
    results_map = dict(rt.get("subagent_results", {}) or {})
    errors_map = dict(rt.get("subagent_errors", {}) or {})
    stats = dict(rt.get("subagent_stats", {}) or {})

    st.markdown("#### Runtime Snapshot")
    st.json(
        {
            "subagent_runs_count": len(runs),
            "subagent_results_count": len(results_map),
            "subagent_errors_count": len(errors_map),
            "last_subagent_request_id": rt.get("last_subagent_request_id"),
            "subagent_stats": stats,
        }
    )

    st.markdown("#### Worker Prompts")
    shown = 0
    for task_id in sorted(results_map.keys()):
        rec = results_map.get(task_id)
        if not isinstance(rec, dict):
            continue
        prompt = str(rec.get("task_prompt", "") or "").strip()
        if not prompt:
            continue
        shown += 1
        title = str(rec.get("title", "") or "").strip() or task_id
        status = str(rec.get("status", "unknown"))
        with st.expander(f"{task_id} · {title} · status={status}", expanded=False):
            st.markdown("**Prompt sent to this sub-agent**")
            st.code(prompt, language="text")
            if rec.get("summary"):
                st.markdown("**Result summary**")
                st.markdown(str(rec.get("summary")))
            st.markdown("**Per-turn trace**")
            _render_subagent_turn_traces(
                list(rec.get("turn_traces", []) or []),
            )
            if rec.get("artifact_path"):
                st.caption(f"Artifact: {rec.get('artifact_path')}")
    if shown == 0:
        st.info("No captured sub-agent worker prompts in current snapshot.")

    st.markdown("#### Failed Worker Traces")
    failed_shown = 0
    for task_id in sorted(errors_map.keys()):
        rec = errors_map.get(task_id)
        if not isinstance(rec, dict):
            continue
        failed_shown += 1
        code = str(rec.get("code", "error"))
        message = str(rec.get("message", ""))
        with st.expander(f"{task_id} · {code}", expanded=False):
            if message:
                st.caption(message)
            tp = str(rec.get("task_prompt", "") or "").strip()
            if tp:
                st.markdown("**Prompt sent to failed sub-agent**")
                st.code(tp, language="text")
            st.markdown("**Per-turn trace**")
            _render_subagent_turn_traces(list(rec.get("turn_traces", []) or []))
    if failed_shown == 0:
        st.info("No failed sub-agent traces in current runtime snapshot.")

    st.markdown("#### `spawn_subagents` Tool Calls")
    hist = get_history_from_state(snap)
    tool_msgs_by_id: Dict[str, ToolMessage] = {}
    for m in hist:
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None):
            tool_msgs_by_id[str(m.tool_call_id)] = m

    call_count = 0
    for m in hist:
        if not isinstance(m, AIMessage):
            continue
        for call in extract_tool_calls(m):
            if str(call.get("name", "")).strip() != "spawn_subagents":
                continue
            call_count += 1
            call_id = str(call.get("id") or "")
            raw_args = call.get("args")
            parsed_args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = raw_args

            payload = None
            if call_id and call_id in tool_msgs_by_id:
                payload = _parse_json_payload(tool_msgs_by_id[call_id].content)

            task_count = 0
            if isinstance(parsed_args, dict):
                task_count = len(list(parsed_args.get("tasks", []) or []))

            label = f"Call {call_count} · tool_call_id={call_id or 'unknown'} · tasks={task_count}"
            with st.expander(label, expanded=False):
                st.markdown("**Tool call arguments**")
                st.json(parsed_args if isinstance(parsed_args, (dict, list)) else {"raw_args": str(parsed_args)})
                if isinstance(payload, dict):
                    st.markdown("**Tool output payload**")
                    st.json(
                        {
                            "request_id": payload.get("request_id"),
                            "status": payload.get("status"),
                            "results_count": len(payload.get("results", []) or []),
                            "errors_count": len(payload.get("errors", []) or []),
                            "summary": payload.get("summary", ""),
                        }
                    )
                    prompts = [
                        r
                        for r in (payload.get("results", []) or [])
                        if isinstance(r, dict) and str(r.get("task_prompt", "") or "").strip()
                    ]
                    if prompts:
                        st.markdown("**Worker prompts captured in tool result**")
                        for r in prompts[:12]:
                            tid = str(r.get("task_id", "task"))
                            st.caption(tid)
                            st.code(str(r.get("task_prompt", "")), language="text")
                            _render_subagent_turn_traces(
                                list(r.get("turn_traces", []) or []),
                            )
                    error_items = [e for e in (payload.get("errors", []) or []) if isinstance(e, dict)]
                    if error_items:
                        st.markdown("**Failed sub-agent traces**")
                        for e in error_items[:12]:
                            tid = str(e.get("task_id", "task"))
                            code = str(e.get("code", "error"))
                            with st.expander(f"{tid} · {code}", expanded=False):
                                tp = str(e.get("task_prompt", "") or "").strip()
                                if tp:
                                    st.markdown("**Prompt sent to failed sub-agent**")
                                    st.code(tp, language="text")
                                _render_subagent_turn_traces(
                                    list(e.get("turn_traces", []) or []),
                                )
    if call_count == 0:
        st.info("No `spawn_subagents` tool calls captured in history.")


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
        _run_graph(None, runtime, resume_answer=answer)
        st.rerun()


def _run_user_turn(prompt: str, runtime: Dict[str, Any]) -> None:
    cur_state = _apply_runtime_controls(st.session_state.current_state)
    hist = list(cur_state.get("history", []))
    hist.append(HumanMessage(content=prompt))

    next_state = {
        **cur_state,
        "history": hist,
    }

    _run_graph(next_state, runtime, resume_answer=None)


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
        _render_subagents_debug_tab(snap)

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
        "agent_profile_id_ui",
        "runtime_engine_ui",
        "enabled_tool_names_ui",
        "policy_profile_id_ui",
        "subagent_enabled_ui",
        "subagent_max_workers_ui",
        "subagent_max_worker_turns_ui",
        "subagent_max_wall_time_ui",
        "session_started_at",
        "artifact_session_only_ui",
        "artifact_session_only_tab_ui",
        "artifact_selected_path_ui",
        "artifact_selected_path_tab_ui",
        "artifact_sidebar_preview_ui",
    ]:
        if k in st.session_state:
            del st.session_state[k]


def _render_loaded_skills_sidebar(runtime: Dict[str, Any]) -> None:
    skills = list(runtime.get("skills", []) or [])
    st.sidebar.subheader("Loaded Skills")
    st.sidebar.caption(f"Discovered: {len(skills)}")
    if not skills:
        roots = runtime.get("skills_roots_resolved", [".skills"])
        st.sidebar.warning(f"No skills discovered under: {', '.join(roots)}")
        return

    with st.sidebar.expander("View skills", expanded=False):
        for sk in skills:
            name = getattr(sk, "name", "unknown")
            desc = (getattr(sk, "description", "") or "").strip()
            path = str(getattr(sk, "path", ""))
            st.markdown(f"**{name}**")
            if desc:
                st.caption(desc)
            if path:
                st.code(path, language="text")


def _tool_group_for_name(name: str) -> str:
    for label, names in TOOL_GROUPS:
        if name in names:
            return label
    return "Other Tools"


def _render_tool_selector(catalog: List[Dict[str, str]]) -> List[str]:
    enabled_names: List[str] = []
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for entry in catalog:
        grouped.setdefault(_tool_group_for_name(entry["name"]), []).append(entry)

    ordered_groups = [label for label, _ in TOOL_GROUPS]
    if "Other Tools" in grouped:
        ordered_groups.append("Other Tools")

    for group_label in ordered_groups:
        entries = sorted(grouped.get(group_label, []), key=lambda e: e["name"])
        if not entries:
            continue
        expanded = group_label == "Office Document Tools"
        with st.expander(group_label, expanded=expanded):
            for entry in entries:
                tname = entry["name"]
                checked = st.checkbox(tname, value=True, key=f"tool_enabled_{tname}")
                if checked:
                    enabled_names.append(tname)
                desc = (entry["description"] or "").strip()
                if desc:
                    st.caption(desc.splitlines()[0][:180])
    return enabled_names


def _resolve_initial_config_path() -> Path:
    default_config_path = os.environ.get("GENERIC_AGENT_CONFIG", "agent_config.yaml")
    config_path_text = str(st.session_state.get("config_path_ui", default_config_path)).strip()
    return Path(config_path_text or default_config_path).expanduser()


def _resolve_streamlit_branding(config_path: Path) -> tuple[str, str]:
    default_app_name = "Emergent Planner"
    default_page_title = "Emergent Planner UI"
    try:
        cfg = load_agent_config(config_path)
        app_name = str(getattr(cfg.streamlit, "app_name", default_app_name) or "").strip() or default_app_name
        page_title = str(getattr(cfg.streamlit, "page_title", default_page_title) or "").strip() or default_page_title
        return app_name, page_title
    except Exception:
        return default_app_name, default_page_title


def main() -> None:
    initial_config_path = _resolve_initial_config_path()
    app_name, page_title = _resolve_streamlit_branding(initial_config_path)
    st.set_page_config(page_title=page_title, layout="wide")
    st.title(app_name)

    with st.sidebar:
        st.header("Configuration")
        default_config_path = os.environ.get("GENERIC_AGENT_CONFIG", "agent_config.yaml")
        config_path_text = st.text_input(
            "Config path",
            value=str(st.session_state.get("config_path_ui", initial_config_path.as_posix())),
            key="config_path_ui",
            help="Profile/model/tool configuration YAML path.",
        ).strip()
        config_path = Path(config_path_text or default_config_path).expanduser()
        try:
            cfg = load_agent_config(config_path)
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            st.stop()
        subcfg = _resolve_subagent_cfg(cfg)
        agent_profile_ids = [p.id for p in cfg.agent_profiles] or [cfg.default_agent_profile]
        current_agent_profile = str(st.session_state.get("agent_profile_id_ui", cfg.default_agent_profile))
        if current_agent_profile not in agent_profile_ids:
            current_agent_profile = agent_profile_ids[0]
            st.session_state["agent_profile_id_ui"] = current_agent_profile
        agent_profile_id = st.selectbox(
            "Agent profile",
            options=agent_profile_ids,
            index=agent_profile_ids.index(current_agent_profile),
            key="agent_profile_id_ui",
        )
        selected_agent_profile = cfg.get_agent_profile(agent_profile_id)
        try:
            resolved_runtime_default = resolve_runtime_engine(
                cfg=cfg,
                profile_runtime_engine=getattr(selected_agent_profile, "runtime_engine", None),
                explicit_runtime_engine=None,
            )
        except Exception:
            resolved_runtime_default = "langgraph"
        runtime_options = [e for e in list(cfg.runtime.allowed_engines or []) if e in {"langgraph", "google_adk"}]
        if not runtime_options:
            runtime_options = ["langgraph", "google_adk"]
        current_runtime_engine = str(st.session_state.get("runtime_engine_ui", resolved_runtime_default))
        if current_runtime_engine not in runtime_options:
            current_runtime_engine = resolved_runtime_default if resolved_runtime_default in runtime_options else runtime_options[0]
            st.session_state["runtime_engine_ui"] = current_runtime_engine
        runtime_engine = st.selectbox(
            "Runtime engine",
            options=runtime_options,
            index=runtime_options.index(current_runtime_engine),
            key="runtime_engine_ui",
            help="Switch between LangGraph and Google ADK runtime adapters.",
        )

        card_ids = [c.id for c in cfg.model_cards]
        default_model_id = selected_agent_profile.model_card_id or cfg.default_model_card
        default_idx = card_ids.index(default_model_id) if default_model_id in card_ids else 0
        model_card_id = st.selectbox("Model card", options=card_ids, index=default_idx)
        profile_ids = [p.id for p in cfg.policy_profiles] or [cfg.default_policy_profile]
        default_policy_id = selected_agent_profile.policy_profile_id or cfg.default_policy_profile
        current_profile = str(st.session_state.get("policy_profile_id_ui", default_policy_id))
        if current_profile not in profile_ids:
            current_profile = profile_ids[0]
            st.session_state["policy_profile_id_ui"] = current_profile
        policy_profile_id = st.selectbox(
            "Policy profile",
            options=profile_ids,
            index=profile_ids.index(current_profile),
            key="policy_profile_id_ui",
            help="Controls prompt budgets, summarization thresholds, and tool-log truncation.",
        )

        selected = cfg.get_model_card(model_card_id)
        model_name = st.text_input("Model name override", value=selected.model_name)
        thinking_budget = st.number_input(
            "Thinking budget override",
            min_value=0,
            step=128,
            value=int(selected.thinking_budget or 0),
            help="Set 0 to disable extra reasoning budget for supported models.",
        )
        st.subheader("Available Tools")
        full_tool_catalog = build_tool_catalog(cfg, DEFAULT_TOOLS)
        profile_tool_catalog = resolve_tools_for_profile(full_tool_catalog, selected_agent_profile)
        catalog = tool_catalog(profile_tool_catalog)
        enabled_names = _render_tool_selector(catalog)
        st.caption(f"Enabled: {len(enabled_names)} / {len(catalog)}")
        st.subheader("Sub-agent Controls")
        subagent_enabled_ui = st.checkbox(
            "Enable sub-agents",
            value=bool(st.session_state.get("subagent_enabled_ui", subcfg["enabled"])),
            key="subagent_enabled_ui",
        )
        st.number_input(
            "Sub-agent max workers",
            min_value=1,
            max_value=max(1, int(subcfg["max_workers_limit"])),
            value=int(st.session_state.get("subagent_max_workers_ui", subcfg["max_workers_default"])),
            step=1,
            key="subagent_max_workers_ui",
        )
        st.number_input(
            "Sub-agent max turns",
            min_value=1,
            max_value=max(1, int(subcfg["max_worker_turns_limit"])),
            value=int(st.session_state.get("subagent_max_worker_turns_ui", subcfg["max_worker_turns_default"])),
            step=1,
            key="subagent_max_worker_turns_ui",
        )
        st.number_input(
            "Sub-agent max wall time (s)",
            min_value=1.0,
            max_value=max(1.0, float(subcfg["max_wall_time_s_limit"])),
            value=float(st.session_state.get("subagent_max_wall_time_ui", subcfg["max_wall_time_s_default"])),
            step=5.0,
            key="subagent_max_wall_time_ui",
        )
        st.button(
            "Apply Deep Research Preset",
            help="Sets deep_research policy profile and higher sub-agent budgets for long synthesis tasks.",
            on_click=_apply_deep_research_preset,
            kwargs={
                "profile_ids": profile_ids,
                "available_tool_names": [entry["name"] for entry in catalog],
                "subcfg": subcfg,
            },
        )
        st.caption("Set GOOGLE_API_KEY in .env or environment before running.")
        if st.button("Refresh Runtime Cache"):
            _build_runtime.clear()
            _reset_session()
            st.rerun()
        if st.button("Reset Session"):
            _reset_session()
            st.rerun()
        st.session_state["enabled_tool_names_ui"] = list(sorted(enabled_names))

    resolved_skill_roots = _resolve_skills_roots(list(selected_agent_profile.skills.roots or [".skills"]))
    skills_sig = _skills_signature(resolved_skill_roots)

    try:
        runtime = _build_runtime(
            config_path=config_path.as_posix(),
            agent_profile_id=agent_profile_id,
            runtime_engine=runtime_engine,
            model_card_id=model_card_id,
            policy_profile_id=policy_profile_id,
            skills_signature=skills_sig,
            model_name_override=model_name,
            thinking_budget_override=int(thinking_budget),
            enabled_tool_names=tuple(sorted(enabled_names)),
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    _init_session(runtime)

    st.sidebar.caption(
        "Resolved: "
        f"{runtime['model_card'].id} | "
        f"{runtime['model_card'].provider} | "
        f"thinking_budget={runtime['model_card'].thinking_budget}"
    )
    st.sidebar.caption(f"Agent profile: {runtime['agent_profile_id']}")
    st.sidebar.caption(f"Runtime engine: {runtime.get('runtime_engine', 'langgraph')}")
    if runtime.get("agent_profile_description"):
        st.sidebar.caption(str(runtime["agent_profile_description"]))
    st.sidebar.caption(f"Policy profile: {runtime['policy_profile_id']}")
    if runtime.get("policy_profile_description"):
        st.sidebar.caption(str(runtime["policy_profile_description"]))
    st.sidebar.caption("Enabled tools: " + ", ".join(runtime["enabled_tool_names"]))
    st.sidebar.caption(
        "Sub-agents: "
        + ("enabled" if st.session_state.get("subagent_enabled_ui", runtime["subagent_defaults"]["enabled"]) else "disabled")
    )
    st.sidebar.caption("Skills roots: " + ", ".join(runtime.get("skills_roots_resolved", [])))
    _render_loaded_skills_sidebar(runtime)
    _render_artifacts_sidebar()

    user_tab, debug_tab, artifacts_tab = st.tabs(["User View", "Debug View", "Artifacts"])
    with user_tab:
        _render_user_view(runtime)

    with debug_tab:
        _render_debug_view()

    with artifacts_tab:
        _render_artifacts_view()


if __name__ == "__main__":
    main()
