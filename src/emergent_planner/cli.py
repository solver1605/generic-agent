"""
CLI runtime for Emergent Planner with Streamlit-equivalent controls.

Examples:
  generic-agent
  generic-agent --policy-profile deep_research --deep-research-preset
  generic-agent --tools "load_skill,search_web,spawn_subagents,verify_with_user"
  generic-agent --prompt "Research PEFT methods for 8B models" --non-interactive
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import replace
from queue import Empty, Queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from emergent_planner import (
    DEFAULT_TOOLS,
    build_app,
    build_prompt_lib_for_profile,
    discover_skills_in_roots,
)
from emergent_planner.config import (
    build_llm_from_model_card,
    load_agent_config,
    resolve_runtime_policies,
)
from emergent_planner.data_models import (
    DataModelValidationError,
    build_data_model_catalog,
    build_record,
    ensure_runtime_data_model_state,
    load_persisted_records,
    normalize_user_id,
    persist_record,
    resolve_data_models_for_profile,
    validate_instance,
)
from emergent_planner.skills import find_project_root
from emergent_planner.tool_loader import build_tool_catalog, resolve_tools_for_profile
from emergent_planner.tool_registry import tool_catalog, tool_name
from emergent_planner.utils import extract_tool_calls, get_history_from_state, normalize_content

load_dotenv()


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


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
    # Deduplicate while preserving order.
    deduped: List[Path] = []
    seen = set()
    for p in out:
        k = p.as_posix()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(p)
    return deduped


def _artifact_root() -> Path:
    return (find_project_root(Path.cwd()) / "artifacts").resolve()


def _list_artifacts(*, session_started_at: float, session_only: bool = True) -> List[Path]:
    root = _artifact_root()
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    if session_only:
        files = [p for p in files if p.stat().st_mtime >= (session_started_at - 1.0)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _resolve_data_model_registry(cfg, agent_profile) -> Dict[str, Any]:
    models = resolve_data_models_for_profile(build_data_model_catalog(cfg), agent_profile)
    meta: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[str, Any] = {}
    for m in models:
        by_id[str(m.id)] = m
        schema = m.schema_cls.model_json_schema()
        required = [str(x) for x in list(schema.get("required", []) or []) if str(x).strip()]
        props = dict(schema.get("properties", {}) or {})
        fields = []
        for k, spec in props.items():
            if not isinstance(spec, dict):
                continue
            fields.append(
                {
                    "name": str(k),
                    "type": str(spec.get("type", "string") or "string"),
                    "required": str(k) in set(required),
                    "description": str(spec.get("description", "") or "").strip(),
                }
            )
        field_names = [f["name"] for f in fields]
        context_fields = [f for f in (m.context_fields or field_names) if f in set(field_names)]
        meta[str(m.id)] = {
            "id": str(m.id),
            "description": str(m.description),
            "fields": fields,
            "required_fields": required,
            "context_fields": context_fields,
        }
    return {"models": by_id, "meta": meta, "ids": sorted(meta.keys())}


def _upsert_data_model_in_state(
    state: Dict[str, Any],
    *,
    model_id: str,
    payload: Dict[str, Any],
    merge: bool = True,
) -> tuple[Dict[str, Any], str]:
    rt = ensure_runtime_data_model_state(dict(state.get("runtime", {}) or {}))
    cfg_path = Path(str(rt.get("config_path", "agent_config.yaml"))).expanduser()
    cfg_dir_raw = str(rt.get("config_dir", "") or "").strip()
    if not cfg_path.is_absolute() and cfg_dir_raw:
        cfg_path = (Path(cfg_dir_raw).expanduser() / cfg_path).resolve()
    else:
        cfg_path = cfg_path.resolve()
    cfg = load_agent_config(cfg_path)
    profile = cfg.get_agent_profile(str(rt.get("agent_profile_id", cfg.default_agent_profile)))
    reg = _resolve_data_model_registry(cfg, profile)
    model_map = dict(reg["models"])
    meta_map = dict(reg["meta"])
    if model_id not in model_map:
        raise ValueError(f"Unknown model '{model_id}'. Available: {', '.join(sorted(model_map.keys()))}")

    values = dict(rt.get("data_model_values", {}) or {})
    existing = dict(values.get(model_id, {}) or {})
    incoming = dict(payload or {})
    candidate = {**existing, **incoming} if merge else incoming
    validated = validate_instance(model_map[model_id], candidate, strict=True)
    record = build_record(model_id, validated, meta_map.get(model_id, {}))
    user_id = normalize_user_id(rt.get("active_user_id", "default"))
    path = persist_record(record, user_id=user_id)

    values[model_id] = dict(record.data)
    updated = dict(rt.get("data_model_last_updated", {}) or {})
    updated[model_id] = record.updated_at
    rt["data_model_values"] = values
    rt["data_model_last_updated"] = updated
    rt["data_model_meta"] = meta_map
    new_state = dict(state)
    new_state["runtime"] = rt
    return new_state, path.as_posix()


def _normalize_interrupt_payload(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        base = {
            "type": raw.get("type") or raw.get("kind") or "confirm",
            "reason": raw.get("reason", ""),
            "question": raw.get("question", "Please provide input to continue."),
            "choices": raw.get("choices"),
            "context": raw.get("context"),
            "default": raw.get("default"),
        }
        for k, v in raw.items():
            if k not in base:
                base[k] = v
        return base
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return _normalize_interrupt_payload(raw[0])
    value = getattr(raw, "value", None)
    if value is not None:
        return _normalize_interrupt_payload(value)
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


def _resolve_subagent_cfg(cfg) -> Dict[str, Any]:
    scfg = getattr(cfg, "subagents", None)
    if scfg is None:
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


def _apply_deep_research_preset(
    *,
    profile_ids: List[str],
    available_tool_names: List[str],
    selected_tools: List[str],
    subcfg: Dict[str, Any],
) -> tuple[str, List[str], bool, int, int, float]:
    profile_id = "deep_research" if "deep_research" in profile_ids else (profile_ids[0] if profile_ids else "balanced")
    out_tools = list(selected_tools)
    needed = ["search_web", "spawn_subagents", "load_skill", "verify_with_user"]
    for tname in needed:
        if tname in available_tool_names and tname not in out_tools:
            out_tools.append(tname)
    max_workers = min(6, int(subcfg["max_workers_limit"]))
    max_turns = min(14, int(subcfg["max_worker_turns_limit"]))
    max_wall = min(180.0, float(subcfg["max_wall_time_s_limit"]))
    return profile_id, out_tools, True, max_workers, max_turns, max_wall


def _prompt_interrupt(payload: Dict[str, Any]) -> Any:
    kind = payload.get("type", "confirm")
    reason = str(payload.get("reason", ""))
    question = str(payload.get("question", "Please provide input to continue."))
    context = str(payload.get("context", "") or "").strip()
    choices = list(payload.get("choices") or [])
    default = payload.get("default")
    model_id = str(payload.get("model_id", "") or "")
    field_schema = list(payload.get("field_schema", []) or [])

    print("\n" + "=" * 72)
    print("PAUSED FOR INPUT")
    if reason:
        print(f"Reason: {reason}")
    print(question)
    if context:
        print("\nContext:")
        print(context)

    if kind == "model_form" and field_schema:
        print(f"\nModel form: {model_id}")
        out: Dict[str, Any] = {}
        for field in field_schema:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name", "") or "").strip()
            if not name:
                continue
            required = bool(field.get("required", False))
            ftype = str(field.get("type", "string") or "string")
            desc = str(field.get("description", "") or "").strip()
            if desc:
                print(f"  - {name}: {desc}")
            raw = input(f"{name}{' *' if required else ''}: ").strip()
            if not raw and required:
                raw = input(f"{name} is required. Enter value: ").strip()
            if not raw and not required:
                out[name] = None
                continue
            try:
                if ftype == "integer":
                    out[name] = int(raw)
                elif ftype == "number":
                    out[name] = float(raw)
                elif ftype == "boolean":
                    out[name] = raw.lower() in {"1", "true", "yes", "y"}
                elif ftype in {"array", "object"}:
                    out[name] = json.loads(raw)
                else:
                    out[name] = raw
            except Exception:
                out[name] = raw
        return {"model_id": model_id, "values": out}

    if kind == "pick_one" and choices:
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        raw = input("Select number or type answer: ").strip()
        try:
            idx = int(raw) - 1
            return choices[idx]
        except (ValueError, IndexError):
            return raw or default

    if kind == "pick_many" and choices:
        print("Choose one or more values separated by commas.")
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        raw = input("Selections: ").strip()
        if not raw:
            return []
        out = []
        for part in raw.split(","):
            v = part.strip()
            if not v:
                continue
            try:
                idx = int(v) - 1
                if 0 <= idx < len(choices):
                    out.append(choices[idx])
            except ValueError:
                out.append(v)
        return out

    if kind == "confirm":
        d = str(default or "yes")
        raw = input(f"Confirm [yes/no] (default={d}): ").strip().lower()
        return raw or d

    raw = input(f"Answer{f' (default={default})' if default else ''}: ").strip()
    return raw or default


def _print_new_messages(messages: List[Any], *, show_tools: bool) -> None:
    for m in messages:
        if isinstance(m, HumanMessage):
            continue
        if isinstance(m, AIMessage):
            txt = normalize_content(getattr(m, "content", ""))
            calls = extract_tool_calls(m)
            if txt.strip():
                print("\nAssistant:")
                print(txt)
            elif calls:
                print("\nAssistant: (tool-call-only response)")
            if calls:
                print("Tool calls:")
                for c in calls:
                    print(f"- {c.get('name')} id={c.get('id')}")
            continue
        if isinstance(m, ToolMessage) and show_tools:
            print("\nTool output:")
            print(normalize_content(getattr(m, "content", "")))


def _run_graph(
    *,
    app,
    input_obj: Any,
    state: Dict[str, Any],
    thread_id: str,
    prev_hist_len: int,
    debug: bool,
    run_label: str = "agent step",
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], int, List[Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    final_state = state
    pending_interrupt = None
    q: Queue[tuple[str, Any]] = Queue()
    started = time.time()
    last_heartbeat = started

    def _worker() -> None:
        try:
            for full_state in app.stream(input_obj, config=config, stream_mode="values"):
                q.put(("state", full_state))
                # Stop this stream immediately on interrupt; resume happens in a new run.
                if _extract_interrupt_payload(full_state) is not None:
                    q.put(("interrupted", None))
                    return
            q.put(("done", None))
        except Exception as e:  # pragma: no cover - surfaced in main thread
            q.put(("error", e))

    worker = Thread(target=_worker, daemon=True)
    worker.start()

    print(f"[run] {run_label} started")
    while True:
        try:
            kind, payload_obj = q.get(timeout=1.0)
        except Empty:
            now = time.time()
            if now - last_heartbeat >= 10.0:
                print(f"[run] {run_label} still running... {int(now - started)}s elapsed")
                last_heartbeat = now
            continue

        if kind == "error":
            raise payload_obj
        if kind == "interrupted":
            break
        if kind == "done":
            break
        if kind != "state":
            continue

        full_state = payload_obj
        final_state = full_state
        if debug:
            rt = dict(full_state.get("runtime", {}) or {})
            hist_len = len(get_history_from_state(full_state))
            print(
                f"[debug] turn={rt.get('turn_index', 0)} "
                f"history={hist_len} "
                f"after_tool={bool(rt.get('after_tool', False))}"
            )
        payload = _extract_interrupt_payload(full_state)
        if payload is not None:
            pending_interrupt = payload
            break

    elapsed = int(time.time() - started)
    print(f"[run] {run_label} completed in {elapsed}s")

    hist = get_history_from_state(final_state)
    new_messages = hist[prev_hist_len:]
    return final_state, pending_interrupt, len(hist), new_messages


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emergent Planner CLI")
    parser.add_argument("--config", default="agent_config.yaml", help="Path to config YAML.")
    parser.add_argument("--agent-profile", default="", help="Agent profile id override.")
    parser.add_argument("--model-card", default="", help="Model card id override.")
    parser.add_argument("--model-name", default="", help="Model name override.")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Thinking budget override.")
    parser.add_argument("--policy-profile", default="", help="Policy profile id override.")
    parser.add_argument("--skills-root", default="", help="Optional override for skills root path.")
    parser.add_argument("--tools", default="", help="Comma-separated tool names to enable.")
    parser.add_argument("--disable-tools", default="", help="Comma-separated tool names to disable.")
    parser.add_argument("--show-tools", action="store_true", help="Print tool outputs in chat.")
    parser.add_argument("--debug", action="store_true", help="Print per-step debug telemetry.")
    parser.add_argument("--deep-research-preset", action="store_true", help="Apply deep-research runtime preset.")
    parser.add_argument("--subagent-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subagent-max-workers", type=int, default=None)
    parser.add_argument("--subagent-max-turns", type=int, default=None)
    parser.add_argument("--subagent-max-wall-time", type=float, default=None)
    parser.add_argument("--prompt", default="", help="Optional initial user prompt.")
    parser.add_argument("--non-interactive", action="store_true", help="Run one prompt and exit.")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_agent_config(config_path)
    config_dir = config_path.resolve().parent
    subcfg = _resolve_subagent_cfg(cfg)
    agent_profile_id = args.agent_profile.strip() or cfg.default_agent_profile
    agent_profile = cfg.get_agent_profile(agent_profile_id)

    # Model selection.
    model_card_id = args.model_card.strip() or agent_profile.model_card_id or cfg.default_model_card
    model_card = cfg.get_model_card(model_card_id)
    if args.model_name.strip():
        model_card = replace(model_card, model_name=args.model_name.strip())
    if args.thinking_budget is not None:
        model_card = replace(model_card, thinking_budget=int(args.thinking_budget))

    # Policy profile.
    profile_ids = [p.id for p in cfg.policy_profiles]
    policy_profile_id = (
        args.policy_profile.strip()
        or agent_profile.policy_profile_id
        or cfg.default_policy_profile
    )
    if policy_profile_id not in profile_ids:
        policy_profile_id = cfg.default_policy_profile

    # Tool selection.
    full_tool_catalog = build_tool_catalog(cfg, DEFAULT_TOOLS)
    profile_tool_catalog = resolve_tools_for_profile(full_tool_catalog, agent_profile)
    catalog = tool_catalog(profile_tool_catalog)
    available_tool_names = [entry["name"] for entry in catalog]
    selected_tool_names = _split_csv(args.tools) if args.tools.strip() else list(available_tool_names)
    disabled_names = set(_split_csv(args.disable_tools))

    # Validate selected tool names.
    unknown = sorted(set(selected_tool_names) - set(available_tool_names))
    if unknown:
        raise ValueError(f"Unknown tool(s): {', '.join(unknown)}. Available: {', '.join(sorted(available_tool_names))}")

    # Sub-agent controls.
    subagent_enabled = bool(subcfg["enabled"]) if args.subagent_enabled is None else bool(args.subagent_enabled)
    subagent_max_workers = int(args.subagent_max_workers or subcfg["max_workers_default"])
    subagent_max_turns = int(args.subagent_max_turns or subcfg["max_worker_turns_default"])
    subagent_max_wall_time = float(args.subagent_max_wall_time or subcfg["max_wall_time_s_default"])

    # Optional one-click preset parity with Streamlit.
    if args.deep_research_preset:
        (
            policy_profile_id,
            selected_tool_names,
            subagent_enabled,
            subagent_max_workers,
            subagent_max_turns,
            subagent_max_wall_time,
        ) = _apply_deep_research_preset(
            profile_ids=profile_ids,
            available_tool_names=available_tool_names,
            selected_tools=selected_tool_names,
            subcfg=subcfg,
        )

    tools = resolve_tools_for_profile(
        full_tool_catalog,
        agent_profile,
        extra_allow=selected_tool_names,
        extra_deny=sorted(disabled_names),
    )

    # Clamp to configured sub-agent limits.
    subagent_max_workers = max(1, min(subagent_max_workers, int(subcfg["max_workers_limit"])))
    subagent_max_turns = max(1, min(subagent_max_turns, int(subcfg["max_worker_turns_limit"])))
    subagent_max_wall_time = max(1.0, min(subagent_max_wall_time, float(subcfg["max_wall_time_s_limit"])))

    # Build runtime.
    base_llm = build_llm_from_model_card(model_card, env=os.environ)
    llm_with_tools = base_llm.bind_tools(tools)

    budget_policy, tool_policy, summary_policy, resolved_profile_id = resolve_runtime_policies(cfg, policy_profile_id)
    profile_skill_roots = [args.skills_root.strip()] if args.skills_root.strip() else list(agent_profile.skills.roots or [".skills"])
    resolved_skill_roots = _resolve_skills_roots(profile_skill_roots)
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
    skills = discover_skills_in_roots(resolved_skill_roots, strict_scope=True)
    if allowlist_norm:
        allowed = set(allowlist_norm)
        skills = [s for s in skills if _normalize_skill_key(getattr(s, "name", "")) in allowed]
    if denylist_norm:
        denied = set(denylist_norm)
        skills = [s for s in skills if _normalize_skill_key(getattr(s, "name", "")) not in denied]
    data_models_reg = _resolve_data_model_registry(cfg, agent_profile)
    active_user_id = "default"
    persisted_values, persisted_updated = load_persisted_records(
        list(data_models_reg["ids"]),
        user_id=active_user_id,
    )

    prompt_lib = build_prompt_lib_for_profile(cfg, agent_profile, config_dir=config_dir)
    app = build_app(
        llm=llm_with_tools,
        prompt_lib=prompt_lib,
        skills_root=(resolved_skill_roots[0] if resolved_skill_roots else Path(".skills")),
        budget_policy=budget_policy,
        tool_log_policy=tool_policy,
        summary_policy=summary_policy,
        tools=tools,
    )

    run_id = str(uuid.uuid4())
    state: Dict[str, Any] = {
        "history": [],
        "memory": {},
        "runtime": {
            "run_id": run_id,
            "turn_index": 0,
            "config_path": config_path.as_posix(),
            "config_dir": config_dir.as_posix(),
            "model_card_id": model_card.id,
            "model_name": model_card.model_name,
            "thinking_budget": model_card.thinking_budget,
            "agent_profile_id": agent_profile.id,
            "enabled_tool_names": [tool_name(t) for t in tools],
            "skills_roots_resolved": [p.as_posix() for p in resolved_skill_roots],
            "skills_allowlist_norm": allowlist_norm,
            "skills_denylist_norm": denylist_norm,
            "active_user_id": active_user_id,
            "data_model_values": persisted_values,
            "data_model_meta": dict(data_models_reg["meta"]),
            "data_model_last_updated": persisted_updated,
            "subagent_enabled": subagent_enabled,
            "subagent_max_workers": subagent_max_workers,
            "subagent_max_worker_turns": subagent_max_turns,
            "subagent_max_wall_time_s": subagent_max_wall_time,
            "policy_profile_id": resolved_profile_id,
        },
        "skills": skills,
    }
    session_started_at = time.time()
    thread_id = run_id
    pending_interrupt: Optional[Dict[str, Any]] = None
    prev_hist_len = 0

    print(
        f"Run: {run_id}\n"
        f"Agent profile: {agent_profile.id}\n"
        f"Model: {model_card.id} ({model_card.model_name})\n"
        f"Policy profile: {resolved_profile_id}\n"
        f"Skills roots: {', '.join([p.as_posix() for p in resolved_skill_roots])}\n"
        f"Skills allowlist: {', '.join(allowlist_norm) if allowlist_norm else '(none)'}\n"
        f"Skills denylist: {', '.join(denylist_norm) if denylist_norm else '(none)'}\n"
        f"Skills discovered: {len(skills)}\n"
        f"Data models: {', '.join(data_models_reg['ids']) if data_models_reg['ids'] else '(none)'}\n"
        f"Enabled tools: {', '.join([tool_name(t) for t in tools])}\n"
        f"Sub-agents: enabled={subagent_enabled}, workers={subagent_max_workers}, "
        f"turns={subagent_max_turns}, wall_time_s={subagent_max_wall_time}\n"
    )
    print("Commands: /help, /status, /tools, /skills, /profile, /profile set key=value..., /model <id>, /model set <id> <json>, /artifacts, /artifacts all, /quit")

    # Optional initial turn.
    initial_prompt = args.prompt.strip()
    if initial_prompt:
        state["history"] = [HumanMessage(content=initial_prompt)]
        state, pending_interrupt, prev_hist_len, new_messages = _run_graph(
            app=app,
            input_obj=state,
            state=state,
            thread_id=thread_id,
            prev_hist_len=0,
            debug=bool(args.debug),
            run_label="initial prompt",
        )
        _print_new_messages(new_messages, show_tools=bool(args.show_tools))
        if args.non_interactive and pending_interrupt is None:
            return

    # Interactive loop.
    while True:
        if pending_interrupt is not None:
            answer = _prompt_interrupt(pending_interrupt)
            state, pending_interrupt, prev_hist_len, new_messages = _run_graph(
                app=app,
                input_obj=Command(resume=answer),
                state=state,
                thread_id=thread_id,
                prev_hist_len=prev_hist_len,
                debug=bool(args.debug),
                run_label="resume from interrupt",
            )
            _print_new_messages(new_messages, show_tools=bool(args.show_tools))
            if args.non_interactive:
                break
            continue

        raw = input("\nYou> ").strip()
        if not raw:
            continue
        if raw in {"/quit", "/exit"}:
            break
        if raw == "/help":
            print("Commands: /help, /status, /tools, /skills, /profile, /profile set key=value..., /model <id>, /model set <id> <json>, /artifacts, /artifacts all, /quit")
            continue
        if raw == "/status":
            rt = dict(state.get("runtime", {}) or {})
            print(
                f"turn={rt.get('turn_index', 0)} "
                f"policy={rt.get('policy_profile_id')} "
                f"subagents_enabled={rt.get('subagent_enabled')} "
                f"workers={rt.get('subagent_max_workers')} "
                f"turns={rt.get('subagent_max_worker_turns')} "
                f"wall_time={rt.get('subagent_max_wall_time_s')}"
            )
            continue
        if raw == "/tools":
            rt = dict(state.get("runtime", {}) or {})
            print("Enabled tools: " + ", ".join(rt.get("enabled_tool_names", [])))
            continue
        if raw == "/skills":
            names = [getattr(s, "name", "unknown") for s in (state.get("skills", []) or [])]
            print("Skills: " + (", ".join(names) if names else "(none)"))
            continue
        if raw == "/profile":
            rt = ensure_runtime_data_model_state(dict(state.get("runtime", {}) or {}))
            prof = dict((rt.get("data_model_values", {}) or {}).get("user_profile", {}) or {})
            print("User profile: " + (str(prof) if prof else "{}"))
            model_meta = dict(rt.get("data_model_meta", {}) or {})
            vals = dict(rt.get("data_model_values", {}) or {})
            for mid in sorted(model_meta.keys()):
                req = list((model_meta.get(mid, {}) or {}).get("required_fields", []) or [])
                data = dict(vals.get(mid, {}) or {})
                missing = []
                for f in req:
                    v = data.get(f)
                    if f not in data or v is None or (isinstance(v, str) and not v.strip()):
                        missing.append(f)
                status = "complete" if not missing else f"partial missing={missing}"
                print(f"- {mid}: {status}")
            continue
        if raw.startswith("/profile set "):
            args_txt = raw[len("/profile set "):].strip()
            payload: Dict[str, Any] = {}
            for part in args_txt.split():
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                key = k.strip()
                if key in {"name", "birth_details", "place"}:
                    payload[key] = v.strip()
            if not payload:
                print("No valid fields provided. Use: /profile set name=... birth_details=... place=...")
                continue
            try:
                state, path = _upsert_data_model_in_state(
                    state,
                    model_id="user_profile",
                    payload=payload,
                    merge=True,
                )
                print(f"Saved user_profile -> {path}")
            except DataModelValidationError as e:
                print(f"Validation error: {e}")
            except Exception as e:
                print(f"Failed to save user_profile: {e}")
            continue
        if raw.startswith("/model "):
            parts = raw.split(maxsplit=3)
            if len(parts) == 2:
                model_id = parts[1].strip()
                rt = ensure_runtime_data_model_state(dict(state.get("runtime", {}) or {}))
                meta = dict((rt.get("data_model_meta", {}) or {}).get(model_id, {}) or {})
                data = dict((rt.get("data_model_values", {}) or {}).get(model_id, {}) or {})
                if not meta:
                    print(f"Unknown model '{model_id}'.")
                else:
                    print(f"Model {model_id}:")
                    print("meta=" + str(meta))
                    print("data=" + str(data))
                continue
            if len(parts) >= 4 and parts[1] == "set":
                model_id = parts[2].strip()
                raw_json = parts[3].strip()
                try:
                    payload = json.loads(raw_json) if raw_json else {}
                    if not isinstance(payload, dict):
                        raise ValueError("Payload must be JSON object.")
                    state, path = _upsert_data_model_in_state(
                        state,
                        model_id=model_id,
                        payload=payload,
                        merge=False,
                    )
                    print(f"Saved {model_id} -> {path}")
                except Exception as e:
                    print(f"Failed to save model: {e}")
                continue
        if raw.startswith("/artifacts"):
            parts = raw.split()
            session_only = not (len(parts) > 1 and parts[1].lower() == "all")
            files = _list_artifacts(session_started_at=session_started_at, session_only=session_only)
            if not files:
                print("No artifacts found.")
            else:
                print(f"Artifacts ({'session' if session_only else 'all'}):")
                for p in files[:40]:
                    print(f"- {p.as_posix()}")
            continue

        hist = list(state.get("history", []) or [])
        hist.append(HumanMessage(content=raw))
        state = {**state, "history": hist}
        state, pending_interrupt, prev_hist_len, new_messages = _run_graph(
            app=app,
            input_obj=state,
            state=state,
            thread_id=thread_id,
            prev_hist_len=prev_hist_len,
            debug=bool(args.debug),
            run_label="user turn",
        )
        _print_new_messages(new_messages, show_tools=bool(args.show_tools))


if __name__ == "__main__":
    main()
