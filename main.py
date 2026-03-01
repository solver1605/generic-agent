"""
CLI runtime for Emergent Planner with Streamlit-equivalent controls.

Examples:
  python main.py
  python main.py --policy-profile deep_research --deep-research-preset
  python main.py --tools "load_skill,search_web,spawn_subagents,verify_with_user"
  python main.py --prompt "Research PEFT methods for 8B models" --non-interactive
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from src.emergent_planner import (
    DEFAULT_TOOLS,
    build_app,
    discover_skills,
    make_default_prompt_lib,
)
from src.emergent_planner.config import (
    build_llm_from_model_card,
    load_agent_config,
    resolve_runtime_policies,
)
from src.emergent_planner.skills import find_project_root
from src.emergent_planner.tool_registry import select_tools, tool_catalog, tool_name
from src.emergent_planner.utils import extract_tool_calls, get_history_from_state, normalize_content

load_dotenv()


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def _resolve_skills_root(skills_root: str) -> Path:
    p = Path(skills_root).expanduser()
    if p.is_absolute():
        return p
    cwd_resolved = (Path.cwd() / p).resolve()
    if cwd_resolved.exists():
        return cwd_resolved
    project_root = find_project_root(Path.cwd())
    return (project_root / p).resolve()


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


def _normalize_interrupt_payload(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {
            "type": raw.get("type") or raw.get("kind") or "confirm",
            "reason": raw.get("reason", ""),
            "question": raw.get("question", "Please provide input to continue."),
            "choices": raw.get("choices"),
            "context": raw.get("context"),
            "default": raw.get("default"),
        }
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

    print("\n" + "=" * 72)
    print("PAUSED FOR INPUT")
    if reason:
        print(f"Reason: {reason}")
    print(question)
    if context:
        print("\nContext:")
        print(context)

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
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], int, List[Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    final_state = state
    pending_interrupt = None

    for full_state in app.stream(input_obj, config=config, stream_mode="values"):
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

    hist = get_history_from_state(final_state)
    new_messages = hist[prev_hist_len:]
    return final_state, pending_interrupt, len(hist), new_messages


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emergent Planner CLI")
    parser.add_argument("--config", default="agent_config.yaml", help="Path to config YAML.")
    parser.add_argument("--model-card", default="", help="Model card id override.")
    parser.add_argument("--model-name", default="", help="Model name override.")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Thinking budget override.")
    parser.add_argument("--policy-profile", default="", help="Policy profile id override.")
    parser.add_argument("--skills-root", default=".skills", help="Skills root path.")
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
    cfg = load_agent_config(Path(args.config))
    subcfg = _resolve_subagent_cfg(cfg)

    # Model selection.
    model_card_id = args.model_card.strip() or cfg.default_model_card
    model_card = cfg.get_model_card(model_card_id)
    if args.model_name.strip():
        model_card = replace(model_card, model_name=args.model_name.strip())
    if args.thinking_budget is not None:
        model_card = replace(model_card, thinking_budget=int(args.thinking_budget))

    # Policy profile.
    profile_ids = [p.id for p in cfg.policy_profiles]
    policy_profile_id = args.policy_profile.strip() or cfg.default_policy_profile
    if policy_profile_id not in profile_ids:
        policy_profile_id = cfg.default_policy_profile

    # Tool selection.
    catalog = tool_catalog(DEFAULT_TOOLS)
    available_tool_names = [entry["name"] for entry in catalog]
    selected_tool_names = _split_csv(args.tools) if args.tools.strip() else list(available_tool_names)
    disabled_names = set(_split_csv(args.disable_tools))
    selected_tool_names = [t for t in selected_tool_names if t not in disabled_names]

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

    # Clamp to configured sub-agent limits.
    subagent_max_workers = max(1, min(subagent_max_workers, int(subcfg["max_workers_limit"])))
    subagent_max_turns = max(1, min(subagent_max_turns, int(subcfg["max_worker_turns_limit"])))
    subagent_max_wall_time = max(1.0, min(subagent_max_wall_time, float(subcfg["max_wall_time_s_limit"])))

    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env or environment.")

    # Build runtime.
    base_llm = build_llm_from_model_card(model_card, google_api_key=google_api_key)
    tools = select_tools(DEFAULT_TOOLS, selected_tool_names)
    if not tools:
        raise ValueError("No tools are enabled.")
    llm_with_tools = base_llm.bind_tools(tools)

    budget_policy, tool_policy, summary_policy, resolved_profile_id = resolve_runtime_policies(cfg, policy_profile_id)
    skills_root = _resolve_skills_root(args.skills_root)
    skills = discover_skills(skills_root)
    app = build_app(
        llm=llm_with_tools,
        prompt_lib=make_default_prompt_lib(),
        skills_root=skills_root,
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
            "model_card_id": model_card.id,
            "model_name": model_card.model_name,
            "thinking_budget": model_card.thinking_budget,
            "enabled_tool_names": [tool_name(t) for t in tools],
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
        f"Model: {model_card.id} ({model_card.model_name})\n"
        f"Policy profile: {resolved_profile_id}\n"
        f"Skills root: {skills_root}\n"
        f"Skills discovered: {len(skills)}\n"
        f"Enabled tools: {', '.join([tool_name(t) for t in tools])}\n"
        f"Sub-agents: enabled={subagent_enabled}, workers={subagent_max_workers}, "
        f"turns={subagent_max_turns}, wall_time_s={subagent_max_wall_time}\n"
    )
    print("Commands: /help, /status, /tools, /skills, /artifacts, /artifacts all, /quit")

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
            print("Commands: /help, /status, /tools, /skills, /artifacts, /artifacts all, /quit")
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
        )
        _print_new_messages(new_messages, show_tools=bool(args.show_tools))


if __name__ == "__main__":
    main()
