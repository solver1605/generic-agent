"""Runtime factory for LangGraph and Google ADK engines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from ..config import AgentConfig
from ..graph import build_app as build_langgraph_graph
from ..models import PromptLibrary
from ..policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy
from .adk_engine import ADKRuntimeEngine
from .engine import AgentRuntimeEngine
from .langgraph_engine import LangGraphRuntimeEngine


def resolve_runtime_engine(
    *,
    cfg: Optional[AgentConfig],
    profile_runtime_engine: Optional[str] = None,
    explicit_runtime_engine: Optional[str] = None,
) -> str:
    selected = (explicit_runtime_engine or "").strip() or (profile_runtime_engine or "").strip()
    if not selected:
        selected = (cfg.runtime.default_engine if cfg is not None else "langgraph")

    if selected not in {"langgraph", "google_adk"}:
        raise ValueError(f"Unsupported runtime engine '{selected}'.")

    if cfg is not None and selected not in set(cfg.runtime.allowed_engines or []):
        raise ValueError(
            f"Runtime engine '{selected}' is not in allowed_engines: {cfg.runtime.allowed_engines}"
        )

    return selected


def build_runtime_app(
    llm,
    prompt_lib: PromptLibrary,
    skills_root: Path = Path(".skills"),
    budget_policy: BudgetPolicy = None,
    tool_log_policy: ToolLogPolicy = None,
    summary_policy: SummaryPolicy = None,
    tools: Optional[List[Any]] = None,
    *,
    engine: str = "langgraph",
    cfg: Optional[AgentConfig] = None,
) -> AgentRuntimeEngine:
    selected = resolve_runtime_engine(cfg=cfg, explicit_runtime_engine=engine)

    langgraph_app = build_langgraph_graph(
        llm=llm,
        prompt_lib=prompt_lib,
        skills_root=skills_root,
        budget_policy=budget_policy,
        tool_log_policy=tool_log_policy,
        summary_policy=summary_policy,
        tools=tools,
    )
    langgraph_engine = LangGraphRuntimeEngine(langgraph_app)

    if selected == "langgraph":
        return langgraph_engine

    if cfg is not None and not bool(getattr(cfg, "adk", None) and cfg.adk.enabled):
        raise RuntimeError(
            "google_adk runtime selected but disabled in config. "
            "Set adk.enabled=true in config to enable ADK runtime."
        )

    return ADKRuntimeEngine(
        langgraph_engine,
        adk_config=(cfg.adk if cfg is not None else None),
    )


def build_engine(*args, **kwargs) -> AgentRuntimeEngine:
    """Alias for build_runtime_app for explicit runtime-oriented call sites."""
    return build_runtime_app(*args, **kwargs)


def build_app(*args, **kwargs):
    """Legacy compatibility wrapper returning LangGraph compiled app object."""
    kwargs.pop("engine", None)
    kwargs.pop("cfg", None)
    runtime = build_runtime_app(*args, engine="langgraph", **kwargs)
    if isinstance(runtime, LangGraphRuntimeEngine):
        return runtime.app
    return runtime
