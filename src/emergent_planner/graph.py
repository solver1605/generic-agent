"""
graph.py — LangGraph graph builder for the Emergent Planner agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .context_manager import ContextManager
from .models import AgentState, PromptLibrary
from .nodes import (
    activate_skill_from_tool_result_node,
    context_node,
    has_tool_calls,
    instrument_node,
    llm_node,
    persist_prompt_artifact_node,
    persist_tool_outputs_node,
    should_summarize,
    summarize_node,
    tools_node,
)
from .policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy


def build_app(
    llm,                                          # e.g. ChatGoogleGenerativeAI(...).bind_tools(tools)
    prompt_lib: PromptLibrary,
    skills_root: Path = Path(".skills"),
    budget_policy: BudgetPolicy = None,
    tool_log_policy: ToolLogPolicy = None,
    summary_policy: SummaryPolicy = None,
    tools: Optional[List[Any]] = None,
):
    """
    Build and compile the LangGraph agent app.

    Requirements:
      - llm must support .invoke(messages) and should be bound to tools:
          llm = base_llm.bind_tools(tools)
      - tools list will be executed by ToolNode on each tool-calling turn.

    Recommended initial state when invoking:
      {
        "history": [HumanMessage(...)],
        "memory":  {},
        "runtime": {"run_id": "...", "turn_index": 0},
        "skills":  discover_skills(Path(".skills")),
      }

    Returns:
      A compiled LangGraph CompiledStateGraph with MemorySaver checkpointing.
    """
    if budget_policy is None:
        budget_policy = BudgetPolicy()
    if tool_log_policy is None:
        tool_log_policy = ToolLogPolicy()
    if summary_policy is None:
        summary_policy = SummaryPolicy()
    if tools is None:
        tools = []

    ctx_mgr = ContextManager(prompt_lib=prompt_lib, budget=budget_policy)
    tool_node_impl = ToolNode(tools)

    # --- Node wrappers ---

    def summarize_wrapped(state: AgentState):
        return summarize_node(state, llm=llm, policy=summary_policy) or {}

    def context_node_wrapped(state: AgentState):
        return context_node(state, ctx_mgr) or {}

    def tools_node_wrapped(state: AgentState):
        return tools_node(state, tool_node_impl) or {}

    def persist_node(state: AgentState):
        return persist_tool_outputs_node(state, tool_log_policy) or {}

    def activate_skill_node(state: AgentState):
        return activate_skill_from_tool_result_node(state) or {}

    def llm_node_wrapped(state: AgentState):
        return llm_node(state, llm=llm) or {}

    # --- Build graph ---
    graph = StateGraph(AgentState)

    graph.add_node("summarize",            instrument_node("summarize",            summarize_wrapped))
    graph.add_node("context",              instrument_node("context",              context_node_wrapped))
    graph.add_node("persist_prompt",       instrument_node("persist_prompt",       persist_prompt_artifact_node))
    graph.add_node("llm",                  instrument_node("llm",                  llm_node_wrapped))
    graph.add_node("tools",                instrument_node("tools",                tools_node_wrapped))
    graph.add_node("persist_tool_outputs", instrument_node("persist_tool_outputs", persist_node))
    graph.add_node("activate_skill",       instrument_node("activate_skill",       activate_skill_node))

    # --- Entry ---
    graph.set_entry_point("summarize")

    # summarize -> (summarize again or skip) -> context
    graph.add_conditional_edges(
        "summarize",
        lambda s: should_summarize(s, summary_policy),
        {"summarize": "summarize", "skip": "context"},
    )
    graph.add_edge("summarize", "context")
    graph.add_edge("context", "persist_prompt")
    graph.add_edge("persist_prompt", "llm")

    # llm -> tools or end
    graph.add_conditional_edges("llm", has_tool_calls, {"tools": "tools", "end": END})

    # tools -> persist -> activate_skill -> summarize -> context ...
    graph.add_edge("tools", "persist_tool_outputs")
    graph.add_edge("persist_tool_outputs", "activate_skill")
    graph.add_edge("activate_skill", "summarize")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
