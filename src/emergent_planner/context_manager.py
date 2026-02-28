"""
ContextManager: assembles the prompt message list for each LLM call.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .models import AgentState, PromptCard, PromptLibrary, SkillMeta
from .policies import BudgetPolicy, ContextSignals
from .skills import render_skills_topk
from .utils import compact_tool_message, msg_tokens


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def detect_signals(state: Dict[str, Any]) -> ContextSignals:
    hist: List[BaseMessage] = state.get("history", [])
    runtime = state.get("runtime", {}) or {}

    is_first = len(hist) == 0 or runtime.get("turn_index", 0) == 0
    after_tool = bool(runtime.get("after_tool", False))
    has_error = bool(runtime.get("last_error", ""))

    user_msg = next((m for m in reversed(hist) if isinstance(m, HumanMessage)), None)
    user_text = (user_msg.content if user_msg else "") or ""
    needs_planning = runtime.get("force_planning", False) or (len(user_text) > 280)

    user_asked_cap = any(k in user_text.lower() for k in [
        "what can you do", "capabilities", "skills", "tools available"
    ])

    return ContextSignals(
        is_first_turn=is_first,
        after_tool=after_tool,
        has_error=has_error,
        needs_planning=needs_planning,
        user_asked_capabilities=user_asked_cap,
    )


def memory_messages(memory: Dict[str, Any]) -> List[SystemMessage]:
    out: List[SystemMessage] = []
    if not memory:
        return out
    summary = memory.get("summary", "")
    if summary:
        out.append(SystemMessage(content="Conversation summary:\n" + summary))
    plan = memory.get("plan", "")
    if plan:
        out.append(SystemMessage(content="Current plan:\n" + plan))
    return out


def active_skill_messages(state: Dict[str, Any]) -> List[SystemMessage]:
    runtime = state.get("runtime", {}) or {}
    name = runtime.get("active_skill_name")
    body = runtime.get("active_skill_body")
    if not name or not body:
        return []
    return [SystemMessage(content=f"ACTIVE SKILL: {name}\n\n{body}")]


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    def __init__(self, prompt_lib: PromptLibrary, budget: BudgetPolicy):
        self.prompt_lib = prompt_lib
        self.budget = budget

    def compose(self, state: Dict[str, Any]) -> List[BaseMessage]:
        hist: List[BaseMessage] = state.get("history", [])
        memory = state.get("memory", {}) or {}
        skills: List[SkillMeta] = state.get("skills", []) or []
        sig = detect_signals(state)

        # 1) Prompt cards (system stack)
        cards = self._select_cards(sig)
        system_msgs = [SystemMessage(content=c.text) for c in cards]

        # 2) Active skill injection (one skill body)
        active_msgs = active_skill_messages(state)

        # 3) Skills registry gating (Top-K)
        skills_msg: List[SystemMessage] = []
        if self._should_inject_skills(sig) and skills:
            user_msg = next((m for m in reversed(hist) if isinstance(m, HumanMessage)), None)
            user_text = (user_msg.content if user_msg else "") or ""
            skills_text = render_skills_topk(skills, user_text, self.budget.max_skills_chars, k=12)
            skills_msg = [SystemMessage(content=skills_text)]

        # 4) Memory
        mem_msgs = memory_messages(memory)

        # 5) History trimming + tool compaction
        curated_hist = self._curate_history(hist, sig)

        assembled = system_msgs + active_msgs + skills_msg + mem_msgs + curated_hist
        return self._fit_to_budget(assembled)

    def _select_cards(self, sig: ContextSignals) -> List[PromptCard]:
        tags = {"core"}
        if sig.after_tool:
            tags.add("after_tool")
        if sig.has_error:
            tags.add("error")
        if sig.needs_planning:
            tags.add("planning")
        return self.prompt_lib.select(lambda c: len(c.tags & tags) > 0)

    def _should_inject_skills(self, sig: ContextSignals) -> bool:
        return sig.is_first_turn or sig.user_asked_capabilities or sig.needs_planning or sig.has_error

    def _curate_history(
        self, hist: List[BaseMessage], sig: ContextSignals
    ) -> List[BaseMessage]:
        if not hist:
            return []
        out: List[BaseMessage] = []
        for m in hist:
            if isinstance(m, ToolMessage):
                out.append(compact_tool_message(m, self.budget.max_tool_snippet_chars))
            else:
                out.append(m)
        return out

    def _fit_to_budget(self, msgs: List[BaseMessage]) -> List[BaseMessage]:
        max_in = max(1000, self.budget.max_prompt_tokens - self.budget.reserved_for_generation)

        system = [m for m in msgs if isinstance(m, SystemMessage)]
        others = [m for m in msgs if not isinstance(m, SystemMessage)]

        def total(ms: List[BaseMessage]) -> int:
            return sum(msg_tokens(m) for m in ms)

        kept_system = system
        if total(kept_system) > max_in:
            truncated: List[BaseMessage] = []
            running = 0
            for sm in kept_system:
                c = sm.content or ""
                allow_chars = max(200, (max_in - running) * 4)
                truncated.append(SystemMessage(
                    content=c[:allow_chars] + ("\n...[truncated]..." if len(c) > allow_chars else "")
                ))
                running = total(truncated)
                if running >= max_in:
                    return truncated
            kept_system = truncated

        kept_others: List[BaseMessage] = []
        running = total(kept_system)
        for m in reversed(others):
            t = msg_tokens(m)
            if running + t > max_in:
                continue
            kept_others.append(m)
            running += t
        kept_others.reverse()
        return kept_system + kept_others
