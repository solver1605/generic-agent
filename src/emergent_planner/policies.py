"""
Policy dataclasses that configure the agent's behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BudgetPolicy:
    """Controls how many tokens are allowed in each part of the assembled prompt."""
    max_prompt_tokens: int = 16_000
    reserved_for_generation: int = 2_000
    max_tool_snippet_chars: int = 1_200
    max_skills_chars: int = 2_500
    max_skills_top_k: int = 12
    planning_trigger_chars: int = 280
    min_input_tokens: int = 1_000
    min_system_message_chars: int = 200


@dataclass
class ToolLogPolicy:
    """Controls how large tool outputs are persisted to disk vs. inlined."""
    artifacts_dir: Path = Path("artifacts/tool_logs")
    max_inline_chars: int = 1_200


@dataclass
class SummaryPolicy:
    """Controls when the conversation history is summarised."""
    summarize_when_history_len_exceeds: int = 50
    keep_last_n_messages: int = 18


@dataclass
class AppPolicy:
    """Top-level application policy (catch-all for simple settings)."""
    artifacts_dir: Path = Path("artifacts/")


@dataclass
class ContextSignals:
    """Signals derived from state used by the ContextManager to decide what to include."""
    is_first_turn: bool
    after_tool: bool
    has_error: bool
    needs_planning: bool
    user_asked_capabilities: bool
