"""
Configuration primitives for Emergent Planner.

This module introduces a lightweight config system with model cards that can be
selected by id and optionally overridden via a local YAML file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class ModelCard:
    id: str
    provider: str
    model_name: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_llm_kwargs(self) -> Dict[str, Any]:
        """
        Convert card settings to kwargs used by the model constructor.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
        }

        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        extra = dict(self.model_kwargs or {})
        if self.thinking_budget is not None and "thinking_budget" not in extra:
            # Forwarded as provider-specific model kwargs for Gemini families.
            extra["thinking_budget"] = self.thinking_budget
        if extra:
            kwargs["model_kwargs"] = extra

        return kwargs


@dataclass
class SearchProviderConfig:
    name: str
    enabled: bool = True
    api_key_env: str = ""
    timeout_s: float = 8.0
    weight: float = 1.0


@dataclass
class SearchBudgetConfig:
    max_providers_per_call: int = 2
    max_results_per_provider: int = 8
    max_total_results_before_rerank: int = 40
    max_enriched_results: int = 3
    global_timeout_s: float = 12.0


@dataclass
class SearchDefaults:
    default_top_k: int = 8
    default_recency_days: Optional[int] = 30
    default_mode: str = "balanced"
    default_enrich: bool = False
    provider_priority: List[str] = field(default_factory=lambda: ["tavily", "brave"])


@dataclass
class SearchConfig:
    providers: List[SearchProviderConfig] = field(default_factory=list)
    budgets: SearchBudgetConfig = field(default_factory=SearchBudgetConfig)
    defaults: SearchDefaults = field(default_factory=SearchDefaults)


@dataclass
class SubAgentToolPolicyConfig:
    allow_by_task_type: Dict[str, List[str]] = field(default_factory=dict)
    denylist: List[str] = field(default_factory=list)
    permitted_overrides: List[str] = field(default_factory=list)


@dataclass
class SubAgentConfig:
    enabled: bool = True
    max_workers_default: int = 4
    max_worker_turns_default: int = 8
    max_wall_time_s_default: float = 45.0
    max_retries_default: int = 1
    artifact_dir: Path = Path("artifacts/subagents")
    tool_policy: SubAgentToolPolicyConfig = field(default_factory=SubAgentToolPolicyConfig)


@dataclass
class AgentConfig:
    model_cards: List[ModelCard]
    default_model_card: str
    search: SearchConfig = field(default_factory=SearchConfig)
    subagents: SubAgentConfig = field(default_factory=SubAgentConfig)

    def get_model_card(self, model_card_id: Optional[str] = None) -> ModelCard:
        selected = model_card_id or self.default_model_card
        for card in self.model_cards:
            if card.id == selected:
                return card
        available = ", ".join(c.id for c in self.model_cards)
        raise ValueError(f"Unknown model card '{selected}'. Available: {available}")


def default_agent_config() -> AgentConfig:
    return AgentConfig(
        model_cards=[
            ModelCard(
                id="gemini_flash_fast",
                provider="google_genai",
                model_name="models/gemini-3-flash-preview",
                temperature=0.0,
                thinking_budget=0,
            ),
            ModelCard(
                id="gemini_flash_reasoning",
                provider="google_genai",
                model_name="models/gemini-3-flash-preview",
                temperature=0.0,
                thinking_budget=1024,
            ),
        ],
        default_model_card="gemini_flash_fast",
        search=SearchConfig(
            providers=[
                SearchProviderConfig(
                    name="tavily",
                    enabled=True,
                    api_key_env="TAVILY_API_KEY",
                    timeout_s=8.0,
                    weight=1.0,
                ),
                SearchProviderConfig(
                    name="brave",
                    enabled=True,
                    api_key_env="BRAVE_API_KEY",
                    timeout_s=8.0,
                    weight=1.0,
                ),
            ],
            budgets=SearchBudgetConfig(
                max_providers_per_call=2,
                max_results_per_provider=8,
                max_total_results_before_rerank=40,
                max_enriched_results=3,
                global_timeout_s=12.0,
            ),
            defaults=SearchDefaults(
                default_top_k=8,
                default_recency_days=30,
                default_mode="balanced",
                default_enrich=False,
                provider_priority=["tavily", "brave"],
            ),
        ),
        subagents=SubAgentConfig(
            enabled=True,
            max_workers_default=4,
            max_worker_turns_default=8,
            max_wall_time_s_default=45.0,
            max_retries_default=1,
            artifact_dir=Path("artifacts/subagents"),
            tool_policy=SubAgentToolPolicyConfig(
                allow_by_task_type={
                    "default": ["read_file", "read_file_range", "search_web", "python_repl"],
                    "research": ["search_web", "read_file", "read_file_range"],
                    "analysis": ["python_repl", "read_file", "read_file_range", "search_web"],
                    "writing": ["read_file", "read_file_range", "search_web"],
                },
                denylist=["write_file", "verify_with_user", "spawn_subagents"],
                permitted_overrides=["search_web", "python_repl", "read_file", "read_file_range"],
            ),
        ),
    )


def load_agent_config(path: Path = Path("agent_config.yaml")) -> AgentConfig:
    """
    Load agent config from YAML if present; otherwise return defaults.

    Expected YAML shape:
      default_model_card: gemini_flash_fast
      model_cards:
        - id: gemini_flash_fast
          provider: google_genai
          model_name: models/gemini-2.0-flash
          temperature: 0
          thinking_budget: 0
          max_output_tokens: 2048
          model_kwargs:
            top_p: 0.95
    """
    if not path.exists():
        return default_agent_config()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cards_raw = raw.get("model_cards", []) or []
    cards: List[ModelCard] = []
    for item in cards_raw:
        cards.append(
            ModelCard(
                id=str(item["id"]),
                provider=str(item.get("provider", "google_genai")),
                model_name=str(item["model_name"]),
                temperature=float(item.get("temperature", 0.0)),
                max_output_tokens=item.get("max_output_tokens"),
                thinking_budget=item.get("thinking_budget"),
                model_kwargs=dict(item.get("model_kwargs", {}) or {}),
            )
        )

    if not cards:
        base = default_agent_config()
        cards = base.model_cards

    default_id = str(raw.get("default_model_card") or cards[0].id)
    base = default_agent_config()

    search_raw = raw.get("search", {}) or {}
    providers_raw = search_raw.get("providers", []) or []
    providers = list(base.search.providers)
    if providers_raw:
        providers = [
            SearchProviderConfig(
                name=str(item.get("name", "")).strip().lower(),
                enabled=bool(item.get("enabled", True)),
                api_key_env=str(item.get("api_key_env", "")).strip(),
                timeout_s=float(item.get("timeout_s", 8.0)),
                weight=float(item.get("weight", 1.0)),
            )
            for item in providers_raw
            if str(item.get("name", "")).strip()
        ]

    budgets_raw = search_raw.get("budgets", {}) or {}
    budgets = SearchBudgetConfig(
        max_providers_per_call=int(budgets_raw.get("max_providers_per_call", base.search.budgets.max_providers_per_call)),
        max_results_per_provider=int(budgets_raw.get("max_results_per_provider", base.search.budgets.max_results_per_provider)),
        max_total_results_before_rerank=int(
            budgets_raw.get("max_total_results_before_rerank", base.search.budgets.max_total_results_before_rerank)
        ),
        max_enriched_results=int(budgets_raw.get("max_enriched_results", base.search.budgets.max_enriched_results)),
        global_timeout_s=float(budgets_raw.get("global_timeout_s", base.search.budgets.global_timeout_s)),
    )

    defaults_raw = search_raw.get("defaults", {}) or {}
    defaults = SearchDefaults(
        default_top_k=int(defaults_raw.get("default_top_k", base.search.defaults.default_top_k)),
        default_recency_days=defaults_raw.get("default_recency_days", base.search.defaults.default_recency_days),
        default_mode=str(defaults_raw.get("default_mode", base.search.defaults.default_mode)),
        default_enrich=bool(defaults_raw.get("default_enrich", base.search.defaults.default_enrich)),
        provider_priority=list(defaults_raw.get("provider_priority", base.search.defaults.provider_priority)),
    )

    sub_raw = raw.get("subagents", {}) or {}
    tool_policy_raw = sub_raw.get("tool_policy", {}) or {}
    subagents = SubAgentConfig(
        enabled=bool(sub_raw.get("enabled", base.subagents.enabled)),
        max_workers_default=int(sub_raw.get("max_workers_default", base.subagents.max_workers_default)),
        max_worker_turns_default=int(sub_raw.get("max_worker_turns_default", base.subagents.max_worker_turns_default)),
        max_wall_time_s_default=float(sub_raw.get("max_wall_time_s_default", base.subagents.max_wall_time_s_default)),
        max_retries_default=int(sub_raw.get("max_retries_default", base.subagents.max_retries_default)),
        artifact_dir=Path(str(sub_raw.get("artifact_dir", base.subagents.artifact_dir))),
        tool_policy=SubAgentToolPolicyConfig(
            allow_by_task_type=dict(tool_policy_raw.get("allow_by_task_type", base.subagents.tool_policy.allow_by_task_type)),
            denylist=list(tool_policy_raw.get("denylist", base.subagents.tool_policy.denylist)),
            permitted_overrides=list(
                tool_policy_raw.get("permitted_overrides", base.subagents.tool_policy.permitted_overrides)
            ),
        ),
    )

    return AgentConfig(
        model_cards=cards,
        default_model_card=default_id,
        search=SearchConfig(providers=providers, budgets=budgets, defaults=defaults),
        subagents=subagents,
    )


def build_llm_from_model_card(card: ModelCard, *, google_api_key: str):
    """
    Construct an LLM instance from a model card.
    """
    if card.provider != "google_genai":
        raise ValueError(
            f"Unsupported provider '{card.provider}'. "
            "Current implementation supports only 'google_genai'."
        )
    return ChatGoogleGenerativeAI(google_api_key=google_api_key, **card.to_llm_kwargs())
