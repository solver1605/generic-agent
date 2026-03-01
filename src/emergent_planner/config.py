"""
Configuration primitives for Emergent Planner.

This module introduces a lightweight config system with model cards that can be
selected by id and optionally overridden via a local YAML file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from langchain_google_genai import ChatGoogleGenerativeAI

from .policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy


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
    max_workers_limit: int = 16
    max_worker_turns_default: int = 8
    max_worker_turns_limit: int = 64
    max_wall_time_s_default: float = 45.0
    max_wall_time_s_limit: float = 600.0
    max_retries_default: int = 1
    max_retries_limit: int = 6
    artifact_dir: Path = Path("artifacts/subagents")
    tool_policy: SubAgentToolPolicyConfig = field(default_factory=SubAgentToolPolicyConfig)


@dataclass
class PolicyProfileConfig:
    id: str
    description: str = ""
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    summary: SummaryPolicy = field(default_factory=SummaryPolicy)
    tool_log: ToolLogPolicy = field(default_factory=ToolLogPolicy)


@dataclass
class PromptCardConfig:
    name: str
    tags: List[str] = field(default_factory=list)
    priority: int = 50
    text: Optional[str] = None
    file: Optional[str] = None


@dataclass
class PromptConfig:
    strategy: Literal["merge", "replace"] = "merge"
    cards: List[PromptCardConfig] = field(default_factory=list)
    disable_cards: List[str] = field(default_factory=list)


@dataclass
class ToolCatalogConfig:
    custom_imports: List[str] = field(default_factory=list)
    allow_module_prefixes: List[str] = field(default_factory=lambda: ["emergent_planner", "src.emergent_planner", "custom_tools"])


@dataclass
class StreamlitUIConfig:
    app_name: str = "Emergent Planner"
    page_title: str = "Emergent Planner UI"


@dataclass
class ProfileToolPolicyConfig:
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)


@dataclass
class SkillsProfileConfig:
    roots: List[str] = field(default_factory=lambda: [".skills"])
    allowlist: List[str] = field(default_factory=list)
    denylist: List[str] = field(default_factory=list)


@dataclass
class AgentProfileConfig:
    id: str
    description: str = ""
    model_card_id: Optional[str] = None
    policy_profile_id: Optional[str] = None
    prompts: PromptConfig = field(default_factory=PromptConfig)
    tools: ProfileToolPolicyConfig = field(default_factory=ProfileToolPolicyConfig)
    skills: SkillsProfileConfig = field(default_factory=SkillsProfileConfig)


@dataclass
class AgentConfig:
    model_cards: List[ModelCard]
    default_model_card: str
    search: SearchConfig = field(default_factory=SearchConfig)
    subagents: SubAgentConfig = field(default_factory=SubAgentConfig)
    policy_profiles: List[PolicyProfileConfig] = field(default_factory=list)
    default_policy_profile: str = "balanced"
    streamlit: StreamlitUIConfig = field(default_factory=StreamlitUIConfig)
    tool_catalog: ToolCatalogConfig = field(default_factory=ToolCatalogConfig)
    agent_profiles: List[AgentProfileConfig] = field(default_factory=list)
    default_agent_profile: str = "default"

    def get_model_card(self, model_card_id: Optional[str] = None) -> ModelCard:
        selected = model_card_id or self.default_model_card
        for card in self.model_cards:
            if card.id == selected:
                return card
        available = ", ".join(c.id for c in self.model_cards)
        raise ValueError(f"Unknown model card '{selected}'. Available: {available}")

    def get_policy_profile(self, policy_profile_id: Optional[str] = None) -> PolicyProfileConfig:
        selected = policy_profile_id or self.default_policy_profile
        for profile in self.policy_profiles:
            if profile.id == selected:
                return profile
        available = ", ".join(p.id for p in self.policy_profiles)
        raise ValueError(f"Unknown policy profile '{selected}'. Available: {available}")

    def get_agent_profile(self, profile_id: Optional[str] = None) -> AgentProfileConfig:
        selected = profile_id or self.default_agent_profile
        for profile in self.agent_profiles:
            if profile.id == selected:
                return profile
        available = ", ".join(p.id for p in self.agent_profiles)
        raise ValueError(f"Unknown agent profile '{selected}'. Available: {available}")


def _default_policy_profiles() -> List[PolicyProfileConfig]:
    return [
        PolicyProfileConfig(
            id="compact",
            description="Tighter prompt context budget for low-latency chat.",
            budget=BudgetPolicy(
                max_prompt_tokens=10_000,
                reserved_for_generation=2_200,
                max_tool_snippet_chars=700,
                max_skills_chars=1_200,
                max_skills_top_k=6,
                planning_trigger_chars=220,
                min_input_tokens=1_200,
                min_system_message_chars=180,
            ),
            summary=SummaryPolicy(
                summarize_when_history_len_exceeds=30,
                keep_last_n_messages=12,
            ),
            tool_log=ToolLogPolicy(
                artifacts_dir=Path("artifacts/tool_logs"),
                max_inline_chars=900,
            ),
        ),
        PolicyProfileConfig(
            id="balanced",
            description="Default balance between context richness and latency.",
            budget=BudgetPolicy(
                max_prompt_tokens=16_000,
                reserved_for_generation=2_000,
                max_tool_snippet_chars=1_200,
                max_skills_chars=2_500,
                max_skills_top_k=12,
                planning_trigger_chars=280,
                min_input_tokens=1_000,
                min_system_message_chars=200,
            ),
            summary=SummaryPolicy(
                summarize_when_history_len_exceeds=50,
                keep_last_n_messages=18,
            ),
            tool_log=ToolLogPolicy(
                artifacts_dir=Path("artifacts/tool_logs"),
                max_inline_chars=1_200,
            ),
        ),
        PolicyProfileConfig(
            id="deep_research",
            description="Larger prompt budgets for synthesis-heavy research tasks.",
            budget=BudgetPolicy(
                max_prompt_tokens=24_000,
                reserved_for_generation=3_000,
                max_tool_snippet_chars=2_200,
                max_skills_chars=4_000,
                max_skills_top_k=20,
                planning_trigger_chars=180,
                min_input_tokens=1_400,
                min_system_message_chars=220,
            ),
            summary=SummaryPolicy(
                summarize_when_history_len_exceeds=70,
                keep_last_n_messages=24,
            ),
            tool_log=ToolLogPolicy(
                artifacts_dir=Path("artifacts/tool_logs"),
                max_inline_chars=2_000,
            ),
        ),
    ]


def _default_agent_profiles(default_model_card: str, default_policy_profile: str) -> List[AgentProfileConfig]:
    return [
        AgentProfileConfig(
            id="default",
            description="Legacy-compatible default generic agent profile.",
            model_card_id=default_model_card,
            policy_profile_id=default_policy_profile,
            prompts=PromptConfig(strategy="merge"),
            tools=ProfileToolPolicyConfig(),
            skills=SkillsProfileConfig(
                roots=[".skills"],
                allowlist=[],
                denylist=[],
            ),
        )
    ]


def default_agent_config() -> AgentConfig:
    default_model_card = "gemini_flash_fast"
    default_policy_profile = "balanced"
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
        default_model_card=default_model_card,
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
            max_workers_limit=16,
            max_worker_turns_default=8,
            max_worker_turns_limit=64,
            max_wall_time_s_default=45.0,
            max_wall_time_s_limit=600.0,
            max_retries_default=1,
            max_retries_limit=6,
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
        policy_profiles=_default_policy_profiles(),
        default_policy_profile=default_policy_profile,
        tool_catalog=ToolCatalogConfig(
            custom_imports=[],
            allow_module_prefixes=["emergent_planner", "src.emergent_planner", "custom_tools"],
        ),
        agent_profiles=_default_agent_profiles(default_model_card, default_policy_profile),
        default_agent_profile="default",
    )


def _parse_prompt_config(raw_prompt: Dict[str, Any]) -> PromptConfig:
    strategy = str(raw_prompt.get("strategy", "merge")).strip().lower() or "merge"
    if strategy not in {"merge", "replace"}:
        strategy = "merge"

    cards: List[PromptCardConfig] = []
    for card_raw in raw_prompt.get("cards", []) or []:
        if not isinstance(card_raw, dict):
            continue
        name = str(card_raw.get("name", "")).strip()
        if not name:
            continue
        cards.append(
            PromptCardConfig(
                name=name,
                tags=[str(t).strip() for t in (card_raw.get("tags", []) or []) if str(t).strip()],
                priority=int(card_raw.get("priority", 50)),
                text=card_raw.get("text"),
                file=card_raw.get("file"),
            )
        )

    return PromptConfig(
        strategy=strategy,
        cards=cards,
        disable_cards=[str(x).strip() for x in (raw_prompt.get("disable_cards", []) or []) if str(x).strip()],
    )


def _parse_agent_profiles(raw: Dict[str, Any], *, fallback_model: str, fallback_policy: str) -> tuple[List[AgentProfileConfig], str]:
    profiles_raw = raw.get("agent_profiles", []) or []
    profile_list: List[AgentProfileConfig] = []

    for item in profiles_raw:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", "")).strip()
        if not pid:
            continue

        prompts = _parse_prompt_config(item.get("prompts", {}) or {})
        tools_raw = item.get("tools", {}) or {}
        skills_raw = item.get("skills", {}) or {}

        profile_list.append(
            AgentProfileConfig(
                id=pid,
                description=str(item.get("description", "") or "").strip(),
                model_card_id=(str(item.get("model_card_id")).strip() if item.get("model_card_id") is not None else None),
                policy_profile_id=(
                    str(item.get("policy_profile_id")).strip() if item.get("policy_profile_id") is not None else None
                ),
                prompts=prompts,
                tools=ProfileToolPolicyConfig(
                    allow=[str(x).strip() for x in (tools_raw.get("allow", []) or []) if str(x).strip()],
                    deny=[str(x).strip() for x in (tools_raw.get("deny", []) or []) if str(x).strip()],
                ),
                skills=SkillsProfileConfig(
                    roots=[str(x).strip() for x in (skills_raw.get("roots", [".skills"]) or [".skills"]) if str(x).strip()],
                    allowlist=[str(x).strip() for x in (skills_raw.get("allowlist", []) or []) if str(x).strip()],
                    denylist=[str(x).strip() for x in (skills_raw.get("denylist", []) or []) if str(x).strip()],
                ),
            )
        )

    if not profile_list:
        profile_list = _default_agent_profiles(fallback_model, fallback_policy)
        return profile_list, "default"

    default_agent_profile = str(raw.get("default_agent_profile", profile_list[0].id)).strip() or profile_list[0].id
    if default_agent_profile not in {p.id for p in profile_list}:
        default_agent_profile = profile_list[0].id
    return profile_list, default_agent_profile


def load_agent_config(path: Path = Path("agent_config.yaml")) -> AgentConfig:
    """
    Load agent config from YAML if present; otherwise return defaults.
    """
    if not path.exists():
        return default_agent_config()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = default_agent_config()

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
        cards = base.model_cards

    default_id = str(raw.get("default_model_card") or cards[0].id)

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
        max_workers_limit=int(sub_raw.get("max_workers_limit", base.subagents.max_workers_limit)),
        max_worker_turns_default=int(sub_raw.get("max_worker_turns_default", base.subagents.max_worker_turns_default)),
        max_worker_turns_limit=int(sub_raw.get("max_worker_turns_limit", base.subagents.max_worker_turns_limit)),
        max_wall_time_s_default=float(sub_raw.get("max_wall_time_s_default", base.subagents.max_wall_time_s_default)),
        max_wall_time_s_limit=float(sub_raw.get("max_wall_time_s_limit", base.subagents.max_wall_time_s_limit)),
        max_retries_default=int(sub_raw.get("max_retries_default", base.subagents.max_retries_default)),
        max_retries_limit=int(sub_raw.get("max_retries_limit", base.subagents.max_retries_limit)),
        artifact_dir=Path(str(sub_raw.get("artifact_dir", base.subagents.artifact_dir))),
        tool_policy=SubAgentToolPolicyConfig(
            allow_by_task_type=dict(tool_policy_raw.get("allow_by_task_type", base.subagents.tool_policy.allow_by_task_type)),
            denylist=list(tool_policy_raw.get("denylist", base.subagents.tool_policy.denylist)),
            permitted_overrides=list(
                tool_policy_raw.get("permitted_overrides", base.subagents.tool_policy.permitted_overrides)
            ),
        ),
    )
    subagents.max_workers_limit = max(int(subagents.max_workers_limit), int(subagents.max_workers_default))
    subagents.max_worker_turns_limit = max(int(subagents.max_worker_turns_limit), int(subagents.max_worker_turns_default))
    subagents.max_wall_time_s_limit = max(float(subagents.max_wall_time_s_limit), float(subagents.max_wall_time_s_default))
    subagents.max_retries_limit = max(int(subagents.max_retries_limit), int(subagents.max_retries_default))

    profiles_raw = raw.get("policy_profiles", []) or []
    base_profiles = {p.id: p for p in base.policy_profiles}
    profile_list: List[PolicyProfileConfig] = list(base.policy_profiles)
    if profiles_raw:
        profile_list = []
        for item in profiles_raw:
            pid = str(item.get("id", "")).strip()
            if not pid:
                continue
            fallback = base_profiles.get(pid, PolicyProfileConfig(id=pid))
            budget_raw = item.get("budget", {}) or {}
            summary_raw = item.get("summary", {}) or {}
            tool_raw = item.get("tool_log", {}) or {}
            profile_list.append(
                PolicyProfileConfig(
                    id=pid,
                    description=str(item.get("description", fallback.description or "")).strip(),
                    budget=BudgetPolicy(
                        max_prompt_tokens=int(budget_raw.get("max_prompt_tokens", fallback.budget.max_prompt_tokens)),
                        reserved_for_generation=int(
                            budget_raw.get("reserved_for_generation", fallback.budget.reserved_for_generation)
                        ),
                        max_tool_snippet_chars=int(
                            budget_raw.get("max_tool_snippet_chars", fallback.budget.max_tool_snippet_chars)
                        ),
                        max_skills_chars=int(budget_raw.get("max_skills_chars", fallback.budget.max_skills_chars)),
                        max_skills_top_k=int(budget_raw.get("max_skills_top_k", fallback.budget.max_skills_top_k)),
                        planning_trigger_chars=int(
                            budget_raw.get("planning_trigger_chars", fallback.budget.planning_trigger_chars)
                        ),
                        min_input_tokens=int(budget_raw.get("min_input_tokens", fallback.budget.min_input_tokens)),
                        min_system_message_chars=int(
                            budget_raw.get("min_system_message_chars", fallback.budget.min_system_message_chars)
                        ),
                    ),
                    summary=SummaryPolicy(
                        summarize_when_history_len_exceeds=int(
                            summary_raw.get(
                                "summarize_when_history_len_exceeds",
                                fallback.summary.summarize_when_history_len_exceeds,
                            )
                        ),
                        keep_last_n_messages=int(
                            summary_raw.get("keep_last_n_messages", fallback.summary.keep_last_n_messages)
                        ),
                    ),
                    tool_log=ToolLogPolicy(
                        artifacts_dir=Path(str(tool_raw.get("artifacts_dir", fallback.tool_log.artifacts_dir))),
                        max_inline_chars=int(tool_raw.get("max_inline_chars", fallback.tool_log.max_inline_chars)),
                    ),
                )
            )

    default_policy_profile = str(raw.get("default_policy_profile", base.default_policy_profile)).strip()
    if not default_policy_profile and profile_list:
        default_policy_profile = profile_list[0].id
    if profile_list and default_policy_profile not in {p.id for p in profile_list}:
        default_policy_profile = profile_list[0].id
    if not profile_list:
        profile_list = list(base.policy_profiles)
        default_policy_profile = base.default_policy_profile

    tool_catalog_raw = raw.get("tool_catalog", {}) or {}
    tool_catalog = ToolCatalogConfig(
        custom_imports=[str(x).strip() for x in (tool_catalog_raw.get("custom_imports", []) or []) if str(x).strip()],
        allow_module_prefixes=[
            str(x).strip()
            for x in (tool_catalog_raw.get("allow_module_prefixes", base.tool_catalog.allow_module_prefixes) or [])
            if str(x).strip()
        ]
        or list(base.tool_catalog.allow_module_prefixes),
    )
    streamlit_raw = raw.get("streamlit", {}) or raw.get("ui", {}) or {}
    streamlit_cfg = StreamlitUIConfig(
        app_name=(
            str(streamlit_raw.get("app_name", base.streamlit.app_name)).strip()
            or base.streamlit.app_name
        ),
        page_title=(
            str(streamlit_raw.get("page_title", base.streamlit.page_title)).strip()
            or base.streamlit.page_title
        ),
    )

    agent_profiles, default_agent_profile = _parse_agent_profiles(
        raw,
        fallback_model=default_id,
        fallback_policy=default_policy_profile,
    )

    return AgentConfig(
        model_cards=cards,
        default_model_card=default_id,
        search=SearchConfig(providers=providers, budgets=budgets, defaults=defaults),
        subagents=subagents,
        policy_profiles=profile_list,
        default_policy_profile=default_policy_profile,
        streamlit=streamlit_cfg,
        tool_catalog=tool_catalog,
        agent_profiles=agent_profiles,
        default_agent_profile=default_agent_profile,
    )


def resolve_runtime_policies(
    cfg: AgentConfig,
    policy_profile_id: Optional[str] = None,
) -> tuple[BudgetPolicy, ToolLogPolicy, SummaryPolicy, str]:
    profile = cfg.get_policy_profile(policy_profile_id)
    return profile.budget, profile.tool_log, profile.summary, profile.id


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
