"""
emergent_planner — Public API
"""
import warnings

if __name__.startswith("src."):
    warnings.warn(
        "`src.emergent_planner` is deprecated and will be removed in the next minor release; "
        "use `emergent_planner` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

from .runtime.factory import build_app, build_runtime_app
from .config import (
    ADKConfig,
    AgentConfig,
    AgentProfileConfig,
    ModelCard,
    PolicyProfileConfig,
    ProfileToolPolicyConfig,
    PromptCardConfig,
    PromptConfig,
    SearchBudgetConfig,
    SearchConfig,
    SearchDefaults,
    SearchProviderConfig,
    SkillsProfileConfig,
    RuntimeEngineConfig,
    StreamlitUIConfig,
    SubAgentConfig,
    SubAgentToolPolicyConfig,
    ToolCatalogConfig,
    build_llm_from_model_card,
    default_agent_config,
    load_agent_config,
    resolve_runtime_policies,
)
from .models import AgentState, PromptCard, PromptLibrary, SkillMeta, Step, VerifyRequest
from .policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy
from .prompt_loader import build_prompt_lib_for_profile
from .prompts import make_default_prompt_lib
from .skills import discover_skills, discover_skills_in_roots
from .tool_loader import build_tool_catalog, resolve_tools_for_profile
from .subagents import (
    SubAgentError,
    SubAgentExecutionConfig,
    SubAgentResult,
    SubAgentRunRecord,
    SubAgentTask,
    run_subagents,
)
from .tools import DEFAULT_TOOLS

__all__ = [
    "build_app",
    "build_runtime_app",
    "ADKConfig",
    "AgentConfig",
    "AgentProfileConfig",
    "ModelCard",
    "PolicyProfileConfig",
    "PromptCardConfig",
    "PromptConfig",
    "ToolCatalogConfig",
    "ProfileToolPolicyConfig",
    "SkillsProfileConfig",
    "RuntimeEngineConfig",
    "SearchProviderConfig",
    "SearchBudgetConfig",
    "SearchDefaults",
    "SearchConfig",
    "StreamlitUIConfig",
    "SubAgentConfig",
    "SubAgentToolPolicyConfig",
    "default_agent_config",
    "load_agent_config",
    "resolve_runtime_policies",
    "build_llm_from_model_card",
    "AgentState",
    "PromptCard",
    "PromptLibrary",
    "SkillMeta",
    "Step",
    "VerifyRequest",
    "BudgetPolicy",
    "SummaryPolicy",
    "ToolLogPolicy",
    "build_prompt_lib_for_profile",
    "build_tool_catalog",
    "resolve_tools_for_profile",
    "make_default_prompt_lib",
    "discover_skills",
    "discover_skills_in_roots",
    "SubAgentTask",
    "SubAgentExecutionConfig",
    "SubAgentResult",
    "SubAgentError",
    "SubAgentRunRecord",
    "run_subagents",
    "DEFAULT_TOOLS",
]
