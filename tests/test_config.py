from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.config import (
    AgentConfig,
    ModelCard,
    build_llm_from_model_card,
    default_agent_config,
    load_agent_config,
    resolve_runtime_policies,
)


class TestConfig(TestCase):
    def test_default_agent_config_has_expected_cards(self):
        cfg = default_agent_config()
        self.assertIsInstance(cfg, AgentConfig)
        ids = {c.id for c in cfg.model_cards}
        self.assertIn("gemini_flash_fast", ids)
        self.assertIn("gemini_flash_reasoning", ids)
        self.assertEqual(cfg.get_model_card().id, cfg.default_model_card)

    def test_model_card_to_llm_kwargs_includes_thinking_budget_and_generation_limits(self):
        card = ModelCard(
            id="x",
            provider="google_genai",
            model_name="models/gemini-2.0-flash",
            temperature=0.1,
            max_output_tokens=123,
            thinking_budget=456,
            model_kwargs={"top_p": 0.8},
        )
        kwargs = card.to_llm_kwargs()

        self.assertEqual(kwargs["model"], "models/gemini-2.0-flash")
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_output_tokens"], 123)
        self.assertEqual(kwargs["model_kwargs"]["thinking_budget"], 456)
        self.assertEqual(kwargs["model_kwargs"]["top_p"], 0.8)

    def test_model_card_does_not_override_explicit_model_kwargs_thinking_budget(self):
        card = ModelCard(
            id="x",
            provider="google_genai",
            model_name="m",
            thinking_budget=999,
            model_kwargs={"thinking_budget": 321},
        )
        kwargs = card.to_llm_kwargs()
        self.assertEqual(kwargs["model_kwargs"]["thinking_budget"], 321)

    def test_load_agent_config_from_missing_file_returns_defaults(self):
        cfg = load_agent_config(Path("/tmp/this-file-should-not-exist-xyz.yaml"))
        self.assertEqual(cfg.get_model_card().id, cfg.default_model_card)
        self.assertGreaterEqual(len(cfg.model_cards), 1)

    def test_load_agent_config_from_yaml(self):
        p = Path("/tmp/test_agent_config.yaml")
        p.write_text(
            """
default_model_card: card_b
model_cards:
  - id: card_a
    provider: google_genai
    model_name: models/gemini-2.0-flash
    temperature: 0.0
    thinking_budget: 0
  - id: card_b
    provider: google_genai
    model_name: models/gemini-2.0-flash
    temperature: 0.2
    thinking_budget: 1024
    max_output_tokens: 4096
    model_kwargs:
      top_p: 0.9
""".strip(),
            encoding="utf-8",
        )
        try:
            cfg = load_agent_config(p)
            selected = cfg.get_model_card()

            self.assertEqual(cfg.default_model_card, "card_b")
            self.assertEqual(selected.id, "card_b")
            self.assertEqual(selected.temperature, 0.2)
            self.assertEqual(selected.thinking_budget, 1024)
            self.assertEqual(selected.max_output_tokens, 4096)
            self.assertEqual(selected.model_kwargs["top_p"], 0.9)
        finally:
            if p.exists():
                p.unlink()

    def test_get_model_card_raises_for_unknown_id(self):
        cfg = default_agent_config()
        with self.assertRaises(ValueError):
            cfg.get_model_card("does-not-exist")

    def test_build_llm_from_model_card_uses_google_class(self):
        captured = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch("src.emergent_planner.config.ChatGoogleGenerativeAI", FakeLLM):
            card = ModelCard(
                id="x",
                provider="google_genai",
                model_name="models/gemini-2.0-flash",
                temperature=0.0,
                thinking_budget=64,
            )
            llm = build_llm_from_model_card(card, google_api_key="k")

        self.assertEqual(llm.__class__, FakeLLM)
        self.assertEqual(captured["google_api_key"], "k")
        self.assertEqual(captured["model"], "models/gemini-2.0-flash")
        self.assertEqual(captured["model_kwargs"]["thinking_budget"], 64)

    def test_build_llm_from_model_card_rejects_unsupported_provider(self):
        card = ModelCard(id="x", provider="openai", model_name="gpt")
        with self.assertRaises(ValueError):
            build_llm_from_model_card(card, google_api_key="k")

    def test_default_config_contains_subagent_defaults(self):
        cfg = default_agent_config()
        self.assertTrue(cfg.subagents.enabled)
        self.assertGreaterEqual(cfg.subagents.max_workers_default, 1)
        self.assertIn("spawn_subagents", cfg.subagents.tool_policy.denylist)

    def test_default_config_contains_policy_profiles(self):
        cfg = default_agent_config()
        ids = [p.id for p in cfg.policy_profiles]
        self.assertIn("compact", ids)
        self.assertIn("balanced", ids)
        self.assertIn("deep_research", ids)
        self.assertEqual(cfg.default_policy_profile, "balanced")

    def test_load_agent_config_parses_subagent_section(self):
        p = Path("/tmp/test_agent_config_sub.yaml")
        p.write_text(
            """
default_model_card: card_a
model_cards:
  - id: card_a
    provider: google_genai
    model_name: models/gemini-2.0-flash
subagents:
  enabled: true
  max_workers_default: 3
  max_workers_limit: 12
  max_worker_turns_default: 6
  max_worker_turns_limit: 20
  max_wall_time_s_default: 30
  max_wall_time_s_limit: 180
  max_retries_default: 2
  max_retries_limit: 5
  artifact_dir: artifacts/custom_subagents
  tool_policy:
    allow_by_task_type:
      default: [read_file, search_web]
    denylist: [write_file, verify_with_user]
    permitted_overrides: [search_web]
""".strip(),
            encoding="utf-8",
        )
        try:
            cfg = load_agent_config(p)
            self.assertEqual(cfg.subagents.max_workers_default, 3)
            self.assertEqual(cfg.subagents.max_workers_limit, 12)
            self.assertEqual(cfg.subagents.max_worker_turns_default, 6)
            self.assertEqual(cfg.subagents.max_worker_turns_limit, 20)
            self.assertEqual(cfg.subagents.max_wall_time_s_default, 30)
            self.assertEqual(cfg.subagents.max_wall_time_s_limit, 180)
            self.assertEqual(cfg.subagents.max_retries_default, 2)
            self.assertEqual(cfg.subagents.max_retries_limit, 5)
            self.assertEqual(str(cfg.subagents.artifact_dir), "artifacts/custom_subagents")
            self.assertEqual(cfg.subagents.tool_policy.allow_by_task_type["default"], ["read_file", "search_web"])
            self.assertIn("write_file", cfg.subagents.tool_policy.denylist)
        finally:
            if p.exists():
                p.unlink()

    def test_load_agent_config_parses_policy_profiles(self):
        p = Path("/tmp/test_agent_config_policy.yaml")
        p.write_text(
            """
default_model_card: card_a
default_policy_profile: deep_research
model_cards:
  - id: card_a
    provider: google_genai
    model_name: models/gemini-2.0-flash
policy_profiles:
  - id: balanced
    description: custom balanced
    budget:
      max_prompt_tokens: 12345
      reserved_for_generation: 2345
      max_tool_snippet_chars: 345
      max_skills_chars: 678
      max_skills_top_k: 7
      planning_trigger_chars: 111
      min_input_tokens: 888
      min_system_message_chars: 99
    summary:
      summarize_when_history_len_exceeds: 22
      keep_last_n_messages: 9
    tool_log:
      artifacts_dir: artifacts/custom_tool_logs
      max_inline_chars: 444
  - id: deep_research
    description: research profile
""".strip(),
            encoding="utf-8",
        )
        try:
            cfg = load_agent_config(p)
            self.assertEqual(cfg.default_policy_profile, "deep_research")
            balanced = cfg.get_policy_profile("balanced")
            self.assertEqual(balanced.description, "custom balanced")
            self.assertEqual(balanced.budget.max_prompt_tokens, 12345)
            self.assertEqual(balanced.budget.max_skills_top_k, 7)
            self.assertEqual(balanced.summary.keep_last_n_messages, 9)
            self.assertEqual(str(balanced.tool_log.artifacts_dir), "artifacts/custom_tool_logs")
            budget, tool_log, summary, resolved_id = resolve_runtime_policies(cfg, "balanced")
            self.assertEqual(resolved_id, "balanced")
            self.assertEqual(budget.max_prompt_tokens, 12345)
            self.assertEqual(tool_log.max_inline_chars, 444)
            self.assertEqual(summary.summarize_when_history_len_exceeds, 22)
        finally:
            if p.exists():
                p.unlink()

    def test_default_config_contains_default_agent_profile(self):
        cfg = default_agent_config()
        self.assertTrue(cfg.agent_profiles)
        self.assertEqual(cfg.default_agent_profile, "default")
        prof = cfg.get_agent_profile()
        self.assertEqual(prof.id, "default")

    def test_default_config_contains_streamlit_branding(self):
        cfg = default_agent_config()
        self.assertEqual(cfg.streamlit.app_name, "Emergent Planner")
        self.assertEqual(cfg.streamlit.page_title, "Emergent Planner UI")

    def test_load_agent_config_parses_streamlit_branding(self):
        p = Path("/tmp/test_agent_config_streamlit_branding.yaml")
        p.write_text(
            """
default_model_card: card_a
model_cards:
  - id: card_a
    provider: google_genai
    model_name: models/gemini-2.0-flash
streamlit:
  app_name: Research Copilot
  page_title: Research Copilot Console
""".strip(),
            encoding="utf-8",
        )
        try:
            cfg = load_agent_config(p)
            self.assertEqual(cfg.streamlit.app_name, "Research Copilot")
            self.assertEqual(cfg.streamlit.page_title, "Research Copilot Console")
        finally:
            if p.exists():
                p.unlink()

    def test_load_agent_config_legacy_autowrap_profile(self):
        p = Path("/tmp/test_agent_config_legacy_wrap.yaml")
        p.write_text(
            """
default_model_card: card_a
default_policy_profile: balanced
model_cards:
  - id: card_a
    provider: google_genai
    model_name: models/gemini-2.0-flash
""".strip(),
            encoding="utf-8",
        )
        try:
            cfg = load_agent_config(p)
            self.assertEqual(cfg.default_agent_profile, "default")
            self.assertEqual(len(cfg.agent_profiles), 1)
            prof = cfg.get_agent_profile("default")
            self.assertEqual(prof.model_card_id, "card_a")
            self.assertEqual(prof.policy_profile_id, "balanced")
            self.assertEqual(prof.skills.roots, [".skills"])
            self.assertEqual(prof.skills.denylist, [])
        finally:
            if p.exists():
                p.unlink()

    def test_load_agent_config_with_multiple_agent_profiles(self):
        p = Path("/tmp/test_agent_config_profiles.yaml")
        p.write_text(
            """
default_model_card: card_a
default_policy_profile: balanced
default_agent_profile: researcher
model_cards:
  - id: card_a
    provider: google_genai
    model_name: models/gemini-2.0-flash
tool_catalog:
  allow_module_prefixes: [src.emergent_planner, custom_tools]
  custom_imports: [custom_tools.research:build_tools]
agent_profiles:
  - id: default
    model_card_id: card_a
    policy_profile_id: balanced
    skills:
      roots: [.skills]
      allowlist: []
  - id: researcher
    description: Research profile
    model_card_id: card_a
    policy_profile_id: deep_research
    prompts:
      strategy: merge
      disable_cards: [guidelines]
      cards:
        - name: guidelines
          tags: [core]
          priority: 10
          text: custom guidelines
    tools:
      allow: [load_skill, search_web]
      deny: [python_repl]
    skills:
      roots: [.skills, services/research/.skills]
      allowlist: [deep-research]
      denylist: [internal-only-skill]
""".strip(),
            encoding="utf-8",
        )
        try:
            cfg = load_agent_config(p)
            self.assertEqual(cfg.default_agent_profile, "researcher")
            self.assertEqual(cfg.get_agent_profile("researcher").policy_profile_id, "deep_research")
            self.assertEqual(cfg.get_agent_profile("researcher").tools.allow, ["load_skill", "search_web"])
            self.assertEqual(
                cfg.get_agent_profile("researcher").skills.roots,
                [".skills", "services/research/.skills"],
            )
            self.assertEqual(
                cfg.get_agent_profile("researcher").skills.denylist,
                ["internal-only-skill"],
            )
            self.assertEqual(
                cfg.tool_catalog.custom_imports,
                ["custom_tools.research:build_tools"],
            )
        finally:
            if p.exists():
                p.unlink()
