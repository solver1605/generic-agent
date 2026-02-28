from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.config import (
    AgentConfig,
    ModelCard,
    build_llm_from_model_card,
    default_agent_config,
    load_agent_config,
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
