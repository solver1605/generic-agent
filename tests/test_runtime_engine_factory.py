from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.config import default_agent_config
from src.emergent_planner.models import PromptCard, PromptLibrary
from src.emergent_planner.runtime.adk_engine import ADKRuntimeEngine
from src.emergent_planner.runtime.factory import build_runtime_app, resolve_runtime_engine
from src.emergent_planner.runtime.langgraph_engine import LangGraphRuntimeEngine


class _FakeApp:
    def __init__(self):
        self.calls = []

    def stream(self, input_obj, *, config=None, stream_mode=None):
        self.calls.append({"input_obj": input_obj, "config": config, "stream_mode": stream_mode})
        yield {
            "history": [],
            "messages": [],
            "runtime": {"turn_index": 1},
            "memory": {},
            "telemetry": [],
            "__interrupt__": None,
        }


class TestRuntimeEngineFactory(TestCase):
    def _prompt_lib(self) -> PromptLibrary:
        return PromptLibrary(cards=[PromptCard(name="guidelines", text="x", tags={"core"}, priority=10)])

    def test_resolve_runtime_engine_precedence(self):
        cfg = default_agent_config()
        self.assertEqual(
            resolve_runtime_engine(cfg=cfg, profile_runtime_engine="google_adk", explicit_runtime_engine="langgraph"),
            "langgraph",
        )
        self.assertEqual(
            resolve_runtime_engine(cfg=cfg, profile_runtime_engine="google_adk", explicit_runtime_engine=None),
            "google_adk",
        )
        self.assertEqual(
            resolve_runtime_engine(cfg=cfg, profile_runtime_engine=None, explicit_runtime_engine=None),
            "langgraph",
        )

    def test_resolve_runtime_engine_rejects_disallowed(self):
        cfg = default_agent_config()
        cfg.runtime.allowed_engines = ["langgraph"]
        with self.assertRaises(ValueError):
            resolve_runtime_engine(cfg=cfg, explicit_runtime_engine="google_adk")

    def test_build_runtime_app_langgraph(self):
        fake_app = _FakeApp()
        with patch("src.emergent_planner.runtime.factory.build_langgraph_graph", return_value=fake_app):
            runtime = build_runtime_app(
                llm=object(),
                prompt_lib=self._prompt_lib(),
                skills_root=Path(".skills"),
                tools=[],
                engine="langgraph",
                cfg=default_agent_config(),
            )

        self.assertIsInstance(runtime, LangGraphRuntimeEngine)
        states = list(runtime.stream({"history": []}, config={"configurable": {"thread_id": "t1"}}))
        self.assertEqual(states[-1]["runtime"]["turn_index"], 1)
        self.assertEqual(fake_app.calls[-1]["stream_mode"], "values")

    def test_build_runtime_app_google_adk_disabled(self):
        cfg = default_agent_config()
        cfg.adk.enabled = False
        fake_app = _FakeApp()
        with patch("src.emergent_planner.runtime.factory.build_langgraph_graph", return_value=fake_app):
            with self.assertRaises(RuntimeError):
                build_runtime_app(
                    llm=object(),
                    prompt_lib=self._prompt_lib(),
                    skills_root=Path(".skills"),
                    tools=[],
                    engine="google_adk",
                    cfg=cfg,
                )

    def test_build_runtime_app_google_adk_missing_dependency(self):
        cfg = default_agent_config()
        cfg.adk.enabled = True
        fake_app = _FakeApp()
        with patch("src.emergent_planner.runtime.factory.build_langgraph_graph", return_value=fake_app), \
             patch("src.emergent_planner.runtime.adk_engine.importlib.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaises(RuntimeError) as ctx:
                build_runtime_app(
                    llm=object(),
                    prompt_lib=self._prompt_lib(),
                    skills_root=Path(".skills"),
                    tools=[],
                    engine="google_adk",
                    cfg=cfg,
                )
        self.assertIn("google_adk runtime selected", str(ctx.exception))

    def test_adk_engine_parity_contract_shape(self):
        cfg = default_agent_config()
        cfg.adk.enabled = True
        fake_app = _FakeApp()
        with patch("src.emergent_planner.runtime.factory.build_langgraph_graph", return_value=fake_app), \
             patch("src.emergent_planner.runtime.adk_engine.importlib.import_module", return_value=object()):
            runtime = build_runtime_app(
                llm=object(),
                prompt_lib=self._prompt_lib(),
                skills_root=Path(".skills"),
                tools=[],
                engine="google_adk",
                cfg=cfg,
            )

        self.assertIsInstance(runtime, ADKRuntimeEngine)
        states = list(runtime.stream({"history": []}, config={"configurable": {"thread_id": "t2"}}))
        self.assertTrue(states)
        st = states[-1]
        for k in ["history", "messages", "runtime", "memory", "telemetry", "__interrupt__"]:
            self.assertIn(k, st)

