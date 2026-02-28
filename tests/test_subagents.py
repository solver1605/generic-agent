import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.config import default_agent_config
from src.emergent_planner.subagents.context import build_worker_initial_state
from src.emergent_planner.subagents.orchestrator import run_subagents
from src.emergent_planner.subagents.policy import infer_task_type, resolve_worker_tool_names
from src.emergent_planner.subagents.types import SubAgentExecutionConfig, SubAgentTask
from src.emergent_planner.tools import spawn_subagents


class TestSubagents(TestCase):
    def test_infer_task_type(self):
        t = SubAgentTask(id="1", title="Deep research", objective="Investigate vendor options", expected_output="table")
        self.assertEqual(infer_task_type(t), "research")

    def test_resolve_worker_tool_names_applies_policy_and_denylist(self):
        cfg = default_agent_config().subagents.tool_policy
        task = SubAgentTask(
            id="1",
            title="analysis",
            objective="Analyze data",
            expected_output="findings",
            tool_overrides=["read_file", "write_file"],
        )
        names = resolve_worker_tool_names(
            task,
            supervisor_enabled=["read_file", "read_file_range", "python_repl", "write_file", "spawn_subagents"],
            policy=cfg,
        )
        self.assertIn("python_repl", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("spawn_subagents", names)

    def test_build_worker_initial_state_scopes_context(self):
        parent_state = {
            "history": [],
            "memory": {"summary": "long summary", "plan": "1. task"},
            "runtime": {"model_card_id": "gemini_flash_fast", "model_name": "m", "thinking_budget": 0},
            "skills": [],
        }
        task = SubAgentTask(id="t1", title="Research", objective="Find APIs", expected_output="links")
        st = build_worker_initial_state(
            parent_state,
            task,
            worker_run_id="w1",
            parent_run_id="p1",
            request_id="r1",
            subagent_depth=1,
            tool_names=["search_web"],
        )
        self.assertTrue(st["runtime"]["subagent_mode"])
        self.assertEqual(st["runtime"]["subagent_depth"], 1)
        self.assertEqual(st["runtime"]["enabled_tool_names"], ["search_web"])

    def test_orchestrator_blocks_recursion(self):
        t = SubAgentTask(id="t1", title="Research", objective="Find APIs", expected_output="links")
        record = run_subagents(
            tasks=[t],
            execution=SubAgentExecutionConfig(),
            parent_state={"runtime": {"run_id": "p1", "subagent_depth": 1}},
            all_tools=[],
            config_path=Path("/tmp/nonexistent-config.yaml"),
        )
        self.assertEqual(record.status, "failed")
        self.assertTrue(any(e.code == "subagent_recursion_blocked" for e in record.errors))

    def test_spawn_subagents_tool_contract(self):
        fake_record = {
            "request_id": "subreq_x",
            "parent_run_id": "p1",
            "status": "success",
            "summary": "ok",
            "results": [{"task_id": "t1", "status": "ok", "summary": "done"}],
            "errors": [],
            "stats": {"tasks_completed": 1},
        }

        class _R:
            def to_dict(self):
                return fake_record

        with patch("src.emergent_planner.tools.run_subagents", return_value=_R()):
            out = spawn_subagents.invoke(
                {
                    "tasks": [
                        {
                            "id": "t1",
                            "title": "Research",
                            "objective": "Find options",
                            "expected_output": "summary",
                        }
                    ]
                }
            )

        self.assertEqual(out["__tool"], "spawn_subagents")
        self.assertEqual(out["request_id"], "subreq_x")
        self.assertIn("results", out)

    def test_orchestrator_partial_success(self):
        task_ok = SubAgentTask(id="ok1", title="Research", objective="Find one source", expected_output="summary")
        task_bad = SubAgentTask(id="bad1", title="Analyze", objective="Compute metric", expected_output="number")

        class _Succ:
            output = "done"
            summary = "done"
            turns_used = 2
            worker_run_id = "w1"
            timings_ms = {"total": 1}

        class _FailErr:
            task_id = "bad1"
            code = "worker_execution_failed"
            message = "boom"
            retryable = False
            attempts = 1

        class _Fail:
            error = _FailErr()

        def fake_run_worker_task_once(**kwargs):
            if kwargs["task"].id == "ok1":
                return _Succ(), None
            return None, _Fail()

        with patch("src.emergent_planner.subagents.orchestrator.run_worker_task_once", side_effect=fake_run_worker_task_once), \
             patch("src.emergent_planner.subagents.orchestrator.persist_task_artifact", return_value=Path("/tmp/a.json")):
            rec = run_subagents(
                tasks=[task_ok, task_bad],
                execution=SubAgentExecutionConfig(max_workers=2, max_worker_turns=3, max_wall_time_s=20, max_retries=0),
                parent_state={"runtime": {"run_id": "p1", "enabled_tool_names": ["read_file", "search_web"]}},
                all_tools=[],
                config_path=Path("/tmp/nonexistent-config.yaml"),
            )

        self.assertEqual(rec.status, "partial")
        self.assertEqual(len(rec.results), 1)
        self.assertEqual(len(rec.errors), 1)
