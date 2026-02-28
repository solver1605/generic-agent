import json
import tempfile
from pathlib import Path
from unittest import TestCase

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.emergent_planner.nodes import (
    activate_subagent_from_tool_result_node,
    activate_skill_from_tool_result_node,
    has_tool_calls,
    persist_tool_outputs_node,
    should_summarize,
    summarize_node,
)
from src.emergent_planner.policies import SummaryPolicy, ToolLogPolicy


class StubLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        class R:
            pass

        r = R()
        r.content = self.content
        return r


class TestNodes(TestCase):
    def test_should_summarize_respects_policy_threshold(self):
        policy = SummaryPolicy(summarize_when_history_len_exceeds=3, keep_last_n_messages=1)
        state = {"history": [HumanMessage(content="a")] * 4}
        self.assertEqual(should_summarize(state, policy), "summarize")

        state = {"history": [HumanMessage(content="a")] * 3}
        self.assertEqual(should_summarize(state, policy), "skip")

    def test_has_tool_calls_routes_to_tools_only_for_ai_tool_calls(self):
        state = {
            "history": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc1", "name": "read_file", "args": {}}],
                )
            ]
        }
        self.assertEqual(has_tool_calls(state), "tools")

        state = {"history": [AIMessage(content="final answer")]}
        self.assertEqual(has_tool_calls(state), "end")

        state = {"history": [HumanMessage(content="hello")]}
        self.assertEqual(has_tool_calls(state), "end")

    def test_summarize_node_updates_memory_and_trims_history(self):
        llm = StubLLM("- Goals\n- Decisions\n- Tool outcomes\n- Open tasks\n- Constraints")
        policy = SummaryPolicy(summarize_when_history_len_exceeds=2, keep_last_n_messages=2)
        state = {
            "history": [
                HumanMessage(content="u1"),
                AIMessage(content="a1"),
                HumanMessage(content="u2"),
                AIMessage(content="a2"),
            ],
            "memory": {"summary": "old"},
            "runtime": {"run_id": "r1"},
        }

        out = summarize_node(state, llm=llm, policy=policy)

        self.assertIn("memory", out)
        self.assertTrue(out["memory"]["summary"].startswith("- Goals"))
        self.assertEqual(len(out["history"]), 2)
        self.assertTrue(out["runtime"]["summarized"])
        self.assertEqual(len(llm.calls), 1)

    def test_summarize_node_noop_when_below_threshold(self):
        llm = StubLLM("unused")
        policy = SummaryPolicy(summarize_when_history_len_exceeds=10, keep_last_n_messages=2)
        state = {"history": [HumanMessage(content="one")], "memory": {}, "runtime": {}}

        out = summarize_node(state, llm=llm, policy=policy)
        self.assertEqual(out, {})
        self.assertEqual(llm.calls, [])

    def test_persist_tool_outputs_node_truncates_and_persists_large_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            policy = ToolLogPolicy(artifacts_dir=tmp_path / "artifacts", max_inline_chars=10)
            tool_call_id = "abc123"
            state = {
                "history": [
                    HumanMessage(content="q"),
                    ToolMessage(content="0123456789EXTRA", tool_call_id=tool_call_id),
                ],
                "runtime": {"run_id": "run1"},
            }

            out = persist_tool_outputs_node(state, policy)

            self.assertIn("history", out)
            updated_tool = out["history"][1]
            self.assertIsInstance(updated_tool, ToolMessage)
            self.assertIn("...[truncated]...", updated_tool.content)
            self.assertIn("Full tool output saved at:", updated_tool.content)

            persisted = tmp_path / "artifacts" / "run1" / f"{tool_call_id}.txt"
            self.assertTrue(persisted.exists())
            self.assertEqual(persisted.read_text(encoding="utf-8"), "0123456789EXTRA")

    def test_persist_tool_outputs_node_noop_when_small_output(self):
        with tempfile.TemporaryDirectory() as td:
            policy = ToolLogPolicy(artifacts_dir=Path(td) / "artifacts", max_inline_chars=50)
            state = {
                "history": [ToolMessage(content="short", tool_call_id="tc")],
                "runtime": {"run_id": "r"},
            }
            self.assertEqual(persist_tool_outputs_node(state, policy), {})

    def test_activate_skill_from_tool_result_node_sets_runtime_fields(self):
        payload = {
            "name": "skill-x",
            "description": "desc",
            "body": "# SKILL BODY",
            "meta": {"owner": "team"},
        }
        state = {
            "history": [ToolMessage(content=json.dumps(payload), tool_call_id="tc")],
            "runtime": {"run_id": "r"},
        }

        out = activate_skill_from_tool_result_node(state)
        rt = out["runtime"]
        self.assertEqual(rt["active_skill_name"], "skill-x")
        self.assertEqual(rt["active_skill_body"], "# SKILL BODY")
        self.assertEqual(rt["active_skill_meta"]["owner"], "team")

    def test_activate_skill_from_tool_result_node_ignores_non_json_payloads(self):
        state = {"history": [ToolMessage(content="not-json", tool_call_id="tc")], "runtime": {}}
        self.assertEqual(activate_skill_from_tool_result_node(state), {})

    def test_activate_skill_from_tool_result_node_handles_block_content(self):
        payload = {
            "name": "deep-research",
            "description": "desc",
            "body": "Skill body",
            "meta": {"owner": "team"},
        }
        state = {
            "history": [
                ToolMessage(
                    content=[{"type": "text", "text": json.dumps(payload)}],
                    tool_call_id="tc",
                )
            ],
            "runtime": {"run_id": "r"},
        }

        out = activate_skill_from_tool_result_node(state)
        rt = out["runtime"]
        self.assertEqual(rt["active_skill_name"], "deep-research")
        self.assertEqual(rt["active_skill_body"], "Skill body")

    def test_activate_subagent_from_tool_result_node_merges_runtime(self):
        payload = {
            "__tool": "spawn_subagents",
            "request_id": "subreq_1",
            "status": "partial",
            "summary": "done",
            "results": [{"task_id": "t1", "status": "ok", "summary": "s1"}],
            "errors": [{"task_id": "t2", "code": "err"}],
            "stats": {"tasks_completed": 1},
        }
        state = {
            "history": [ToolMessage(content=json.dumps(payload), tool_call_id="tc")],
            "runtime": {},
        }
        out = activate_subagent_from_tool_result_node(state)
        rt = out["runtime"]
        self.assertEqual(rt["last_subagent_request_id"], "subreq_1")
        self.assertEqual(len(rt["subagent_runs"]), 1)
        self.assertIn("t1", rt["subagent_results"])
        self.assertIn("t2", rt["subagent_errors"])
