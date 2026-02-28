import tempfile
from pathlib import Path
from unittest import TestCase

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.emergent_planner.context_manager import ContextManager
from src.emergent_planner.models import PromptCard, PromptLibrary
from src.emergent_planner.policies import BudgetPolicy
from src.emergent_planner.skills import discover_skills, parse_skill_md, render_skills_topk
from src.emergent_planner.utils import compact_tool_message, normalize_content


class TestContextSkillsUtils(TestCase):
    def test_context_manager_injects_active_skill_and_memory_and_core_prompts(self):
        lib = PromptLibrary(
            cards=[
                PromptCard("core", "CORE", {"core"}, priority=0),
                PromptCard("after", "AFTER", {"after_tool"}, priority=10),
                PromptCard("error", "ERROR", {"error"}, priority=20),
            ]
        )
        budget = BudgetPolicy(max_prompt_tokens=4000, reserved_for_generation=500)
        cm = ContextManager(prompt_lib=lib, budget=budget)

        state = {
            "history": [HumanMessage(content="what can you do?")],
            "memory": {"summary": "S", "plan": "P"},
            "runtime": {
                "turn_index": 0,
                "after_tool": True,
                "active_skill_name": "MySkill",
                "active_skill_body": "Do X",
            },
            "skills": [],
        }

        msgs = cm.compose(state)
        texts = [getattr(m, "content", "") for m in msgs if isinstance(m, SystemMessage)]

        self.assertTrue(any("CORE" in t for t in texts))
        self.assertTrue(any("AFTER" in t for t in texts))
        self.assertTrue(any("ACTIVE SKILL: MySkill" in t for t in texts))
        self.assertTrue(any("Conversation summary:" in t for t in texts))
        self.assertTrue(any("Current plan:" in t for t in texts))

    def test_context_manager_compacts_large_tool_messages(self):
        lib = PromptLibrary(cards=[PromptCard("core", "CORE", {"core"}, priority=0)])
        budget = BudgetPolicy(max_prompt_tokens=4000, reserved_for_generation=1000, max_tool_snippet_chars=5)
        cm = ContextManager(prompt_lib=lib, budget=budget)

        state = {
            "history": [
                HumanMessage(content="u"),
                ToolMessage(content="123456789", tool_call_id="tc1"),
            ],
            "memory": {},
            "runtime": {"turn_index": 1},
            "skills": [],
        }
        msgs = cm.compose(state)

        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("...[truncated]...", tool_msgs[0].content)

    def test_context_manager_injects_skills_registry_when_requested(self):
        lib = PromptLibrary(cards=[PromptCard("core", "CORE", {"core"}, priority=0)])
        budget = BudgetPolicy(max_prompt_tokens=4000, reserved_for_generation=1000, max_skills_chars=400)
        cm = ContextManager(prompt_lib=lib, budget=budget)

        skills = [
            parse_skill_md(
                """---
name: SQL Analyst
description: Analyze SQL performance.
---
Body.
""",
                Path("/tmp/sql/SKILL.md"),
            ),
            parse_skill_md(
                """---
name: API Debugger
description: Debug HTTP APIs.
---
Body.
""",
                Path("/tmp/api/SKILL.md"),
            ),
        ]
        for s in skills:
            s.body = None

        state = {
            "history": [HumanMessage(content="what can you do with APIs?")],
            "memory": {},
            "runtime": {"turn_index": 0},
            "skills": skills,
        }
        msgs = cm.compose(state)
        texts = [getattr(m, "content", "") for m in msgs if isinstance(m, SystemMessage)]
        joined = "\n".join(texts)
        self.assertIn("Available skills (load only when needed):", joined)
        self.assertIn("API Debugger", joined)

    def test_parse_skill_md_requires_frontmatter_fields(self):
        bad = "---\nname: X\n---\nBody"
        with self.assertRaises(ValueError):
            parse_skill_md(bad, Path("/tmp/skill.md"))

    def test_discover_skills_scans_directory(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            skill_dir = tmp / "my_skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: Data Cleaner
description: Clean datasets.
---
Use these steps.
""".strip(),
                encoding="utf-8",
            )

            skills = discover_skills(tmp)
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "Data Cleaner")
            self.assertIsNone(skills[0].body)

    def test_render_skills_topk_includes_relevant_skills_and_respects_max_chars(self):
        skills = [
            parse_skill_md(
                """---
name: API Debugger
description: Diagnose HTTP issues.
---
Body
""",
                Path("/tmp/s1/SKILL.md"),
            ),
            parse_skill_md(
                """---
name: SQL Tuner
description: Optimize SQL queries.
---
Body
""",
                Path("/tmp/s2/SKILL.md"),
            ),
        ]
        text = render_skills_topk(skills, "Need API diagnostics", max_chars=120, k=2)
        self.assertTrue(text.startswith("Available skills"))
        self.assertIn("API Debugger", text)

    def test_normalize_content_handles_strings_dicts_and_lists(self):
        self.assertEqual(normalize_content("hello"), "hello")
        self.assertEqual(normalize_content({"text": "hi"}), "hi")
        out = normalize_content([{"type": "text", "text": "a"}, {"content": "b"}, "c"])
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertIn("c", out)

    def test_compact_tool_message_truncates_when_needed(self):
        msg = ToolMessage(content="abcdefghij", tool_call_id="tc")
        compacted = compact_tool_message(msg, max_chars=5)
        self.assertIsInstance(compacted, ToolMessage)
        self.assertIn("...[truncated]...", compacted.content)
        self.assertEqual(compacted.tool_call_id, "tc")
