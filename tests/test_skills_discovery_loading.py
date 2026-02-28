import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase

from src.emergent_planner.skills import discover_skills
from src.emergent_planner.tools import load_skill


SKILL_TEXT = """---
name: deep-research
description: deep research skill
---
Body content.
"""


class TestSkillsDiscoveryLoading(TestCase):
    def test_discover_skills_falls_back_to_project_root_recursive_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            (root / ".skills" / "deep-research").mkdir(parents=True)
            (root / ".skills" / "deep-research" / "SKILL.md").write_text(SKILL_TEXT, encoding="utf-8")

            # Move into a subdirectory that does not contain .skills.
            sub = root / "src" / "pkg"
            sub.mkdir(parents=True)
            old = Path.cwd()
            try:
                os.chdir(sub)
                skills = discover_skills(Path(".skills"))
            finally:
                os.chdir(old)

            names = [s.name for s in skills]
            self.assertIn("deep-research", names)

    def test_load_skill_resolves_nested_skill_from_project_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            (root / ".skills" / "research" / "deep-research").mkdir(parents=True)
            (root / ".skills" / "research" / "deep-research" / "SKILL.md").write_text(
                SKILL_TEXT,
                encoding="utf-8",
            )

            sub = root / "apps" / "agent"
            sub.mkdir(parents=True)
            old = Path.cwd()
            try:
                os.chdir(sub)
                out = load_skill.invoke({"skill_name": "deep-research"})
            finally:
                os.chdir(old)

            payload = json.loads(out)
            self.assertEqual(payload["name"], "deep-research")
            self.assertIn("Body content", payload["body"])

    def test_load_skill_supports_underscore_hyphen_aliases(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()
            (root / ".skills" / "deep-research").mkdir(parents=True)
            (root / ".skills" / "deep-research" / "SKILL.md").write_text(
                SKILL_TEXT,
                encoding="utf-8",
            )

            old = Path.cwd()
            try:
                os.chdir(root)
                out = load_skill.invoke({"skill_name": "deep_research"})
            finally:
                os.chdir(old)

            payload = json.loads(out)
            self.assertEqual(payload["name"], "deep-research")

    def test_discover_skills_in_multiple_skills_roots_and_load_presented(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".git").mkdir()

            # Top-level skills root.
            (root / ".skills" / "alpha").mkdir(parents=True)
            (root / ".skills" / "alpha" / "SKILL.md").write_text(
                """---
name: alpha
description: alpha skill
---
Alpha body.
""",
                encoding="utf-8",
            )

            # Nested skills root.
            (root / "services" / "api" / ".skills" / "beta").mkdir(parents=True)
            (root / "services" / "api" / ".skills" / "beta" / "SKILL.md").write_text(
                """---
name: beta
description: beta skill
---
Beta body.
""",
                encoding="utf-8",
            )

            old = Path.cwd()
            try:
                os.chdir(root / "services" / "api")
                discovered = discover_skills(Path(".skills"))
                names = sorted([s.name for s in discovered])

                # Contract: skills visible to discovery should include all project .skills roots.
                self.assertEqual(names, ["alpha", "beta"])

                # Contract: every discovered/presented skill is loadable.
                for n in names:
                    raw = load_skill.invoke({"skill_name": n})
                    payload = json.loads(raw)
                    self.assertEqual(payload["name"], n)
                    self.assertTrue((payload.get("body") or "").strip())
            finally:
                os.chdir(old)
