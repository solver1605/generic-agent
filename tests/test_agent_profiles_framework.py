import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

from src.emergent_planner.config import (
    AgentProfileConfig,
    ProfileToolPolicyConfig,
    PromptCardConfig,
    PromptConfig,
    SkillsProfileConfig,
    default_agent_config,
)
from src.emergent_planner.prompt_loader import build_prompt_lib_for_profile
from src.emergent_planner.skills import discover_skills_in_roots
from src.emergent_planner.tool_loader import build_tool_catalog, resolve_tools_for_profile
from src.emergent_planner.tool_registry import tool_name
from src.emergent_planner.tools import DEFAULT_TOOLS, load_skill


class TestAgentProfilesFramework(TestCase):
    def test_prompt_loader_merge_overrides_card_by_name(self):
        cfg = default_agent_config()
        profile = AgentProfileConfig(
            id="p1",
            prompts=PromptConfig(
                strategy="merge",
                cards=[
                    PromptCardConfig(
                        name="guidelines",
                        tags=["core"],
                        priority=10,
                        text="CUSTOM GUIDELINES",
                    )
                ],
            ),
            tools=ProfileToolPolicyConfig(),
            skills=SkillsProfileConfig(roots=[".skills"]),
        )
        lib = build_prompt_lib_for_profile(cfg, profile, config_dir=Path.cwd())
        by_name = {c.name: c for c in lib.cards}
        self.assertEqual(by_name["guidelines"].text, "CUSTOM GUIDELINES")

    def test_prompt_loader_supports_file_card_and_disable(self):
        cfg = default_agent_config()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts" / "custom.md").write_text("FILE PROMPT", encoding="utf-8")
            profile = AgentProfileConfig(
                id="p1",
                prompts=PromptConfig(
                    strategy="merge",
                    disable_cards=["planning"],
                    cards=[
                        PromptCardConfig(
                            name="guidelines",
                            tags=["core"],
                            priority=10,
                            file="prompts/custom.md",
                        )
                    ],
                ),
                tools=ProfileToolPolicyConfig(),
                skills=SkillsProfileConfig(roots=[".skills"]),
            )
            lib = build_prompt_lib_for_profile(cfg, profile, config_dir=root)
            names = [c.name for c in lib.cards]
            self.assertNotIn("planning", names)
            self.assertEqual({c.name: c for c in lib.cards}["guidelines"].text, "FILE PROMPT")

    def test_prompt_loader_rejects_both_text_and_file(self):
        cfg = default_agent_config()
        profile = AgentProfileConfig(
            id="p1",
            prompts=PromptConfig(
                strategy="merge",
                cards=[
                    PromptCardConfig(
                        name="guidelines",
                        tags=["core"],
                        priority=10,
                        text="a",
                        file="b.md",
                    )
                ],
            ),
            tools=ProfileToolPolicyConfig(),
            skills=SkillsProfileConfig(roots=[".skills"]),
        )
        with self.assertRaises(ValueError):
            build_prompt_lib_for_profile(cfg, profile, config_dir=Path.cwd())

    def test_tool_loader_imports_custom_tools_with_allowlisted_prefix(self):
        cfg = default_agent_config()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg = root / "custom_tools"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "research.py").write_text(
                """
def custom_ping(x='ok'):
    return x

def build_tools():
    return [custom_ping]
""".strip(),
                encoding="utf-8",
            )
            sys.path.insert(0, root.as_posix())
            self.addCleanup(lambda: sys.path.remove(root.as_posix()) if root.as_posix() in sys.path else None)

            cfg.tool_catalog.allow_module_prefixes = ["custom_tools"]
            cfg.tool_catalog.custom_imports = ["custom_tools.research:build_tools"]
            catalog = build_tool_catalog(cfg, DEFAULT_TOOLS)
            names = [tool_name(t) for t in catalog]
            self.assertIn("custom_ping", names)

    def test_tool_loader_rejects_disallowed_module_prefix(self):
        cfg = default_agent_config()
        cfg.tool_catalog.allow_module_prefixes = ["src.emergent_planner"]
        cfg.tool_catalog.custom_imports = ["custom_tools.research:build_tools"]
        with self.assertRaises(ValueError):
            build_tool_catalog(cfg, DEFAULT_TOOLS)

    def test_resolve_tools_for_profile_applies_allow_then_deny(self):
        def alpha():
            return None

        def beta():
            return None

        profile = AgentProfileConfig(
            id="p",
            tools=ProfileToolPolicyConfig(allow=["alpha", "beta"], deny=["beta"]),
            skills=SkillsProfileConfig(roots=[".skills"]),
        )
        resolved = resolve_tools_for_profile([alpha, beta], profile, extra_allow=["alpha"])
        self.assertEqual([tool_name(t) for t in resolved], ["alpha"])

    def test_resolve_tools_for_profile_raises_on_empty_result(self):
        def alpha():
            return None

        profile = AgentProfileConfig(
            id="p",
            tools=ProfileToolPolicyConfig(allow=["alpha"], deny=["alpha"]),
            skills=SkillsProfileConfig(roots=[".skills"]),
        )
        with self.assertRaises(ValueError):
            resolve_tools_for_profile([alpha], profile)

    def test_discover_skills_in_roots_strict_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s1 = root / "skills_one" / "alpha"
            s2 = root / "skills_two" / "beta"
            s1.mkdir(parents=True)
            s2.mkdir(parents=True)
            (s1 / "SKILL.md").write_text(
                """---
name: alpha
description: alpha skill
---
A
""".strip(),
                encoding="utf-8",
            )
            (s2 / "SKILL.md").write_text(
                """---
name: beta
description: beta skill
---
B
""".strip(),
                encoding="utf-8",
            )
            out = discover_skills_in_roots([s1.parent], strict_scope=True)
            self.assertEqual([s.name for s in out], ["alpha"])

    def test_load_skill_enforces_runtime_allowlist_and_roots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "skills" / "alpha-skill").mkdir(parents=True)
            (root / "skills" / "alpha-skill" / "SKILL.md").write_text(
                """---
name: alpha-skill
description: alpha skill
---
Alpha body
""".strip(),
                encoding="utf-8",
            )
            (root / "skills" / "beta-skill").mkdir(parents=True)
            (root / "skills" / "beta-skill" / "SKILL.md").write_text(
                """---
name: beta-skill
description: beta skill
---
Beta body
""".strip(),
                encoding="utf-8",
            )

            state = {
                "runtime": {
                    "skills_roots_resolved": [(root / "skills").as_posix()],
                    "skills_allowlist_norm": ["alpha-skill"],
                    "skills_denylist_norm": [],
                }
            }

            raw = load_skill.func("alpha_skill", state=state)
            payload = json.loads(raw)
            self.assertEqual(payload["name"], "alpha-skill")

            with self.assertRaises(FileNotFoundError):
                load_skill.func("beta-skill", state=state)

    def test_load_skill_enforces_runtime_denylist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "skills" / "blocked-skill").mkdir(parents=True)
            (root / "skills" / "blocked-skill" / "SKILL.md").write_text(
                """---
name: blocked-skill
description: blocked
---
Body
""".strip(),
                encoding="utf-8",
            )
            state = {
                "runtime": {
                    "skills_roots_resolved": [(root / "skills").as_posix()],
                    "skills_allowlist_norm": [],
                    "skills_denylist_norm": ["blocked-skill"],
                }
            }
            with self.assertRaises(FileNotFoundError):
                load_skill.func("blocked_skill", state=state)
