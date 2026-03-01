# Developer Onboarding

This guide is for teams building specialized agents on top of GenericAgent.

## 1) Prerequisites

- Python 3.10+
- `uv` installed
- A valid `GOOGLE_API_KEY`

## 2) Clone and setup

```bash
cd /Users/rishubhkhurana/Work

git clone <your-git-url>/GenericAgent.git
cd GenericAgent

uv sync
```

## 3) Environment variables

Create `.env` (or export in shell):

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY
```

Shell export alternative:

```bash
export GOOGLE_API_KEY="<your_key>"
```

## 4) Run the runtime

CLI:

```bash
uv run generic-agent --config agent_config.yaml --agent-profile default
```

Streamlit UI:

```bash
uv run generic-agent-ui --config agent_config.yaml
```

Custom UI port:

```bash
uv run generic-agent-ui --config agent_config.yaml -- --server.port 8502
```

## 5) Core extension model

You extend behavior using:

1. Agent profiles (YAML)
2. Custom tools (Python modules)
3. Skills (`SKILL.md` folders)
4. Prompt cards (inline or markdown files)

Main schema docs:
- `docs/AGENT_PROFILES.md`
- `docs/PACKAGING.md`

## 6) Create a specialized agent package

Use the template scaffold:

- `templates/specialized_agent_template/pyproject.toml`
- `templates/specialized_agent_template/config/specialized_agent.yaml`
- `templates/specialized_agent_template/custom_tools/`
- `templates/specialized_agent_template/skills/`
- `templates/specialized_agent_template/prompts/`

Recommended flow:

1. Copy `templates/specialized_agent_template` into a new repo.
2. Update package metadata and Git dependency URL.
3. Implement your custom tools (factory expected by `tool_catalog.custom_imports`).
4. Define `agent_profiles` with model/policy/tool/skill scope.
5. Add skill folders with `SKILL.md` frontmatter.

## 7) Configure agent profiles

Your YAML profile controls:

- `model_card_id`
- `policy_profile_id`
- `streamlit.app_name` and `streamlit.page_title` (UI branding)
- `prompts` (merge/replace cards)
- `tools.allow` and `tools.deny`
- `skills.roots`, `skills.allowlist`, `skills.denylist`

Run with explicit config/profile:

```bash
uv run generic-agent --config /path/to/specialized_agent.yaml --agent-profile researcher
uv run generic-agent-ui --config /path/to/specialized_agent.yaml
```

## 8) Add custom tools safely

In config:

```yaml
tool_catalog:
  allow_module_prefixes: [custom_tools, emergent_planner]
  custom_imports:
    - custom_tools.research:build_tools
```

Rules:

- Import spec format is `module.path:symbol`.
- Module must match `allow_module_prefixes`.
- Symbol can return one tool or list of tools.
- Final tool set must be non-empty after allow/deny filters.

## 9) Add skills

Skill folder format:

```text
.skills/<skill-name>/SKILL.md
```

`SKILL.md` must include frontmatter:

```md
---
name: deep-research
description: Deep evidence-backed research workflow.
---
```

Notes:

- `load_skill` supports underscore/hyphen aliases.
- Skill loading is profile-scoped by roots and allow/deny lists.
- Current shipped repository skill: `deep-research`.

## 10) Testing and quality checks

Run full test suite:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Packaging-focused tests:

```bash
uv run python -m unittest tests/test_packaging.py
```

Optional compile smoke check:

```bash
uv run python -m py_compile main.py streamlit_app.py src/emergent_planner/cli.py src/emergent_planner/ui.py
```

## 11) Packaging and import paths

Public namespace:

```python
from emergent_planner import build_app, load_agent_config
```

Entry points:

- `generic-agent`
- `generic-agent-ui`

Compatibility:

- `src.emergent_planner` still works temporarily (deprecated).

## 12) Common troubleshooting

`GOOGLE_API_KEY not set`
- Set key in `.env` or export in shell.

`No tools enabled after profile policy`
- Check `tools.allow` / `tools.deny` and UI tool toggles.

`Skill not found`
- Verify `SKILL.md` path and profile `skills.roots` / allowlist / denylist.

`Rejected custom tool import`
- Add module prefix to `tool_catalog.allow_module_prefixes`.

`Streamlit config not picked`
- Launch with explicit `--config` on `generic-agent-ui`.

## 13) Suggested day-1 checklist

1. Run default profile in CLI.
2. Run Streamlit UI and verify tool toggles.
3. Create one custom tool and load it via profile YAML.
4. Add one skill and load it with `load_skill`.
5. Run tests before opening PR.
