# Release Notes

## 2026-03-01

This release turns GenericAgent into a reusable framework for specialized agents, with profile-driven configuration, package entrypoints, and stronger Streamlit/CLI runtime controls.

## Highlights

- Packaged runtime with public namespace `emergent_planner`.
- Console entrypoints:
  - `generic-agent`
  - `generic-agent-ui`
- YAML-driven specialization via `agent_profiles` for prompts, tools, and skills.
- Profile-scoped skill discovery and enforced allow/deny rules in `load_skill`.
- Streamlit UI improvements:
  - agent profile selection
  - tool toggles and sub-agent controls
  - artifact browser and previews
  - deep debug visibility for sub-agent runs/tool calls
  - configurable app title/tab title from YAML
- New built-in deep-research skill package (repo-tracked): `.skills/deep-research/`.

## New Configuration Surfaces

### Agent profiles and tool catalog

- `default_agent_profile`
- `tool_catalog.allow_module_prefixes`
- `tool_catalog.custom_imports`
- `agent_profiles[]` with:
  - prompt merge/replace overrides
  - tool allow/deny policy
  - skill roots and allow/deny lists

### Streamlit branding (new)

```yaml
streamlit:
  app_name: Research Copilot
  page_title: Research Copilot UI
```

- `app_name` controls the in-app heading.
- `page_title` controls browser tab title.
- Defaults remain:
  - `app_name: Emergent Planner`
  - `page_title: Emergent Planner UI`

## Packaging and Compatibility

- Canonical import path is now:

```python
from emergent_planner import ...
```

- Backward compatibility shim remains for one minor release:
  - `src.emergent_planner` works but emits `DeprecationWarning`.

## Developer Experience

- Added onboarding and specialization docs:
  - `docs/DEVELOPER_ONBOARDING.md`
  - `docs/AGENT_PROFILES.md`
  - `docs/PACKAGING.md`
- Added specialized agent template scaffold:
  - `templates/specialized_agent_template/`

## Runtime and Tooling

- Added/expanded `search_web` stack and integration into agent tool flow.
- Added `spawn_subagents` orchestration with guardrails and deterministic merge behavior.
- Added office output tools and artifact-safe workflows (runtime outputs under `artifacts/`, ignored by git).

## Tests and Quality

- Expanded automated tests for:
  - config/profile parsing and backward compatibility
  - skill discovery/loading and scoping
  - search/sub-agent execution behavior
  - office tools
  - packaging/import/script smoke checks

Current suite passes locally:

- `python -m unittest discover -s tests -p 'test_*.py'`

## Notes for Specialized Agent Authors

1. Build your own tool package and allowlist its module prefix.
2. Define one or more `agent_profiles` in your YAML.
3. Scope skills via `skills.roots` and `skills.allowlist`.
4. Run with explicit config:
   - `generic-agent --config /path/to/agent.yaml --agent-profile <id>`
   - `generic-agent-ui --config /path/to/agent.yaml`
