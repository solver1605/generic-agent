# Packaging and Consumption

Start here for full setup and extension workflow:
- [`docs/DEVELOPER_ONBOARDING.md`](/Users/rishubhkhurana/Work/GenericAgent/docs/DEVELOPER_ONBOARDING.md)

## Public package namespace

Use `emergent_planner` as the public import path:

```python
from emergent_planner import build_app, load_agent_config
```

Legacy `src.emergent_planner` remains available for one minor release and emits a `DeprecationWarning`.

## Install from Git

```bash
pip install "generic-agent-runtime @ git+ssh://<git-host>/<org>/GenericAgent.git@v0.1.0"
```

## Console entrypoints

- `generic-agent` -> package CLI
- `generic-agent-ui` -> Streamlit launcher

Examples:

```bash
generic-agent --config /path/to/agent.yaml --agent-profile default
generic-agent-ui --config /path/to/agent.yaml
```

## Specialized agent plugin model

A specialized agent package should provide:

1. Tool modules imported via `tool_catalog.custom_imports`
2. Profile YAML files (`agent_profiles`)
3. Skill roots with `SKILL.md`
4. Optional prompt markdown files referenced by prompt cards

Use `tool_catalog.allow_module_prefixes` to explicitly allow plugin tool modules.
Use the starter scaffold in [`templates/specialized_agent_template`](/Users/rishubhkhurana/Work/GenericAgent/templates/specialized_agent_template/README.md) to bootstrap quickly.

## Versioning and compatibility

Tag releases and keep plugin/runtime compatibility documented in plugin repos.

| Runtime version | Plugin expectation |
| --- | --- |
| `0.1.x` | Supports both `emergent_planner` and deprecated `src.emergent_planner` imports |
| `0.2.x` | Remove `src.emergent_planner` shim; plugins must import `emergent_planner` |
