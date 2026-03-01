# Agent Profiles

`Emergent Planner` supports YAML-configurable agent profiles so specialized agents can override prompts, tools, and skill scope without changing core code.

## High-level

Config can define:
- global model cards / policy profiles / search / subagent settings
- global tool catalog settings (`tool_catalog`)
- one or more `agent_profiles`

Runtime (CLI + Streamlit) selects one active `agent_profile` per session.

## New config sections

## `tool_catalog`

```yaml
tool_catalog:
  allow_module_prefixes: [emergent_planner, custom_tools]
  custom_imports:
    - custom_tools.research:build_tools
```

- `custom_imports` format: `module.path:symbol`
- imported module must match one of `allow_module_prefixes`

## `agent_profiles`

```yaml
agent_profiles:
  - id: research_assistant
    model_card_id: gemini_flash_reasoning
    policy_profile_id: deep_research
    prompts:
      strategy: merge
      disable_cards: [guidelines]
      cards:
        - name: guidelines
          tags: [core]
          priority: 10
          file: prompts/research_guidelines.md
    tools:
      allow: [load_skill, search_web, spawn_subagents, verify_with_user]
      deny: [python_repl]
    skills:
      roots: [.skills, services/research/.skills]
      allowlist: [deep-research]
```

Fields:
- `model_card_id`: optional override for default model card
- `policy_profile_id`: optional override for default runtime policy profile
- `prompts.strategy`: `merge` or `replace`
- `prompts.cards`: inline text or `file` reference (exactly one)
- `tools.allow`: optional tool allowlist (empty means all catalog tools)
- `tools.deny`: hard denylist
- `skills.roots`: roots scanned for `SKILL.md`
- `skills.allowlist`: optional skill-name allowlist
- `skills.denylist`: optional skill-name denylist (applied after allowlist)

## Prompt override behavior

- `merge`: start from default prompt cards, override by card name, append new cards, remove `disable_cards`
- `replace`: use only profile-declared cards

## Tool resolution behavior

1. Build tool catalog from built-ins + custom imports.
2. Apply profile allowlist (if provided).
3. Apply profile denylist.
4. Apply session toggles/CLI tool switches.

If zero tools remain, runtime fails with a clear error.

## Skill scope behavior

- skill discovery is scoped to `skills.roots`
- `load_skill` enforces runtime scope and optional allowlist
- aliases like `deep_research` and `deep-research` are normalized

## Backward compatibility

Legacy configs without `agent_profiles` are auto-wrapped to a synthetic default profile:
- id: `default`
- model/policy from legacy defaults
- tool policy: allow all built-in catalog tools
- skills roots: `.skills`

## Runtime selectors

- CLI: `--agent-profile <id>`
- Streamlit: sidebar `Agent profile` dropdown

Both default to `default_agent_profile` from config.

## As a dependency

Install runtime + plugin:

```bash
pip install "generic-agent-runtime @ git+ssh://<git-host>/<org>/GenericAgent.git@v0.1.0"
pip install "my-specialized-agent @ git+ssh://<git-host>/<org>/my-specialized-agent.git@v0.1.0"
```

Run with a specialized profile config:

```bash
generic-agent --config /path/to/specialized_agent.yaml --agent-profile researcher
generic-agent-ui --config /path/to/specialized_agent.yaml
```
