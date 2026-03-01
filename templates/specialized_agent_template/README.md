# Specialized Agent Template

1. Copy this folder to a new repo.
2. Replace Git URLs and package metadata in `pyproject.toml`.
3. Add your custom tools under `custom_tools/`.
4. Update `config/specialized_agent.yaml` for your prompts/tools/skills.
5. Run with:

```bash
generic-agent --config /path/to/config/specialized_agent.yaml --agent-profile researcher
generic-agent-ui --config /path/to/config/specialized_agent.yaml
```
