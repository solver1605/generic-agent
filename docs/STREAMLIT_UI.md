# Streamlit UI

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Environment:
- `GOOGLE_API_KEY` is required.
- Optional: `agent_config.yaml` with model cards (see `agent_config.example.yaml`).
- Optional branding in config:
  - `streamlit.app_name` controls on-page title.
  - `streamlit.page_title` controls browser tab title.

Example:

```yaml
streamlit:
  app_name: Research Copilot
  page_title: Research Copilot UI
```

## Views

The app provides two tabs:
- `User View`: chat-first interface for normal end-user interaction.
- `Debug View`: state/step introspection (history, prompt, runtime, memory, telemetry, diffs).

## User View Behavior

- Send prompts via chat input.
- If the agent triggers `verify_with_user`, the UI shows an interrupt card and blocks new prompts until you answer.
- Submitting the interrupt response resumes graph execution with `Command(resume=answer)`.
- Tool messages can be toggled on/off in the chat display.

## Debug View Behavior

- Shows run metrics (history count, prompt count, steps, turn index).
- Step slider to inspect snapshots captured during graph streaming.
- Tabs:
  - `History`
  - `Prompt`
  - `Runtime`
  - `Memory`
  - `Telemetry`
  - `Diff`

## Session Controls

- Sidebar `Reset Session` button clears in-memory run state and starts a fresh thread.

## Model Configuration

- Sidebar lets you choose a `Model card`, then optionally override:
  - `Model name override`
  - `Thinking budget override`
- Cards are loaded from `agent_config.yaml` if present; otherwise built-in defaults are used.
- Sidebar also exposes `Runtime engine` (`langgraph` or `google_adk`) when allowed by config.

## Tool Controls

- Sidebar shows an `Available Tools` section with per-tool toggles.
- Enabled tools are bound to the agent for the current run.
- `search_web` is included in the default tool catalog.
- `spawn_subagents` is included and can be toggled like any other tool.
- If all tools are disabled, runtime initialization is blocked until at least one is enabled.

## Sub-Agent Controls

- Sidebar provides sub-agent runtime knobs:
  - `Enable sub-agents`
  - `Sub-agent max workers`
  - `Sub-agent max wall time (s)`
- These values are injected into runtime and used by `spawn_subagents`.
- User view renders compact sub-agent run cards with expandable details.
- Debug view includes a `Sub-agents` tab with run/result/stats payloads.
