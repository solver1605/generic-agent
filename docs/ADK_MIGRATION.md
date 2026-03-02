# ADK Migration Guide

## Overview

GenericAgent supports dual runtime engines:
- `langgraph` (default)
- `google_adk` (opt-in)

Both engines expose the same runtime contract to CLI/Streamlit:
- state keys: `history`, `messages`, `runtime`, `memory`, `telemetry`, `__interrupt__`
- interrupt payload shape for `verify_with_user`
- same tool and sub-agent merge semantics

## Enable ADK Runtime

In `agent_config.yaml`:

```yaml
runtime:
  default_engine: langgraph
  allowed_engines: [langgraph, google_adk]

adk:
  enabled: true
  timeout_s: 30.0
  max_steps: 64
```

Install optional dependencies:

```bash
uv sync --extra adk
# or
pip install 'generic-agent-runtime[adk]'
```

## Select Runtime

Precedence order:
1. CLI/UI override
2. `agent_profiles[].runtime_engine`
3. `runtime.default_engine`

CLI examples:

```bash
generic-agent --config agent_config.yaml --runtime-engine langgraph
generic-agent --config agent_config.yaml --runtime-engine google_adk
```

Streamlit:
- use sidebar `Runtime engine` selector.

## Current Migration Phase

Current `google_adk` adapter validates ADK dependency/config and preserves strict runtime contract compatibility while dual-runtime wiring stabilizes.

## Compatibility Window

LangGraph remains supported for two minor releases after ADK parity target is reached.

## Troubleshooting

`google_adk runtime selected but disabled in config`
- set `adk.enabled: true`.

`google_adk runtime selected but Google ADK is not installed`
- install ADK extra as shown above.
