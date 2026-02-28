# Sub-Agents

## Overview

The supervisor agent can delegate work to dynamically created worker sub-agents using the `spawn_subagents` tool.

- Invocation model: tool call from supervisor
- Worker model: same model card by default
- Parallelism: bounded pool
- Merge strategy: deterministic structured merge into runtime + tool response

## Tool API

```python
spawn_subagents(
  tasks: List[SubAgentTask],
  execution: Optional[SubAgentExecutionConfig] = None,
) -> Dict[str, Any]
```

### `SubAgentTask`
- `id: str`
- `title: str`
- `objective: str`
- `constraints: List[str] = []`
- `expected_output: str`
- `can_run_parallel: bool = True`
- `tool_overrides: Optional[List[str]] = None`

### `SubAgentExecutionConfig`
- `max_workers: int = 4`
- `max_worker_turns: int = 8`
- `max_wall_time_s: float = 45.0`
- `max_retries: int = 1`

## Runtime Merge

After tool execution, runtime is updated with:
- `runtime.subagent_runs` (append-only run records)
- `runtime.subagent_results` (task-id keyed latest result)
- `runtime.subagent_stats`
- `runtime.last_subagent_request_id`

## Safety and Policy

- No recursion: workers cannot call `spawn_subagents`
- Supervisor-only HITL: worker flows reject `verify_with_user`
- Worker tools are selected from supervisor-enabled tools using task-type policy
- Hard limits enforced for workers/turns/wall-time/retries

## Artifacts

Per-task JSON artifacts are persisted to:

`artifacts/subagents/<parent_run_id>/<request_id>/<task_id>.json`

## Config

Configure in `agent_config.yaml` under `subagents`:

- `enabled`
- `max_workers_default`
- `max_worker_turns_default`
- `max_wall_time_s_default`
- `max_retries_default`
- `artifact_dir`
- `tool_policy`:
  - `allow_by_task_type`
  - `denylist`
  - `permitted_overrides`
