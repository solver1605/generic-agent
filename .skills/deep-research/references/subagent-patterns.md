# Subagent Delegation Patterns

## Delegate Only Independent Tasks

Good candidates:
- Independent vendor/product deep-dives
- Parallel country/region regulatory scans
- Separate methodology comparisons

Avoid delegation for:
- Final synthesis and recommendation writing
- Tasks with tight dependency chains

## Task Payload Pattern

Use structured tasks with explicit outputs.

```json
[
  {
    "id": "vendor_a",
    "title": "Vendor A deep dive",
    "objective": "Assess capabilities, pricing model, limitations, and roadmap signals.",
    "constraints": ["Use only sources from last 12 months unless foundational"],
    "expected_output": "Bullet findings with citations and confidence tags",
    "can_run_parallel": true
  }
]
```

## Merge Rules

- Keep deterministic ordering by task id or plan order.
- Merge factual outputs first, then synthesize at supervisor level.
- Preserve per-task uncertainties and citation context.

## Safety Rules

- Do not use subagents for user-approval steps.
- Escalate clarification to supervisor if worker confidence is too low.
