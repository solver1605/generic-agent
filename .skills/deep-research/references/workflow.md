# Deep Research Workflow

## Phase 1: Quick Report and Approval

1. Define objective:
- Decision to support
- Constraints (time, scope, geography, policy)
- Required confidence level

2. Run quick evidence pass:
- Execute focused `search_web` queries.
- Capture top findings and obvious conflicts.

3. Draft quick report (`artifacts/research/quick_report.md`):
- Objective and context
- Preliminary findings
- Evidence snippets with citations
- Confidence and uncertainty
- Proposed detailed-plan outline

4. Ask for approval with `verify_with_user`:
- Ask one specific go/no-go question.
- Include only enough context to decide quickly.

## Phase 2: Detailed Plan and Long Report

1. Write detailed plan (`artifacts/research/detailed_plan.md`):
- Task list with status
- Parallelizable vs sequential tasks
- Data needed per task

2. Execute tasks:
- Use `spawn_subagents` for independent tasks only.
- Keep dependency-sensitive synthesis in supervisor.

3. Validate findings:
- Reconcile contradictions.
- Check date sensitivity and stale evidence risk.

4. Write long report (`artifacts/research/long_report.md`):
- Follow template exactly.
- Include evidence matrix appendix.

5. Final quality gate:
- Citation coverage complete
- Source quality thresholds met
- Open risks and unknowns explicit
