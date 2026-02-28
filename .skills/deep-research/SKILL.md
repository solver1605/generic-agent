---
name: deep-research
description: Conduct two-phase deep research for complex questions requiring multi-source synthesis, tradeoff analysis, and evidence-backed recommendations. Use when tasks need a quick validated report first, then a detailed long-form report after user approval. Use search_web for source discovery, enforce strict citation and source-quality checks, and delegate independent subtasks with spawn_subagents when parallel execution improves coverage and speed.
---
# Deep Research

Execute rigorous research in two phases: quick validation and detailed reporting.

## Workflow

1. Clarify objective and constraints:
- Restate objective, decision context, constraints, and expected output format.
- Ask one precise clarification question only when a critical unknown blocks quality.

2. Generate a quick report:
- Gather high-signal initial evidence with `search_web`.
- Prioritize authoritative and recent sources where recency matters.
- Save concise quick report to `artifacts/research/quick_report.md`.
- Include: objective, preliminary findings, key evidence, confidence level, major gaps, and proposed detailed plan outline.

3. Gate with user approval:
- Call `verify_with_user` with short context from the quick report.
- If not approved, incorporate requested changes and repeat quick-report gate.
- If approved, proceed to detailed plan and full execution.

4. Create detailed plan:
- Save execution plan to `artifacts/research/detailed_plan.md`.
- Split work into subproblems and mark independent tasks that can run in parallel.
- Use `spawn_subagents` only for independent tasks with clear expected outputs.
- Keep dependent and synthesis tasks in the supervisor flow.

5. Execute detailed research:
- Use `search_web` with explicit recency controls when topic is time-sensitive.
- Cross-check claims across multiple independent sources.
- Track contradictory evidence explicitly and preserve uncertainty.
- Merge subagent outputs deterministically into the main analysis.

6. Produce final long report:
- Save final report to `artifacts/research/long_report.md`.
- Use the required section structure from `references/report-template.md`.

7. Run quality gates before finalizing:
- Ensure non-trivial claims have citations.
- Verify recency where relevant.
- Prefer high-quality sources; flag weaker sources clearly.
- Surface open questions, risks, and confidence boundaries.

## Required Outputs

Always produce these artifacts:
- `artifacts/research/quick_report.md`
- `artifacts/research/detailed_plan.md`
- `artifacts/research/long_report.md`

## Tooling Policy

- Use `search_web` for discovery, comparison, and evidence gathering.
- Use `spawn_subagents` only when tasks are independent and parallelizable.
- Use `verify_with_user` between quick and detailed phases.

## Source and Evidence Standards

Follow `references/source-quality.md`:
- Rank source reliability.
- Prefer primary/official sources.
- Preserve publication date context.
- Include claim-to-evidence mapping in appendix.

## Subagent Delegation Rules

Follow `references/subagent-patterns.md`:
- Delegate only independent subtasks.
- Provide explicit objectives, constraints, and expected outputs.
- Synthesize results centrally in the supervisor output.

## References

- Workflow details: `references/workflow.md`
- Report structure: `references/report-template.md`
- Source quality rules: `references/source-quality.md`
- Subagent usage patterns: `references/subagent-patterns.md`
