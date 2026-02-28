"""
main.py — Example entry point for the Emergent Planner agent.

Usage:
    python main.py

Requires GOOGLE_API_KEY (or OPENAI_API_KEY) in environment or .env file.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()  # loads .env if present

SKILLS_ROOT = Path(".skills")


def on_interrupt(payload, full_state):
    """Default HITL handler: print the question and prompt the user."""
    print("\n" + "=" * 60)
    print("⏸️  AGENT PAUSED — Human input required")
    print("=" * 60)
    kind     = payload.get("kind", payload.get("type", "confirm"))
    question = payload.get("question", str(payload))
    context  = payload.get("context", "")
    choices  = payload.get("choices", [])
    default  = payload.get("default", "")

    if context:
        print(f"\nContext:\n{context}\n")
    print(f"Question: {question}")

    if kind == "pick_one" and choices:
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        raw = input("Enter number or answer: ").strip()
        try:
            idx = int(raw) - 1
            return choices[idx]
        except (ValueError, IndexError):
            return raw
    elif kind == "confirm":
        raw = input(f"Answer [y/n]{f' (default: {default})' if default else ''}: ").strip().lower()
        return raw or default or "yes"
    else:
        return input("Answer: ").strip() or default or "yes"


def main():
    # 1. Import here to show the optional lazy-import pattern
    from src.emergent_planner import (
        DEFAULT_TOOLS,
        build_app,
        discover_skills,
        make_default_prompt_lib,
    )
    from src.emergent_planner.config import build_llm_from_model_card, load_agent_config
    from src.emergent_planner.debug_ui import record_run
    from src.emergent_planner.policies import BudgetPolicy, SummaryPolicy, ToolLogPolicy

    # 2. Set up the LLM
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Create a .env file with GOOGLE_API_KEY=your-key-here"
        )

    cfg = load_agent_config(Path("agent_config.yaml"))
    scfg = getattr(cfg, "subagents", None)
    subagent_enabled = bool(getattr(scfg, "enabled", True))
    subagent_max_workers = int(getattr(scfg, "max_workers_default", 4))
    subagent_max_wall_time_s = float(getattr(scfg, "max_wall_time_s_default", 45.0))
    model_card_id = os.environ.get("MODEL_CARD")
    selected_card = cfg.get_model_card(model_card_id)

    # Backward-compatible model override if MODEL_NAME is set.
    model_name_override = os.environ.get("MODEL_NAME", "").strip()
    if model_name_override:
        selected_card = replace(selected_card, model_name=model_name_override)

    base_llm = build_llm_from_model_card(selected_card, google_api_key=google_api_key)
    tools = DEFAULT_TOOLS
    llm_with_tools = base_llm.bind_tools(tools)

    print(
        "Using model card: "
        f"{selected_card.id} "
        f"(provider={selected_card.provider}, model={selected_card.model_name}, "
        f"thinking_budget={selected_card.thinking_budget})"
    )

    # 3. Discover skills
    skills = discover_skills(SKILLS_ROOT)
    print(f"Discovered {len(skills)} skill(s) from {SKILLS_ROOT}")

    # 4. Build the agent app
    prompt_lib    = make_default_prompt_lib()
    budget_policy = BudgetPolicy()
    tool_policy   = ToolLogPolicy()
    summary_policy = SummaryPolicy()

    app = build_app(
        llm=llm_with_tools,
        prompt_lib=prompt_lib,
        skills_root=SKILLS_ROOT,
        budget_policy=budget_policy,
        tool_log_policy=tool_policy,
        summary_policy=summary_policy,
        tools=tools,
    )

    # 5. Initial state
    run_id = str(uuid.uuid4())
    state = {
        "history": [HumanMessage(content="Hello! Can you tell me what tools and skills you have available?")],
        "memory": {},
        "runtime": {
            "run_id": run_id,
            "turn_index": 0,
            "model_card_id": selected_card.id,
            "model_name": selected_card.model_name,
            "thinking_budget": selected_card.thinking_budget,
            "enabled_tool_names": [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools],
            "subagent_enabled": subagent_enabled,
            "subagent_max_workers": subagent_max_workers,
            "subagent_max_wall_time_s": subagent_max_wall_time_s,
        },
        "skills": skills,
    }

    # 6. Run the agent with HITL support
    config = {"configurable": {"thread_id": run_id}}
    print(f"\n🚀 Starting agent run (run_id={run_id})\n")

    steps = record_run(
        app=app,
        initial_state=state,
        config=config,
        on_interrupt=on_interrupt,
        auto_resume=True,
    )

    print(f"\n✅ Run complete — {len(steps)} step snapshot(s) recorded.")

    # 7. Print final response
    from src.emergent_planner.utils import get_history_from_state, normalize_content
    final_state = steps[-1].state if steps else {}
    hist = get_history_from_state(final_state)
    for m in reversed(hist):
        if m.__class__.__name__ == "AIMessage":
            print("\n=== Final AI response ===")
            print(normalize_content(m.content))
            break


if __name__ == "__main__":
    main()
