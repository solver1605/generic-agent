"""
debug_ui.py — Jupyter-widget-based debug UI for stepping through graph runs.

This module requires ipywidgets and is intended for notebook environments.
For CLI/server use, use record_run() directly without SotaGraphUI.
"""
from __future__ import annotations

import difflib
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langgraph.types import Command
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .models import Step
from .utils import (
    _diff_states,
    _pretty_json,
    _shallow_snapshot,
    extract_tool_calls,
    get_history_from_state,
    get_prompt_messages_from_state,
    get_prompt_text_fallback,
    msg_tokens,
    normalize_content,
    safe_get,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ResumeAnswer = Union[str, int, float, bool, Dict[str, Any], List[Any]]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _role(m) -> str:
    return m.__class__.__name__.replace("Message", "").lower()


def _content(m) -> str:
    return getattr(m, "content", "") or ""


def _truncate(s: str, n: int = 4000) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]..."


def _pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def render_messages(
    msgs: List[Any], title: str, search: str = "", only_matches: bool = False
) -> str:
    if not msgs:
        return f"_(empty: {title})_"

    q = (search or "").strip().lower()
    out = [f"## {title} ({len(msgs)} msgs)"]
    for i, m in enumerate(msgs):
        role = m.__class__.__name__.replace("Message", "").lower()
        raw = getattr(m, "content", None)
        text = normalize_content(raw)

        if q and only_matches and (q not in text.lower()):
            continue

        out.append(f"### {i}. `{role}`")
        calls = extract_tool_calls(m)
        if calls:
            out.append("**tool_calls:**")
            out.append("```json")
            out.append(json.dumps(calls, indent=2, ensure_ascii=False, default=str))
            out.append("```")

        if text.strip():
            out.append("```")
            out.append(text[:8000] + ("\n...[truncated]..." if len(text) > 8000 else ""))
            out.append("```")
        else:
            out.append("_(empty content — tool-call-only or structured output)_")

        if role == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                out.append(f"- tool_call_id: `{tcid}`")

    return "\n".join(out)


def render_prompt_tab(
    state: Dict[str, Any], search: str = "", only_matches: bool = False
) -> str:
    pm = get_prompt_messages_from_state(state)
    if pm:
        return render_messages(pm, "Prompt Messages", search, only_matches)

    fallback = get_prompt_text_fallback(state)
    if fallback.strip():
        return (
            "## Prompt (artifact fallback)\n```text\n"
            + fallback[:12000]
            + ("\n...[truncated]..." if len(fallback) > 12000 else "")
            + "\n```"
        )

    return "_(Prompt not available: ensure context node sets prompt_messages OR add persist_prompt_artifact_node)_"


def render_history(hist) -> str:
    if not hist:
        return "_(history empty)_"

    out = []
    for i, m in enumerate(hist):
        role = _role(m)
        content = normalize_content(_content(m))

        out.append(f"### {i}. `{role}`  (≈{msg_tokens(m)} toks)")

        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            out.append("**tool_calls:**")
            out.append("```json")
            out.append(_pretty(tool_calls))
            out.append("```")

        if content.strip():
            out.append("```")
            out.append(_truncate(content, 6000))
            out.append("```")
        else:
            out.append("_(empty content — likely a tool-call-only AIMessage)_")

        if m.__class__.__name__ == "ToolMessage":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                out.append(f"- tool_call_id: `{tcid}`")

    return "\n".join(out)


def render_prompt(pm) -> str:
    if not pm:
        return "_(messages empty)_"

    out = []
    total = 0
    for i, m in enumerate(pm):
        t = msg_tokens(m)
        total += t
        role = _role(m)
        content = _content(m)
        out.append(f"### {i}. `{role}`  (≈{t} toks)")

        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            out.append("**tool_calls:**")
            out.append("```json")
            out.append(_pretty(tool_calls))
            out.append("```")

        if content.strip():
            out.append("```")
            out.append(_truncate(content, 6000))
            out.append("```")
        else:
            out.append("_(empty content)_")

    out.insert(0, f"**Estimated prompt tokens:** ≈{total}")
    return "\n".join(out)


def render_telemetry(tel) -> str:
    if not tel:
        return "_(telemetry empty — wrap nodes with instrument_node to populate)_"
    return "```json\n" + _pretty_json(tel[-50:]) + "\n```"


def render_runtime(rt) -> str:
    if not rt:
        return "_(runtime empty)_"
    return "```json\n" + _pretty_json(rt) + "\n```"


def render_memory(mem) -> str:
    if not mem:
        return "_(memory empty)_"
    return "```json\n" + _pretty_json(mem) + "\n```"


def render_diff(d) -> str:
    return "```json\n" + _pretty_json(d) + "\n```"


def render_tool_inspector(hist) -> str:
    if not hist:
        return "_(no history)_"
    for m in reversed(hist):
        if m.__class__.__name__ == "AIMessage":
            calls = extract_tool_calls(m)
            if calls:
                return "### Last tool calls\n```json\n" + _pretty_json(calls) + "\n```"
    return "_(no tool calls found in AI messages)_"


def prompt_text(pm) -> str:
    if not pm:
        return ""
    lines = []
    for m in pm:
        lines.append(f"{_role(m)}:\n{_content(m)}\n")
    return "\n".join(lines)


def render_prompt_diff(prev_pm, cur_pm) -> str:
    a = prompt_text(prev_pm).splitlines()
    b = prompt_text(cur_pm).splitlines()
    diff = difflib.unified_diff(a, b, fromfile="prev_prompt", tofile="cur_prompt", lineterm="")
    txt = "\n".join(diff)
    if not txt.strip():
        return "_(no prompt diff)_"
    return "```diff\n" + _truncate(txt, 12000) + "\n```"


def render_rich_step(cur: dict, prev: dict) -> str:
    """Returns a plain-text rich-rendered view of the current step."""
    console = Console(record=True, width=110)
    rt = safe_get(cur, "runtime", {}) or {}
    tel = safe_get(cur, "telemetry", []) or []
    hist = get_history_from_state(cur)

    title = f"Step {safe_get(cur, '_step', None) or ''}".strip()
    hdr = Table.grid(expand=True)
    hdr.add_column(justify="left")
    hdr.add_column(justify="right")
    hdr.add_row(
        f"🧠 run_id={rt.get('run_id','?')}  turn={rt.get('turn_index',0)}  after_tool={rt.get('after_tool',False)}",
        f"history={len(hist)}  telemetry={len(tel)}"
    )
    console.print(Panel(hdr, title="📍 Snapshot", border_style="cyan"))

    intr = cur.get("__interrupt__")
    if intr:
        console.print(Panel(str(intr)[:2000], title="⏸️ Interrupt", border_style="yellow"))

    tree = Tree("🧾 Recent messages")
    for m in hist[-8:]:
        role = m.__class__.__name__.replace("Message", "").lower()
        icon = {"human": "👤", "ai": "🤖", "tool": "🧰", "system": "⚙️"}.get(role, "💬")
        content = normalize_content(getattr(m, "content", None))
        content = content.strip() if isinstance(content, str) else str(content)
        if len(content) > 240:
            content = content[:240] + "…"
        extra = ""
        if m.__class__.__name__ == "AIMessage":
            tc = extract_tool_calls(m)
            if tc:
                extra = f"  🔧 tool_calls={len(tc)}"
        tree.add(f"{icon} {role}{extra} :: {content}")
    console.print(tree)

    if tel:
        last = tel[-1]
        t = Table(title="📊 Last telemetry entry", box=box.SIMPLE, show_lines=False)
        t.add_column("key", style="bold")
        t.add_column("value")
        for k in ["node", "status", "elapsed_ms", "turn_index", "run_id"]:
            if k in last:
                t.add_row(k, str(last.get(k)))
        err = last.get("error") or ""
        if isinstance(err, dict):
            t.add_row("error.type", str(err.get("type", "")))
            t.add_row("error.class", str(err.get("class", "")))
            t.add_row("error.msg", str(err.get("message", ""))[:300])
        elif err:
            t.add_row("error", str(err)[:300])
        console.print(t)

    return console.export_text(clear=False)


# ---------------------------------------------------------------------------
# Interrupt helpers
# ---------------------------------------------------------------------------

def _extract_interrupt_payload(state: Dict[str, Any]) -> Optional[Any]:
    """Normalise LangGraph's __interrupt__ field into a payload."""
    intr = state.get("__interrupt__")
    if not intr:
        return None
    if isinstance(intr, list) and len(intr) > 0:
        first = intr[0]
        if isinstance(first, dict) and "value" in first:
            return first["value"]
        return first
    return intr


# ---------------------------------------------------------------------------
# record_run — stream + HITL handler
# ---------------------------------------------------------------------------

def record_run(
    app,
    initial_state: Dict[str, Any],
    keys: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    on_interrupt: Optional[Callable[[Any, Dict[str, Any]], ResumeAnswer]] = None,
    auto_resume: bool = True,
    max_interrupts: int = 10,
) -> List[Step]:
    """
    Reliable recorder for LangGraph with HITL interrupts.

    - Uses stream_mode='values' to capture full state snapshots.
    - Detects interrupts via state['__interrupt__'].
    - If an interrupt occurs:
        - if auto_resume and on_interrupt provided: resumes via Command(resume=answer)
        - else: records the interrupt and returns (caller/UI resumes later)

    IMPORTANT:
    - For resume to work, app must be compiled with a checkpointer.
    - config must include a stable thread_id:
        config={"configurable":{"thread_id":"..."}}
    """
    if keys is None:
        keys = [
            "history", "messages", "input_messages", "llm_input",
            "telemetry", "runtime", "memory", "__interrupt__",
        ]

    if config is None:
        config = {"configurable": {"thread_id": f"thread-{uuid.uuid4()}"}}
    else:
        config = dict(config)
        cfg = dict(config.get("configurable", {}) or {})
        cfg.setdefault("thread_id", f"thread-{uuid.uuid4()}")
        config["configurable"] = cfg

    steps: List[Step] = []
    prev_snap: Dict[str, Any] = {}
    idx = 0
    interrupts_seen = 0
    has_runtime_resume = bool(getattr(app, "resume", None))

    def _consume_stream(input_obj, *, resume_answer: Optional[ResumeAnswer] = None):
        nonlocal idx, prev_snap, interrupts_seen
        if resume_answer is None:
            if has_runtime_resume:
                iterator = app.stream(input_obj, config=config)
            else:
                iterator = app.stream(input_obj, config=config, stream_mode="values")
        else:
            if has_runtime_resume:
                iterator = app.resume(resume_answer, config=config)
            else:
                iterator = app.stream(Command(resume=resume_answer), config=config, stream_mode="values")

        for full_state in iterator:
            snap = _shallow_snapshot(full_state, keys)
            d = _diff_states(prev_snap, snap) if idx > 0 else {"note": "initial snapshot"}
            steps.append(Step(idx=idx, state=snap, diff=d))
            prev_snap = snap
            idx += 1

            payload = _extract_interrupt_payload(full_state)
            if payload is not None:
                interrupts_seen += 1
                if interrupts_seen > max_interrupts:
                    steps.append(Step(
                        idx=idx,
                        state={"__interrupt__": payload, "note": "max_interrupts exceeded"},
                        diff={"note": "stopped"},
                    ))
                    return payload, False

                if not (auto_resume and on_interrupt):
                    return payload, False

                answer = on_interrupt(payload, full_state)
                return payload, answer

        return None, None

    # 1) Run from initial_state
    payload, resume = _consume_stream(initial_state)

    # 2) If interrupted and we have an answer, resume loop
    while payload is not None and resume is not False:
        answer = resume
        payload, resume = _consume_stream(None, resume_answer=answer)

    return steps


# ---------------------------------------------------------------------------
# SotaGraphUI (Jupyter widgets — optional import guard)
# ---------------------------------------------------------------------------

try:
    import ipywidgets as widgets
    from IPython.display import Markdown, clear_output, display

    class SotaGraphUI:
        """Interactive Jupyter widget for step-through debugging of a graph run."""

        def __init__(self, steps: List[Step]):
            self.steps = steps
            self.ptr = 0

            # Outputs
            self.out_history   = widgets.Output()
            self.out_prompt    = widgets.Output()
            self.out_diff      = widgets.Output()
            self.out_updates   = widgets.Output()
            self.out_tool      = widgets.Output()
            self.out_runtime   = widgets.Output()
            self.out_memory    = widgets.Output()
            self.out_telemetry = widgets.Output()
            self.out_rich      = widgets.Output()

            # Controls
            self.btn_next  = widgets.Button(description="Next", button_style="primary")
            self.btn_prev  = widgets.Button(description="Prev")
            self.btn_play  = widgets.ToggleButton(description="Play ▶", value=False)
            self.btn_clear = widgets.Button(description="Clear Tab")
            self.btn_reset = widgets.Button(description="Reset ⟲", button_style="warning")

            self.speed  = widgets.IntSlider(value=300, min=50, max=2000, step=50,
                                            description="ms/step", continuous_update=False)
            self.slider = widgets.IntSlider(value=0, min=0, max=max(0, len(steps)-1),
                                            step=1, description="step", continuous_update=False)

            self.search          = widgets.Text(value="", description="search",
                                                placeholder="find in prompt/history")
            self.chk_only_matches = widgets.Checkbox(value=False, description="only matching msgs")

            self.tabs = widgets.Tab(children=[
                widgets.VBox([self.out_history]),
                widgets.VBox([self.out_prompt]),
                widgets.VBox([self.out_diff]),
                widgets.VBox([self.out_tool]),
                widgets.VBox([self.out_updates]),
                widgets.VBox([self.out_runtime]),
                widgets.VBox([self.out_memory]),
                widgets.VBox([self.out_telemetry]),
                widgets.VBox([self.out_rich]),
            ])
            for i, t in enumerate(
                ["History", "Prompt", "Prompt Diff", "Tools",
                 "Diff/Updates", "Runtime", "Memory", "Telemetry", "Rich"]
            ):
                self.tabs.set_title(i, t)

            self.hdr = widgets.HTML()

            # Wire events
            self.btn_next.on_click(lambda _: self.step_to(self.ptr + 1))
            self.btn_prev.on_click(lambda _: self.step_to(self.ptr - 1))
            self.btn_reset.on_click(lambda _: self.step_to(0))
            self.btn_clear.on_click(lambda _: self._clear_active_tab())
            self.slider.observe(lambda ch: self.step_to(ch["new"]), names="value")
            self.search.observe(lambda ch: self.render(), names="value")
            self.chk_only_matches.observe(lambda ch: self.render(), names="value")
            self.btn_play.observe(lambda ch: self._on_play(ch["new"]), names="value")

            self.ui = widgets.VBox([
                widgets.HBox([self.btn_prev, self.btn_next, self.btn_play, self.btn_clear, self.btn_reset]),
                widgets.HBox([self.slider, self.speed]),
                widgets.HBox([self.search, self.chk_only_matches, self.hdr]),
                self.tabs,
            ])

            self.render()

        def show(self):
            display(self.ui)

        def _clear_active_tab(self):
            self.btn_play.value = False
            outs = [
                self.out_history, self.out_prompt, self.out_diff, self.out_tool,
                self.out_updates, self.out_runtime, self.out_memory, self.out_telemetry,
                self.out_rich,
            ]
            outs[self.tabs.selected_index].clear_output(wait=True)

        def step_to(self, idx: int):
            idx = max(0, min(idx, len(self.steps)-1))
            self.ptr = idx
            self.slider.value = idx
            self.render()

        def _on_play(self, on: bool):
            if not on:
                return
            while self.btn_play.value and self.ptr < len(self.steps)-1:
                self.step_to(self.ptr + 1)
                time.sleep(self.speed.value / 1000.0)
            self.btn_play.value = False

        def _filter_msgs(self, msgs):
            q = (self.search.value or "").strip().lower()
            if not q or not self.chk_only_matches.value:
                return msgs
            return [m for m in msgs if q in (_content(m).lower())]

        def render(self):
            if not self.steps:
                self.hdr.value = "<b>No steps recorded</b>"
                return

            cur  = self.steps[self.ptr].state
            prev = self.steps[self.ptr-1].state if self.ptr > 0 else {}
            hist = get_history_from_state(cur)
            prompt_md = render_prompt_tab(cur, self.search.value, self.chk_only_matches.value)
            pm      = safe_get(cur, "messages", [])
            pm_prev = safe_get(prev, "messages", [])
            tel     = safe_get(cur, "telemetry", [])
            rt      = safe_get(cur, "runtime", {})
            mem     = safe_get(cur, "memory", {})
            diff    = self.steps[self.ptr].diff

            prompt_tok = sum(msg_tokens(m) for m in (pm or []))
            hist_len   = len(safe_get(cur, "history", []))
            self.hdr.value = (
                f"<b>step</b> {self.ptr}/{len(self.steps)-1} | "
                f"<b>history</b> {hist_len} msgs | "
                f"<b>prompt</b> ≈{prompt_tok} toks"
            )

            with self.out_history:
                clear_output(wait=True)
                display(Markdown(render_messages(hist, "History", self.search.value, self.chk_only_matches.value)))
            with self.out_prompt:
                clear_output(wait=True)
                display(Markdown(prompt_md))
            with self.out_diff:
                clear_output(wait=True)
                display(Markdown(render_prompt_diff(pm_prev, pm)))
            with self.out_tool:
                clear_output(wait=True)
                display(Markdown(render_tool_inspector(safe_get(cur, "history", []))))
            with self.out_updates:
                clear_output(wait=True)
                display(Markdown(render_diff(diff)))
            with self.out_runtime:
                clear_output(wait=True)
                display(Markdown(render_runtime(rt)))
            with self.out_memory:
                clear_output(wait=True)
                display(Markdown(render_memory(mem)))
            with self.out_telemetry:
                clear_output(wait=True)
                display(Markdown(render_telemetry(tel)))
            with self.out_rich:
                clear_output(wait=True)
                pretty = render_rich_step(cur, prev)
                display(widgets.HTML(f"<pre style='font-size:12px; line-height:1.25'>{pretty}</pre>"))

except ImportError:
    # ipywidgets not installed — SotaGraphUI unavailable outside notebooks
    class SotaGraphUI:  # type: ignore
        def __init__(self, steps):
            raise ImportError(
                "SotaGraphUI requires ipywidgets. "
                "Install it with: pip install ipywidgets"
            )
