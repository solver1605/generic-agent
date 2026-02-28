"""
LangChain tools for the Emergent Planner agent.

Tools:
  - read_file          — read a file from disk
  - read_file_range    — read a line range from a file
  - write_file         — write/append to a file
  - python_repl        — sandboxed Python interpreter
  - search_web         — web search with multi-provider routing
  - spawn_subagents    — delegate tasks to dynamic worker sub-agents
  - load_skill         — load a skill body from .skills/
  - verify_with_user   — Human-in-the-Loop gate via LangGraph interrupt
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from .models import VerifyRequest
from .search.engine import SearchRequest, run_search
from .skills import discover_skills
from .subagents.orchestrator import run_subagents
from .subagents.types import SubAgentExecutionConfig, SubAgentTask


def _normalize_skill_key(name: str) -> str:
    """
    Normalize skill identifiers so aliases like deep_research, deep-research,
    and "Deep Research" resolve to the same key.
    """
    raw = (name or "").strip().lower()
    if not raw:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


# ---------------------------------------------------------------------------
# File System Tools
# ---------------------------------------------------------------------------

@tool
def read_file(path: str) -> str:
    """
    Read the full contents of a file.

    Use this tool when you need to inspect or understand an entire file.
    If the file does not exist, an error will be raised.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        The full file contents as a string.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text()


@tool
def read_file_range(path: str, start_line: int, end_line: Optional[int] = None) -> str:
    """
    Read a specific range of lines from a file.

    Use this tool when the file is large or when you only need to inspect
    a portion of the file. Line numbers are 1-indexed.

    If end_line is not provided, reads from start_line to the end.

    Args:
        path: Path to the file.
        start_line: Starting line number (1-indexed).
        end_line: Optional ending line number (inclusive).

    Returns:
        The requested file content as a string.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines = p.read_text().splitlines()
    start_idx = max(start_line - 1, 0)
    end_idx = end_line if end_line is not None else len(lines)
    return "\n".join(lines[start_idx:end_idx])


@tool
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """
    Write content to a file.

    Use this tool to create new files or update existing ones.

    Modes:
    - overwrite: Replace the entire file contents.
    - append: Add content to the end of the file.

    Args:
        path: Path to the file.
        content: Content to write.
        mode: Write mode ("overwrite" or "append").

    Returns:
        Confirmation message describing the operation.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if mode == "overwrite":
        p.write_text(content)
        return f"File written (overwrite): {path}"
    elif mode == "append":
        with p.open("a") as f:
            f.write(content)
        return f"Content appended to file: {path}"
    else:
        raise ValueError("Mode must be 'overwrite' or 'append'")


# ---------------------------------------------------------------------------
# Python REPL (sandboxed)
# ---------------------------------------------------------------------------

class _SafeImportError(ImportError):
    pass


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__ implementation. Extend the allowlist as needed."""
    allow = {"math", "random", "re", "time"}
    root = name.split(".", 1)[0]
    if root not in allow:
        raise _SafeImportError(
            f"Import '{name}' is not allowed. Allowed: {sorted(allow)}"
        )
    return __import__(name, globals, locals, fromlist, level)


def _split_last_expr(code: str) -> Tuple[str, Optional[str]]:
    """
    If the code ends with an expression, separate it so we can eval() it
    and return its value (REPL-like behavior).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    if not tree.body:
        return code, None

    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        exec_body = tree.body[:-1]
        exec_mod = ast.Module(body=exec_body, type_ignores=[])
        exec_code = ast.unparse(exec_mod) if exec_body else ""
        last_expr_code = ast.unparse(last.value)
        return exec_code, last_expr_code

    return code, None


def _make_safe_builtins() -> Dict[str, Any]:
    """Provide a minimal, safer builtins set (no file/network/process capabilities)."""
    return {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "len": len,
        "list": list, "max": max, "min": min, "pow": pow, "print": print,
        "range": range, "repr": repr, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "__import__": _safe_import,
    }


@tool
def python_repl(
    code: str,
    state: Optional[Dict[str, Any]] = None,
    timeout_s: float = 2.0,
    max_output_chars: int = 8000,
) -> dict:
    """
    Execute Python code in a restricted REPL-like environment and return output.

    Security model:
    - Runs with a minimal builtins set (no open(), exec(), eval(), os, subprocess, etc.)
    - Imports are restricted to an allowlist: math, random, re, time
    - No filesystem, network, or shell access via Python

    REPL behavior:
    - Captures stdout from print().
    - If the final statement is a Python expression, its value is returned.

    Args:
        code: Python code to execute (may be multi-line).
        state: Optional persistent dict for variables across invocations.
        timeout_s: Soft timeout (best-effort).
        max_output_chars: Truncate stdout to this many characters.

    Returns:
        A dict with keys: stdout, result, error, elapsed_ms
    """
    start = time.time()
    state = state or {}

    safe_globals = {
        "__builtins__": _make_safe_builtins(),
        "math": math, "random": random, "re": re, "time": time,
    }

    if not isinstance(state, dict):
        raise TypeError("Injected `state` must be a dict.")

    stdout_buf = io.StringIO()
    result_repr = ""
    error = ""

    exec_part, last_expr = _split_last_expr(code)

    try:
        with contextlib.redirect_stdout(stdout_buf):
            if exec_part.strip():
                compiled = compile(exec_part, "<python_repl>", "exec")
                exec(compiled, safe_globals, state)

            if last_expr is not None and last_expr.strip():
                compiled_expr = compile(last_expr, "<python_repl>", "eval")
                val = eval(compiled_expr, safe_globals, state)
                result_repr = repr(val)

            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"Execution exceeded {timeout_s}s (soft timeout).")

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    stdout = stdout_buf.getvalue()
    if len(stdout) > max_output_chars:
        stdout = stdout[:max_output_chars] + "\n...[truncated]..."

    elapsed_ms = int(round((time.time() - start) * 1000))

    return {
        "stdout": stdout,
        "result": result_repr,
        "error": error,
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# System Tools
# ---------------------------------------------------------------------------

@tool
def search_web(
    query: str,
    top_k: int = 8,
    recency_days: Optional[int] = None,
    mode: Literal["balanced", "fresh", "deep"] = "balanced",
    enrich: bool = False,
    max_enriched_results: int = 3,
    provider_preference: Optional[List[str]] = None,
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Unified state-of-the-art web search tool.
    """
    req = SearchRequest(
        query=query,
        top_k=top_k,
        recency_days=recency_days,
        mode=mode,
        enrich=enrich,
        max_enriched_results=max_enriched_results,
        provider_preference=provider_preference,
        timeout_s=timeout_s,
    )
    return run_search(req).to_dict()


@tool
def spawn_subagents(
    tasks: List[SubAgentTask],
    execution: Optional[SubAgentExecutionConfig] = None,
    state: Annotated[Optional[Dict[str, Any]], InjectedState] = None,
) -> Dict[str, Any]:
    """
    Spawn dynamic worker sub-agents for parallelizable tasks.
    """
    state = state or {}
    task_models = [SubAgentTask.model_validate(t) for t in (tasks or [])]
    if not task_models:
        return {
            "__tool": "spawn_subagents",
            "request_id": "",
            "status": "failed",
            "summary": "No tasks provided for sub-agent execution.",
            "results": [],
            "errors": [{"task_id": "*", "code": "empty_tasks", "message": "No tasks provided.", "retryable": False}],
            "stats": {"tasks_requested": 0, "tasks_executed": 0},
        }

    exec_cfg = SubAgentExecutionConfig.model_validate(execution or SubAgentExecutionConfig())
    record = run_subagents(
        tasks=task_models,
        execution=exec_cfg,
        parent_state=state,
        all_tools=DEFAULT_TOOLS,
    )
    out = record.to_dict()
    out["__tool"] = "spawn_subagents"
    return out


@tool
def load_skill(skill_name: str) -> str:
    """
    Load a skill by name from discovered SKILL.md files under project skills roots
    (supports recursive discovery).
    and return JSON: {name, description, body, meta}.
    """
    needle = _normalize_skill_key(skill_name)
    if not needle:
        raise ValueError("skill_name cannot be empty")

    skills = discover_skills(Path(".skills"), include_body=True)
    aliases = {needle}
    aliases.add(needle.replace("-", "_"))
    aliases.add(needle.replace("_", "-"))

    for sk in skills:
        candidates = {
            _normalize_skill_key(sk.name),
            _normalize_skill_key(sk.path.parent.name),
        }
        if candidates & aliases:
            return json.dumps({
                "name": sk.name,
                "description": sk.description,
                "body": sk.body,
                "meta": sk.meta,
            })

    available = sorted({sk.name for sk in skills})
    suffix = f" Available: {', '.join(available)}" if available else " No skills discovered."
    raise FileNotFoundError(f"Skill not found: {skill_name}.{suffix}")


@tool
def verify_with_user(
    request: VerifyRequest,
    state: Annotated[Optional[Dict[str, Any]], InjectedState] = None,
) -> Dict[str, Any]:
    """
    Human-in-the-loop gate. When called, execution pauses until the user answers.
    Returns the user's answer to the agent.

    Use only when:
    - You have created a plan and want approval before executing it, OR
    - You changed the plan in a major way and need re-approval, OR
    - You need a key clarification to proceed, OR
    - You are about to perform a risky/irreversible action.

    IMPORTANT:
    - Provide a short context (plan snippet) to help user decide quickly.
    - Ask ONE precise question.
    """
    if bool((state or {}).get("runtime", {}).get("disable_hitl", False)):
        raise ValueError("verify_with_user is disabled in worker sub-agent flows. Escalate to supervisor instead.")

    answer = interrupt({
        "type": request.kind,
        "reason": request.reason,
        "question": request.question,
        "choices": request.choices,
        "context": request.context,
        "default": request.default,
    })

    return {"approved": True, "answer": answer, "reason": request.reason}


# ---------------------------------------------------------------------------
# Default tools list
# ---------------------------------------------------------------------------

DEFAULT_TOOLS = [
    load_skill,
    read_file,
    read_file_range,
    write_file,
    python_repl,
    search_web,
    spawn_subagents,
    verify_with_user,
]
