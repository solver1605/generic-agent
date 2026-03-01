"""
LangChain tools for the Emergent Planner agent.

Tools:
  - read_file          — read a file from disk
  - read_file_range    — read a line range from a file
  - write_file         — write/append to a file
  - write_excel_file   — create/update Excel workbooks
  - create_pptx_deck   — create/update PowerPoint decks
  - python_repl        — sandboxed Python interpreter
  - search_web         — web search with multi-provider routing
  - spawn_subagents    — delegate tasks to dynamic worker sub-agents
  - load_skill         — load a skill body from .skills/
  - verify_with_user   — Human-in-the-Loop gate via LangGraph interrupt
"""
from __future__ import annotations

import ast
import contextlib
import importlib
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

from .config import load_agent_config
from .models import VerifyRequest
from .search.engine import SearchRequest, run_search
from .skills import discover_skills, discover_skills_in_roots
from .subagents.orchestrator import run_subagents
from .subagents.types import SubAgentExecutionConfig, SubAgentTask
from .tool_loader import build_tool_catalog
from .tool_registry import select_tools

_ARTIFACTS_ROOT = Path("artifacts")
_REPORT_FILE_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".html"}
_REPORT_HINTS = {"report", "research", "summary", "findings", "brief", "plan"}


def _normalize_skill_key(name: str) -> str:
    """
    Normalize skill identifiers so aliases like deep_research, deep-research,
    and "Deep Research" resolve to the same key.
    """
    raw = (name or "").strip().lower()
    if not raw:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _require_module(module_name: str, pip_name: str):
    """
    Lazy import helper so optional dependencies are only required when a tool is called.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Missing optional dependency '{module_name}'. Install with: pip install {pip_name}"
        ) from e


def _sanitize_relative_path(p: Path) -> Path:
    parts = [
        part
        for part in p.parts
        if part and part not in {".", "..", "/", "\\"} and part != p.anchor
    ]
    return Path(*parts) if parts else Path("output")


def _coerce_artifact_path(path: str, category: str) -> Path:
    raw = Path(path).expanduser()
    if "artifacts" in raw.parts:
        idx = raw.parts.index("artifacts")
        tail = Path(*raw.parts[idx + 1:]) if (idx + 1) < len(raw.parts) else Path()
        return _ARTIFACTS_ROOT / _sanitize_relative_path(tail)
    rel = Path(raw.name) if raw.is_absolute() else _sanitize_relative_path(raw)
    return _ARTIFACTS_ROOT / category / rel


def _is_report_like_path(path: Path) -> bool:
    if path.suffix.lower() not in _REPORT_FILE_EXTS:
        return False
    lowered = "/".join(path.parts).lower()
    return any(k in lowered for k in _REPORT_HINTS)


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
    p_raw = Path(path)
    if "artifacts" in p_raw.parts or _is_report_like_path(p_raw):
        p = _coerce_artifact_path(path, category="reports")
    else:
        p = p_raw
    p.parent.mkdir(parents=True, exist_ok=True)

    if mode == "overwrite":
        p.write_text(content)
        return f"File written (overwrite): {p.as_posix()}"
    elif mode == "append":
        with p.open("a") as f:
            f.write(content)
        return f"Content appended to file: {p.as_posix()}"
    else:
        raise ValueError("Mode must be 'overwrite' or 'append'")


# ---------------------------------------------------------------------------
# Office Document Tools
# ---------------------------------------------------------------------------

@tool
def write_excel_file(
    path: str,
    sheets: List[Dict[str, Any]],
    mode: Literal["overwrite", "append"] = "overwrite",
) -> Dict[str, Any]:
    """
    Create or update an Excel workbook (.xlsx/.xlsm) with structured sheet data.

    Sheet spec:
      - name: str
      - rows: List[List[Any]]
      - start_row: int (optional, default=1)
      - start_col: int (optional, default=1)
      - clear_sheet: bool (optional, default=False)
    """
    p = _coerce_artifact_path(path, category="excel")
    if p.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("Excel path must end with .xlsx or .xlsm")
    if mode not in {"overwrite", "append"}:
        raise ValueError("Mode must be 'overwrite' or 'append'")

    openpyxl = _require_module("openpyxl", "openpyxl")
    Workbook = getattr(openpyxl, "Workbook")
    load_workbook = getattr(openpyxl, "load_workbook")

    p.parent.mkdir(parents=True, exist_ok=True)

    if mode == "append" and p.exists():
        wb = load_workbook(p.as_posix())
    else:
        wb = Workbook()
        if len(getattr(wb, "worksheets", [])) == 1 and getattr(wb.active, "title", "") == "Sheet":
            wb.remove(wb.active)

    written_sheets: List[str] = []
    total_rows = 0

    for idx, spec in enumerate(sheets or []):
        if not isinstance(spec, dict):
            raise ValueError(f"sheets[{idx}] must be an object")
        name = str(spec.get("name", "")).strip()
        rows = spec.get("rows", [])
        if not name:
            raise ValueError(f"sheets[{idx}].name is required")
        if not isinstance(rows, list):
            raise ValueError(f"sheets[{idx}].rows must be a list")

        start_row = int(spec.get("start_row", 1))
        start_col = int(spec.get("start_col", 1))
        clear_sheet = bool(spec.get("clear_sheet", False))
        if start_row < 1 or start_col < 1:
            raise ValueError(f"sheets[{idx}] start_row/start_col must be >= 1")

        if name in set(wb.sheetnames):
            ws = wb[name]
        else:
            ws = wb.create_sheet(title=name)

        if clear_sheet and getattr(ws, "max_row", 0) > 0:
            ws.delete_rows(1, ws.max_row)

        for r, row_vals in enumerate(rows):
            if not isinstance(row_vals, list):
                raise ValueError(f"sheets[{idx}].rows[{r}] must be a list")
            for c, val in enumerate(row_vals):
                ws.cell(row=start_row + r, column=start_col + c, value=val)

        written_sheets.append(name)
        total_rows += len(rows)

    if not written_sheets and "Sheet" not in set(wb.sheetnames):
        wb.create_sheet(title="Sheet")

    wb.save(p.as_posix())
    return {
        "path": p.as_posix(),
        "mode": mode,
        "sheets_written": written_sheets,
        "rows_written": total_rows,
        "status": "ok",
    }


@tool
def create_pptx_deck(
    path: str,
    slides: List[Dict[str, Any]],
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    mode: Literal["overwrite", "append"] = "overwrite",
) -> Dict[str, Any]:
    """
    Create or append to a PowerPoint deck (.pptx) using structured slide data.

    Slide spec:
      - title: str (optional)
      - bullets: List[str] (optional)
      - body: str (optional)
      - notes: str (optional)
      - layout: int (optional, default=1)
    """
    p = _coerce_artifact_path(path, category="ppt")
    if p.suffix.lower() != ".pptx":
        raise ValueError("PowerPoint path must end with .pptx")
    if mode not in {"overwrite", "append"}:
        raise ValueError("Mode must be 'overwrite' or 'append'")

    pptx = _require_module("pptx", "python-pptx")
    Presentation = getattr(pptx, "Presentation")

    p.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append" and p.exists():
        pres = Presentation(p.as_posix())
    else:
        pres = Presentation()

    slides_added = 0
    starting_slides = len(list(pres.slides))

    if title and mode == "overwrite":
        title_slide_layout = pres.slide_layouts[0]
        s = pres.slides.add_slide(title_slide_layout)
        if getattr(getattr(s, "shapes", None), "title", None) is not None:
            s.shapes.title.text = str(title)
        if subtitle and len(getattr(s, "placeholders", [])) > 1:
            s.placeholders[1].text = str(subtitle)
        slides_added += 1

    for idx, spec in enumerate(slides or []):
        if not isinstance(spec, dict):
            raise ValueError(f"slides[{idx}] must be an object")
        layout = int(spec.get("layout", 1))
        if layout < 0 or layout >= len(pres.slide_layouts):
            raise ValueError(f"slides[{idx}].layout is out of range (0..{len(pres.slide_layouts)-1})")

        s = pres.slides.add_slide(pres.slide_layouts[layout])
        slide_title = str(spec.get("title", "") or "")
        body = str(spec.get("body", "") or "")
        bullets = spec.get("bullets", []) or []
        notes = str(spec.get("notes", "") or "")

        if slide_title and getattr(getattr(s, "shapes", None), "title", None) is not None:
            s.shapes.title.text = slide_title

        body_placeholder = None
        if len(getattr(s, "placeholders", [])) > 1:
            body_placeholder = s.placeholders[1]

        if body_placeholder is not None:
            tf = body_placeholder.text_frame
            if body:
                tf.text = body
            elif isinstance(bullets, list) and bullets:
                tf.text = str(bullets[0])
                for b in bullets[1:]:
                    tf.add_paragraph().text = str(b)

        if notes:
            s.notes_slide.notes_text_frame.text = notes

        slides_added += 1

    pres.save(p.as_posix())
    return {
        "path": p.as_posix(),
        "mode": mode,
        "slides_added": slides_added,
        "total_slides": len(list(pres.slides)),
        "starting_slides": starting_slides,
        "status": "ok",
    }


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
    cfg = load_agent_config(Path("agent_config.yaml"))
    all_tools = build_tool_catalog(cfg, DEFAULT_TOOLS)
    enabled_names = list((state or {}).get("runtime", {}).get("enabled_tool_names", []) or [])
    if enabled_names:
        all_tools = select_tools(all_tools, enabled_names)
    record = run_subagents(
        tasks=task_models,
        execution=exec_cfg,
        parent_state=state,
        all_tools=all_tools or DEFAULT_TOOLS,
    )
    out = record.to_dict()
    out["__tool"] = "spawn_subagents"
    return out


@tool
def load_skill(
    skill_name: str,
    state: Annotated[Optional[Dict[str, Any]], InjectedState] = None,
) -> str:
    """
    Load a skill by name from discovered SKILL.md files under project skills roots
    (supports recursive discovery).
    and return JSON: {name, description, body, meta}.
    """
    needle = _normalize_skill_key(skill_name)
    if not needle:
        raise ValueError("skill_name cannot be empty")

    runtime = dict((state or {}).get("runtime", {}) or {})
    scoped_roots = list(runtime.get("skills_roots_resolved", []) or [])
    if scoped_roots:
        skills = discover_skills_in_roots(
            [Path(str(p)) for p in scoped_roots],
            include_body=True,
            strict_scope=True,
        )
    else:
        skills = discover_skills(Path(".skills"), include_body=True)

    allowlist_norm = {
        _normalize_skill_key(x)
        for x in (runtime.get("skills_allowlist_norm", []) or [])
        if _normalize_skill_key(x)
    }
    denylist_norm = {
        _normalize_skill_key(x)
        for x in (runtime.get("skills_denylist_norm", []) or [])
        if _normalize_skill_key(x)
    }
    aliases = {needle}
    aliases.add(needle.replace("-", "_"))
    aliases.add(needle.replace("_", "-"))

    for sk in skills:
        sk_norm = _normalize_skill_key(sk.name)
        if allowlist_norm and sk_norm not in allowlist_norm:
            continue
        if sk_norm in denylist_norm:
            continue
        candidates = {
            sk_norm,
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
    write_excel_file,
    create_pptx_deck,
    python_repl,
    search_web,
    spawn_subagents,
    verify_with_user,
]
