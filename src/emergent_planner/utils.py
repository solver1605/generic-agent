"""
Shared utility helpers used across the Emergent Planner modules.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, ToolMessage


# ---------------------------------------------------------------------------
# Token / content helpers
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Rough heuristic: ~1 token per 4 chars."""
    return max(1, len(text) // 4)


def msg_tokens(m: BaseMessage) -> int:
    return approx_tokens(getattr(m, "content", "") or "")


def normalize_content(content: Any) -> str:
    """
    Provider-agnostic normalization of message.content into displayable text.
    Handles: str, None, dict, list[blocks] (Gemini/OpenAI multimodal/tool blocks).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, indent=2, ensure_ascii=False, default=str)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                elif item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p is not None])
    return str(content)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def compact_tool_message(m: ToolMessage, max_chars: int) -> ToolMessage:
    txt = m.content or ""
    if len(txt) <= max_chars:
        return m
    snippet = txt[:max_chars] + "\n...[truncated]..."
    return ToolMessage(content=snippet, tool_call_id=m.tool_call_id)


def _messages_to_compact_text(msgs: List[BaseMessage], max_chars: int = 12000) -> str:
    parts = []
    for m in msgs:
        role = m.__class__.__name__.replace("Message", "").lower()
        content = getattr(m, "content", "") or ""
        parts.append(f"{role}: {content}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]..."
    return text


def extract_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls from AI messages across providers.
    Checks:
      - msg.tool_calls (LC standard)
      - msg.additional_kwargs["tool_calls"]
      - msg.additional_kwargs["function_call"] (older OpenAI style)
    """
    calls = []

    tc = getattr(msg, "tool_calls", None)
    if tc:
        calls.extend(tc)

    ak = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        if ak.get("tool_calls"):
            calls.extend(ak["tool_calls"])
        if ak.get("function_call"):
            calls.append({
                "name": ak["function_call"].get("name"),
                "args": ak["function_call"].get("arguments"),
            })

    # Normalize to {name, args, id?}
    norm = []
    for c in calls:
        if isinstance(c, dict):
            norm.append({
                "id": c.get("id") or c.get("tool_call_id"),
                "name": c.get("name") or (c.get("function") or {}).get("name"),
                "args": c.get("args") or (c.get("function") or {}).get("arguments"),
            })
        else:
            norm.append({"name": str(c), "args": None})
    return norm


def get_history_from_state(state: Dict[str, Any]) -> List[BaseMessage]:
    """Supports graphs that store messages under different keys."""
    return state.get("history") or state.get("messages") or []


def get_prompt_messages_from_state(state: Dict[str, Any]) -> List[BaseMessage]:
    return state.get("messages") or state.get("input_messages") or state.get("llm_input") or []


def get_prompt_text_fallback(state: Dict[str, Any]) -> str:
    runtime = state.get("runtime", {}) or {}
    arts = runtime.get("prompt_artifacts") or []
    if arts:
        return arts[-1].get("prompt_text", "") or ""
    return ""


def safe_get(d: Dict[str, Any], k: str, default):
    v = d.get(k, default)
    return default if v is None else v


# ---------------------------------------------------------------------------
# JSON / formatting helpers
# ---------------------------------------------------------------------------

def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _msg_role(m) -> str:
    return m.__class__.__name__.replace("Message", "").lower()


def _msg_preview(m, max_chars=600) -> str:
    c = getattr(m, "content", "") or ""
    if len(c) > max_chars:
        c = c[:max_chars] + "\n...[truncated]..."
    return c


# ---------------------------------------------------------------------------
# Metrics / observability helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_len(x: Any) -> int:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return 0


def _coarse_size(obj: Any, max_items: int = 64) -> Dict[str, Any]:
    """Safe, cheap size estimator (no deep recursion)."""
    try:
        if isinstance(obj, str):
            return {"type": "str", "chars": len(obj)}
        if isinstance(obj, list):
            return {"type": "list", "len": len(obj)}
        if isinstance(obj, dict):
            keys = list(obj.keys())[:max_items]
            return {"type": "dict", "len": len(obj), "keys_head": keys}
        return {"type": type(obj).__name__}
    except Exception:
        return {"type": "unknown"}


def _fingerprint_prompt(messages: List[Any], max_msgs: int = 12, max_chars: int = 8000) -> str:
    """Short fingerprint to detect prompt drift without storing whole prompt."""
    parts: List[str] = []
    for m in messages[-max_msgs:]:
        role = m.__class__.__name__
        txt = normalize_content(getattr(m, "content", None))
        parts.append(f"{role}:{txt[:800]}")
    blob = "\n".join(parts)[:max_chars].encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _classify_error(exc: BaseException) -> str:
    """Coarse failure taxonomy useful for dashboards + retry policies."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "validationerror" in name or "pydantic" in msg:
        return "schema_validation"
    if "timeout" in name or "timed out" in msg:
        return "timeout"
    if "ratelimit" in msg or "429" in msg:
        return "rate_limit"
    if "permission" in msg or "denied" in msg:
        return "permission"
    if "filenotfound" in name or "no such file" in msg:
        return "file_not_found"
    if "connection" in name or "dns" in msg or "ssl" in msg:
        return "network"
    return "unknown"


# ---------------------------------------------------------------------------
# State diffing (for debug / telemetry)
# ---------------------------------------------------------------------------

def _shallow_snapshot(state: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    snap = {}
    for k in keys:
        if k in state:
            snap[k] = state[k]
    return snap


def _diff_states(prev: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight diff for a few keys.
    - For lists: show length change
    - For dicts: show changed keys
    - For scalars: show old/new
    """
    diff = {}
    all_keys = set(prev.keys()) | set(cur.keys())
    for k in sorted(all_keys):
        a = prev.get(k, None)
        b = cur.get(k, None)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            diff[k] = {"type": "list", "prev_len": len(a), "cur_len": len(b)}
        elif isinstance(a, dict) and isinstance(b, dict):
            changed = sorted(
                set(a.keys()) ^ set(b.keys())
                | {kk for kk in a.keys() & b.keys() if a[kk] != b[kk]}
            )
            diff[k] = {"type": "dict", "changed_keys": changed[:100]}
        else:
            diff[k] = {"type": "value", "prev": str(a)[:200], "cur": str(b)[:200]}
    return diff
