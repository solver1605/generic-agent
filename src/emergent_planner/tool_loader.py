"""Tool catalog loader with profile-aware allow/deny resolution."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List

from .config import AgentConfig, AgentProfileConfig
from .tool_registry import tool_name


def _is_allowed_module(module_name: str, prefixes: List[str]) -> bool:
    for p in prefixes:
        pref = str(p).strip()
        if not pref:
            continue
        if module_name == pref or module_name.startswith(pref + "."):
            return True
    return False


def _ensure_tool_list(obj: Any, *, spec: str) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out = list(obj)
    else:
        out = [obj]

    bad = [x for x in out if not (hasattr(x, "name") or hasattr(x, "__name__"))]
    if bad:
        raise TypeError(f"Imported symbol '{spec}' returned non-tool object(s): {bad!r}")
    return out


def _load_tools_from_spec(spec: str, *, allow_module_prefixes: List[str]) -> List[Any]:
    if ":" not in spec:
        raise ValueError(f"Invalid tool import spec '{spec}'. Expected format: module.path:symbol")
    module_name, symbol = spec.split(":", 1)
    module_name = module_name.strip()
    symbol = symbol.strip()
    if not module_name or not symbol:
        raise ValueError(f"Invalid tool import spec '{spec}'. Expected format: module.path:symbol")

    if not _is_allowed_module(module_name, allow_module_prefixes):
        allowed = ", ".join(allow_module_prefixes)
        raise ValueError(
            f"Rejected custom tool import '{spec}'. Module '{module_name}' is outside allowlist: {allowed}"
        )

    mod = importlib.import_module(module_name)
    if not hasattr(mod, symbol):
        raise AttributeError(f"Custom tool symbol '{symbol}' not found in module '{module_name}'")

    obj = getattr(mod, symbol)
    # LangChain tool objects expose both `name` and `invoke`.
    if hasattr(obj, "name") and hasattr(obj, "invoke"):
        return _ensure_tool_list(obj, spec=spec)

    if callable(obj):
        produced = obj()
        return _ensure_tool_list(produced, spec=spec)

    return _ensure_tool_list(obj, spec=spec)


def _ordered_unique_tools(tools: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for t in tools:
        n = tool_name(t)
        if n in seen:
            continue
        seen.add(n)
        out.append(t)
    return out


def build_tool_catalog(cfg: AgentConfig, default_tools: Iterable[Any]) -> List[Any]:
    """
    Build full catalog = built-in tools + custom imported tools.
    """
    tools = list(default_tools)
    for spec in list(cfg.tool_catalog.custom_imports or []):
        tools.extend(_load_tools_from_spec(spec, allow_module_prefixes=list(cfg.tool_catalog.allow_module_prefixes or [])))
    return _ordered_unique_tools(tools)


def resolve_tools_for_profile(
    catalog_tools: Iterable[Any],
    profile: AgentProfileConfig,
    *,
    extra_allow: Iterable[str] | None = None,
    extra_deny: Iterable[str] | None = None,
) -> List[Any]:
    """
    Resolve final tool binding by profile allow/deny and optional runtime overrides.
    """
    catalog = _ordered_unique_tools(catalog_tools)
    by_name: Dict[str, Any] = {tool_name(t): t for t in catalog}
    all_names = [tool_name(t) for t in catalog]

    profile_allow = [x for x in list(profile.tools.allow or []) if x]
    profile_deny = set(x for x in list(profile.tools.deny or []) if x)

    unknown_profile = sorted((set(profile_allow) | profile_deny) - set(all_names))
    if unknown_profile:
        raise ValueError(
            "Profile references unknown tools: " + ", ".join(unknown_profile)
        )

    if profile_allow:
        selected_names = [n for n in all_names if n in set(profile_allow)]
    else:
        selected_names = list(all_names)

    selected_names = [n for n in selected_names if n not in profile_deny]

    if extra_allow is not None:
        allow_set = {x for x in extra_allow if x}
        unknown_allow = sorted(allow_set - set(selected_names))
        if unknown_allow:
            raise ValueError(
                "Requested tool allowlist contains tools not available for this profile: " + ", ".join(unknown_allow)
            )
        selected_names = [n for n in selected_names if n in allow_set]

    if extra_deny is not None:
        deny_set = {x for x in extra_deny if x}
        selected_names = [n for n in selected_names if n not in deny_set]

    if not selected_names:
        raise ValueError(f"No tools enabled after profile policy for '{profile.id}'.")

    return [by_name[n] for n in selected_names]
