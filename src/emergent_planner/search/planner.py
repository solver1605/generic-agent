"""
Query decomposition and expansion helpers.
"""
from __future__ import annotations

import re
from typing import List

from .types import SearchMode


_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "is", "are",
}


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q[:500]


def _keywords(query: str, limit: int = 6) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9_\-\.]+", query.lower())
    out: List[str] = []
    for t in toks:
        if len(t) < 3 or t in _STOPWORDS:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def expand_subqueries(query: str, mode: SearchMode = "balanced", max_subqueries: int = 4) -> List[str]:
    base = normalize_query(query)
    if not base:
        return []

    kws = _keywords(base, limit=8)
    out = [base]

    if kws:
        out.append(" ".join(kws[: min(4, len(kws))]))

    if mode in {"fresh", "deep"}:
        out.append(f"{base} latest updates")
    if mode == "deep":
        out.append(f"{base} official documentation")
        out.append(f"{base} comparison analysis")

    deduped: List[str] = []
    for q in out:
        q = normalize_query(q)
        if q and q not in deduped:
            deduped.append(q)
        if len(deduped) >= max(1, max_subqueries):
            break
    return deduped
