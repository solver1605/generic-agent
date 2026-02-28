"""
Ranking and deduplication for search results.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .types import SearchResultItem


_TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid",
}


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url.strip())
    scheme = p.scheme.lower() or "https"
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    query_pairs = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in _TRACKING_QUERY_KEYS]
    query = urlencode(sorted(query_pairs))
    path = re.sub(r"/+", "/", p.path or "/")
    return urlunparse((scheme, netloc, path.rstrip("/") or "/", "", query, ""))


def _extract_domain(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def dedupe_results(results: List[SearchResultItem]) -> List[SearchResultItem]:
    merged: Dict[str, SearchResultItem] = {}
    for r in results:
        key = canonicalize_url(r.url)
        if not key:
            continue
        if key not in merged:
            r.url = key
            r.domain = r.domain or _extract_domain(key)
            if not r.citations:
                r.citations = [key]
            merged[key] = r
            continue

        existing = merged[key]
        if len(r.snippet or "") > len(existing.snippet or ""):
            existing.snippet = r.snippet
        if len(r.title or "") > len(existing.title or ""):
            existing.title = r.title
        if r.source_provider not in existing.citations:
            existing.citations.append(r.source_provider)
        if key not in existing.citations:
            existing.citations.insert(0, key)
        existing.score = max(existing.score, r.score)
    return list(merged.values())


def _recency_score(published_at: str | None) -> float:
    if not published_at:
        return 0.0
    try:
        dt = _dt.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    age_days = max(0.0, (_dt.datetime.now(_dt.timezone.utc) - dt).total_seconds() / 86400.0)
    return max(0.0, 1.0 - min(age_days, 365.0) / 365.0)


def _lexical_overlap(query: str, title: str, snippet: str) -> float:
    q_terms = set(re.findall(r"[a-zA-Z0-9_]+", (query or "").lower()))
    q_terms = {t for t in q_terms if len(t) >= 3}
    if not q_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-zA-Z0-9_]+", f"{title} {snippet}".lower()))
    common = len(q_terms & text_terms)
    return common / max(1.0, float(len(q_terms)))


def rerank_results(results: List[SearchResultItem], query: str, mode: str = "balanced") -> List[SearchResultItem]:
    if not results:
        return []

    fresh_weight = 0.5 if mode == "fresh" else 0.2
    lexical_weight = 0.65 if mode != "deep" else 0.55
    source_weight = 0.15

    for r in results:
        lexical = _lexical_overlap(query, r.title, r.snippet)
        freshness = _recency_score(r.published_at)
        provider_bias = 1.0 if r.source_provider in {"tavily", "brave"} else 0.7
        r.score = round((lexical_weight * lexical) + (fresh_weight * freshness) + (source_weight * provider_bias), 6)

    ranked = sorted(results, key=lambda x: x.score, reverse=True)
    for i, r in enumerate(ranked, start=1):
        r.rank = i
        if not r.domain:
            r.domain = _extract_domain(r.url)
        if not r.citations:
            r.citations = [r.url]
    return ranked
