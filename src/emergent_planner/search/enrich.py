"""
Optional URL content enrichment.
"""
from __future__ import annotations

import html
import re
import urllib.request
from typing import List, Tuple

from .types import SearchResultItem


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _extract_text_from_html(raw_html: str) -> str:
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    txt = _TAG_RE.sub(" ", txt)
    txt = html.unescape(txt)
    txt = _WS_RE.sub(" ", txt).strip()
    return txt


def fetch_url_text(url: str, timeout_s: float = 5.0, max_chars: int = 6000) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "EmergentPlannerSearch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        body = resp.read(512_000).decode("utf-8", errors="replace")
    if "html" in ctype or "<html" in body.lower():
        text = _extract_text_from_html(body)
    else:
        text = body
    return text[:max_chars]


def enrich_results(
    results: List[SearchResultItem],
    *,
    max_enriched: int,
    timeout_s: float,
) -> Tuple[List[SearchResultItem], List[dict], int]:
    errors: List[dict] = []
    success = 0
    cap = max(0, max_enriched)
    for i, r in enumerate(results):
        if i >= cap:
            break
        try:
            r.enriched_text = fetch_url_text(r.url, timeout_s=timeout_s)
            success += 1
        except Exception as e:
            errors.append({"stage": "enrich", "url": r.url, "error": f"{type(e).__name__}: {e}"})
    return results, errors, success
