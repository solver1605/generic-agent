"""
Provider adapters for Tavily and Brave web search.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import SearchResultItem


class SearchProviderError(RuntimeError):
    pass


@dataclass
class ProviderConfig:
    name: str
    api_key_env: str
    timeout_s: float = 8.0
    weight: float = 1.0
    enabled: bool = True


def _http_json(method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]], timeout_s: float) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url=url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise SearchProviderError(f"HTTP {e.code} from {url}: {msg[:300]}") from e
    except Exception as e:
        raise SearchProviderError(f"Request failed for {url}: {type(e).__name__}: {e}") from e


def _map_tavily(data: Dict[str, Any], max_items: int) -> List[SearchResultItem]:
    out: List[SearchResultItem] = []
    for row in (data.get("results") or [])[:max_items]:
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        out.append(
            SearchResultItem(
                title=str(row.get("title", "")).strip() or url,
                url=url,
                snippet=str(row.get("content", "")).strip(),
                source_provider="tavily",
                published_at=row.get("published_date") or row.get("published_at"),
                citations=[url],
            )
        )
    return out


def _map_brave(data: Dict[str, Any], max_items: int) -> List[SearchResultItem]:
    out: List[SearchResultItem] = []
    rows = (((data.get("web") or {}).get("results")) or [])[:max_items]
    for row in rows:
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        out.append(
            SearchResultItem(
                title=str(row.get("title", "")).strip() or url,
                url=url,
                snippet=str(row.get("description", "")).strip(),
                source_provider="brave",
                published_at=row.get("age"),
                citations=[url],
            )
        )
    return out


def search_tavily(query: str, api_key: str, *, top_k: int, recency_days: Optional[int], timeout_s: float) -> List[SearchResultItem]:
    payload: Dict[str, Any] = {
        "query": query,
        "max_results": int(max(1, top_k)),
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    if recency_days is not None:
        payload["days"] = int(max(1, recency_days))

    data = _http_json(
        method="POST",
        url="https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
        timeout_s=timeout_s,
    )
    return _map_tavily(data, max_items=top_k)


def search_brave(query: str, api_key: str, *, top_k: int, recency_days: Optional[int], timeout_s: float) -> List[SearchResultItem]:
    params: Dict[str, Any] = {
        "q": query,
        "count": int(max(1, min(20, top_k))),
        "search_lang": "en",
        "country": "us",
    }
    if recency_days is not None:
        if recency_days <= 1:
            params["freshness"] = "pd"
        elif recency_days <= 7:
            params["freshness"] = "pw"
        elif recency_days <= 31:
            params["freshness"] = "pm"
        else:
            params["freshness"] = "py"

    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params)
    data = _http_json(
        method="GET",
        url=url,
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        payload=None,
        timeout_s=timeout_s,
    )
    return _map_brave(data, max_items=top_k)


def run_provider_query(
    provider_name: str,
    *,
    query: str,
    top_k: int,
    recency_days: Optional[int],
    timeout_s: float,
    api_key_env: str,
) -> List[SearchResultItem]:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SearchProviderError(f"Missing API key env var: {api_key_env}")

    name = provider_name.lower().strip()
    if name == "tavily":
        return search_tavily(query, api_key, top_k=top_k, recency_days=recency_days, timeout_s=timeout_s)
    if name == "brave":
        return search_brave(query, api_key, top_k=top_k, recency_days=recency_days, timeout_s=timeout_s)
    raise SearchProviderError(f"Unsupported provider: {provider_name}")
