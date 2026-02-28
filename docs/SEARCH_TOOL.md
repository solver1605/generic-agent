# Search Tool (`search_web`)

## Overview

`search_web` is a web-first agent tool that performs multi-provider search with fallback, query expansion, dedupe, reranking, optional page enrichment, and citation-rich output.

Providers supported in v1:
- `tavily`
- `brave`

## Tool Signature

```python
search_web(
  query: str,
  top_k: int = 8,
  recency_days: Optional[int] = None,
  mode: Literal["balanced", "fresh", "deep"] = "balanced",
  enrich: bool = False,
  max_enriched_results: int = 3,
  provider_preference: Optional[List[str]] = None,
  timeout_s: float = 12.0,
) -> Dict[str, Any]
```

## Response Shape

```json
{
  "query": "...",
  "subqueries": ["..."],
  "results": [
    {
      "rank": 1,
      "title": "...",
      "url": "...",
      "snippet": "...",
      "score": 0.87,
      "source_provider": "tavily",
      "published_at": "2026-02-28T00:00:00Z",
      "domain": "example.com",
      "citations": ["https://example.com/article"],
      "enriched_text": "..."
    }
  ],
  "summary": "Top findings...",
  "providers_used": ["tavily", "brave"],
  "timings_ms": {"fetch": 512, "enrich": 130, "total": 712},
  "budget": {
    "max_providers_per_call": 2,
    "max_results_per_provider": 8,
    "max_total_results_before_rerank": 40,
    "max_enriched_results": 3,
    "global_timeout_s": 12.0,
    "providers_called": 2,
    "candidates_before_dedupe": 30,
    "candidates_after_dedupe": 22,
    "enrichment_attempted": 2,
    "enrichment_succeeded": 2
  },
  "errors": []
}
```

## Config (`agent_config.yaml`)

```yaml
search:
  providers:
    - name: tavily
      enabled: true
      api_key_env: TAVILY_API_KEY
      timeout_s: 8.0
      weight: 1.0
    - name: brave
      enabled: true
      api_key_env: BRAVE_API_KEY
      timeout_s: 8.0
      weight: 1.0
  budgets:
    max_providers_per_call: 2
    max_results_per_provider: 8
    max_total_results_before_rerank: 40
    max_enriched_results: 3
    global_timeout_s: 12.0
  defaults:
    default_top_k: 8
    default_recency_days: 30
    default_mode: balanced
    default_enrich: false
    provider_priority: [tavily, brave]
```

## Required Environment Variables

- `TAVILY_API_KEY` for Tavily provider.
- `BRAVE_API_KEY` for Brave provider.

If a provider key is missing, the tool records a non-fatal provider error and falls back to other configured providers.

## Failure Semantics

- Provider failures are returned in `errors` and do not fail the whole call unless no results are available.
- Timeout and budget caps can reduce fanout/enrichment; tool still returns best available results.
- Empty query returns structured validation error.
