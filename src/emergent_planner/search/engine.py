"""
Search engine orchestrator.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import SearchBudgetConfig, SearchDefaults, SearchProviderConfig, load_agent_config
from .enrich import enrich_results
from .planner import expand_subqueries, normalize_query
from .providers import SearchProviderError, run_provider_query
from .ranking import dedupe_results, rerank_results
from .types import SearchBudgetReport, SearchMode, SearchResponse, SearchResultItem


@dataclass
class SearchRequest:
    query: str
    top_k: int = 8
    recency_days: Optional[int] = None
    mode: SearchMode = "balanced"
    enrich: bool = False
    max_enriched_results: int = 3
    provider_preference: Optional[List[str]] = None
    timeout_s: float = 12.0


def _compact_summary(results: List[SearchResultItem], limit: int = 3) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results[:limit], start=1):
        lines.append(f"{i}. {r.title} ({r.domain}) — {r.url}")
    return "Top findings:\n" + "\n".join(lines)


def _provider_map(config_providers: List[SearchProviderConfig]) -> Dict[str, SearchProviderConfig]:
    return {p.name.lower(): p for p in config_providers if p.enabled}


def _pick_providers(
    provider_preference: Optional[List[str]],
    defaults: SearchDefaults,
    config_providers: List[SearchProviderConfig],
    budget: SearchBudgetConfig,
) -> List[SearchProviderConfig]:
    pmap = _provider_map(config_providers)
    names = [n.lower() for n in (provider_preference or defaults.provider_priority)]
    selected: List[SearchProviderConfig] = []
    for n in names:
        if n in pmap and pmap[n] not in selected:
            selected.append(pmap[n])
        if len(selected) >= max(1, budget.max_providers_per_call):
            break
    return selected


def run_search(req: SearchRequest, *, config_path: Path = Path("agent_config.yaml")) -> SearchResponse:
    t0 = time.perf_counter()
    cfg = load_agent_config(config_path)
    search_cfg = cfg.search

    query = normalize_query(req.query)
    if not query:
        budget = SearchBudgetReport(
            max_providers_per_call=search_cfg.budgets.max_providers_per_call,
            max_results_per_provider=search_cfg.budgets.max_results_per_provider,
            max_total_results_before_rerank=search_cfg.budgets.max_total_results_before_rerank,
            max_enriched_results=search_cfg.budgets.max_enriched_results,
            global_timeout_s=search_cfg.budgets.global_timeout_s,
        )
        return SearchResponse(
            query="",
            subqueries=[],
            results=[],
            summary="Query is empty.",
            providers_used=[],
            timings_ms={"total": int((time.perf_counter() - t0) * 1000)},
            budget=budget,
            errors=[{"stage": "input", "error": "Query cannot be empty."}],
        )

    top_k = max(1, min(int(req.top_k or search_cfg.defaults.default_top_k), 20))
    mode: SearchMode = req.mode if req.mode in {"balanced", "fresh", "deep"} else search_cfg.defaults.default_mode
    recency_days = req.recency_days if req.recency_days is not None else search_cfg.defaults.default_recency_days
    enrich = bool(req.enrich if req.enrich is not None else search_cfg.defaults.default_enrich)

    providers = _pick_providers(req.provider_preference, search_cfg.defaults, search_cfg.providers, search_cfg.budgets)

    budget_report = SearchBudgetReport(
        max_providers_per_call=search_cfg.budgets.max_providers_per_call,
        max_results_per_provider=search_cfg.budgets.max_results_per_provider,
        max_total_results_before_rerank=search_cfg.budgets.max_total_results_before_rerank,
        max_enriched_results=search_cfg.budgets.max_enriched_results,
        global_timeout_s=min(float(req.timeout_s), float(search_cfg.budgets.global_timeout_s)),
    )

    if not providers:
        return SearchResponse(
            query=query,
            subqueries=[query],
            results=[],
            summary="No enabled providers found.",
            providers_used=[],
            timings_ms={"total": int((time.perf_counter() - t0) * 1000)},
            budget=budget_report,
            errors=[{"stage": "provider_selection", "error": "No providers available after filtering."}],
        )

    subqueries = expand_subqueries(query, mode=mode, max_subqueries=4)
    providers_used: List[str] = []
    errors: List[dict] = []
    aggregated: List[SearchResultItem] = []

    stage_start = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, len(providers) * max(1, len(subqueries)))) as ex:
        for provider in providers:
            for sq in subqueries:
                futures.append(
                    ex.submit(
                        run_provider_query,
                        provider.name,
                        query=sq,
                        top_k=min(top_k, search_cfg.budgets.max_results_per_provider),
                        recency_days=recency_days,
                        timeout_s=min(provider.timeout_s, budget_report.global_timeout_s),
                        api_key_env=provider.api_key_env,
                    )
                )

        for fut in as_completed(futures, timeout=budget_report.global_timeout_s):
            try:
                rows = fut.result()
                if rows:
                    aggregated.extend(rows)
                    providers_used.extend({r.source_provider for r in rows})
            except SearchProviderError as e:
                errors.append({"stage": "provider", "error": str(e)})
            except Exception as e:
                errors.append({"stage": "provider", "error": f"{type(e).__name__}: {e}"})

    budget_report.providers_called = len(providers)
    budget_report.candidates_before_dedupe = len(aggregated)

    if len(aggregated) > search_cfg.budgets.max_total_results_before_rerank:
        aggregated = aggregated[: search_cfg.budgets.max_total_results_before_rerank]

    deduped = dedupe_results(aggregated)
    budget_report.candidates_after_dedupe = len(deduped)

    ranked = rerank_results(deduped, query=query, mode=mode)
    ranked = ranked[:top_k]

    fetch_ms = int((time.perf_counter() - stage_start) * 1000)

    enrich_ms = 0
    if enrich and ranked:
        s2 = time.perf_counter()
        max_enrich = min(int(req.max_enriched_results), search_cfg.budgets.max_enriched_results)
        budget_report.enrichment_attempted = max_enrich
        ranked, enrich_errors, success = enrich_results(
            ranked,
            max_enriched=max_enrich,
            timeout_s=min(5.0, budget_report.global_timeout_s),
        )
        budget_report.enrichment_succeeded = success
        errors.extend(enrich_errors)
        enrich_ms = int((time.perf_counter() - s2) * 1000)

    total_ms = int((time.perf_counter() - t0) * 1000)
    summary = _compact_summary(ranked)

    if not ranked and not errors:
        errors.append({"stage": "search", "error": "No results found from configured providers."})

    return SearchResponse(
        query=query,
        subqueries=subqueries,
        results=ranked,
        summary=summary,
        providers_used=sorted(set(providers_used)),
        timings_ms={"fetch": fetch_ms, "enrich": enrich_ms, "total": total_ms},
        budget=budget_report,
        errors=errors,
    )
