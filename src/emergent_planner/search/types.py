"""
Shared search datatypes and lightweight validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


SearchMode = Literal["balanced", "fresh", "deep"]


@dataclass
class SearchResultItem:
    title: str
    url: str
    snippet: str
    source_provider: str
    published_at: Optional[str] = None
    score: float = 0.0
    rank: int = 0
    domain: str = ""
    citations: List[str] = field(default_factory=list)
    enriched_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchBudgetReport:
    max_providers_per_call: int
    max_results_per_provider: int
    max_total_results_before_rerank: int
    max_enriched_results: int
    global_timeout_s: float

    providers_called: int = 0
    candidates_before_dedupe: int = 0
    candidates_after_dedupe: int = 0
    enrichment_attempted: int = 0
    enrichment_succeeded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResponse:
    query: str
    subqueries: List[str]
    results: List[SearchResultItem]
    summary: str
    providers_used: List[str]
    timings_ms: Dict[str, int]
    budget: SearchBudgetReport
    errors: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "subqueries": self.subqueries,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "providers_used": self.providers_used,
            "timings_ms": self.timings_ms,
            "budget": self.budget.to_dict(),
            "errors": self.errors,
        }
