from unittest import TestCase

from src.emergent_planner.search.ranking import canonicalize_url, dedupe_results, rerank_results
from src.emergent_planner.search.types import SearchResultItem


class TestSearchRanking(TestCase):
    def test_canonicalize_url_removes_tracking_and_fragment(self):
        url = "https://www.Example.com/path/?utm_source=x&a=1&fbclid=abc#section"
        out = canonicalize_url(url)
        self.assertEqual(out, "https://example.com/path?a=1")

    def test_dedupe_results_merges_same_url(self):
        a = SearchResultItem(
            title="A",
            url="https://example.com/x?utm_source=ad",
            snippet="short",
            source_provider="tavily",
        )
        b = SearchResultItem(
            title="A extended",
            url="https://www.example.com/x",
            snippet="longer snippet",
            source_provider="brave",
        )
        merged = dedupe_results([a, b])
        self.assertEqual(len(merged), 1)
        self.assertIn("longer snippet", merged[0].snippet)
        self.assertIn("https://example.com/x", merged[0].citations)

    def test_rerank_results_is_deterministic_and_sets_rank(self):
        rows = [
            SearchResultItem(title="Python docs", url="https://docs.python.org", snippet="Official docs", source_provider="tavily"),
            SearchResultItem(title="Weather today", url="https://weather.example", snippet="forecast", source_provider="brave"),
        ]
        ranked = rerank_results(rows, query="python official documentation", mode="deep")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].rank, 2)
        self.assertGreaterEqual(ranked[0].score, ranked[1].score)
