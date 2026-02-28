import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.search.engine import SearchRequest, run_search
from src.emergent_planner.search.types import SearchResultItem
from src.emergent_planner.tool_registry import select_tools, tool_name
from src.emergent_planner.tools import DEFAULT_TOOLS, search_web


CFG = """
default_model_card: gemini_flash_fast
model_cards:
  - id: gemini_flash_fast
    provider: google_genai
    model_name: models/gemini-3-flash-preview
search:
  providers:
    - name: tavily
      enabled: true
      api_key_env: TAVILY_API_KEY
      timeout_s: 2.0
      weight: 1.0
    - name: brave
      enabled: true
      api_key_env: BRAVE_API_KEY
      timeout_s: 2.0
      weight: 1.0
  budgets:
    max_providers_per_call: 2
    max_results_per_provider: 5
    max_total_results_before_rerank: 20
    max_enriched_results: 2
    global_timeout_s: 4.0
  defaults:
    default_top_k: 5
    default_recency_days: 30
    default_mode: balanced
    default_enrich: false
    provider_priority: [tavily, brave]
""".strip()


class TestSearchEngineAndRegistry(TestCase):
    def _cfg_file(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        p = Path(td.name) / "agent_config.yaml"
        p.write_text(CFG, encoding="utf-8")
        return p

    def test_run_search_provider_fallback(self):
        p = self._cfg_file()

        def fake_run_provider_query(provider_name, **kwargs):
            if provider_name == "tavily":
                raise RuntimeError("tavily failed")
            return [
                SearchResultItem(
                    title="Result",
                    url="https://example.com",
                    snippet="Useful content",
                    source_provider="brave",
                )
            ]

        with patch("src.emergent_planner.search.engine.run_provider_query", side_effect=fake_run_provider_query):
            resp = run_search(SearchRequest(query="best llm eval framework"), config_path=p)

        self.assertGreaterEqual(len(resp.results), 1)
        self.assertIn("brave", resp.providers_used)
        self.assertTrue(any(e.get("stage") == "provider" for e in resp.errors))

    def test_run_search_no_provider_keys_returns_structured_errors(self):
        p = self._cfg_file()
        resp = run_search(SearchRequest(query="open source observability tools"), config_path=p)
        self.assertEqual(resp.results, [])
        self.assertTrue(len(resp.errors) >= 1)

    def test_run_search_enforces_top_k_and_citations(self):
        p = self._cfg_file()

        def fake_run_provider_query(provider_name, **kwargs):
            rows = []
            for i in range(7):
                rows.append(
                    SearchResultItem(
                        title=f"R{i}",
                        url=f"https://example.com/{i}",
                        snippet="snippet",
                        source_provider=provider_name,
                    )
                )
            return rows

        with patch("src.emergent_planner.search.engine.run_provider_query", side_effect=fake_run_provider_query):
            resp = run_search(SearchRequest(query="vector db comparison", top_k=3), config_path=p)

        self.assertEqual(len(resp.results), 3)
        self.assertTrue(all(r.citations for r in resp.results))

    def test_search_web_tool_contract(self):
        class FakeResponse:
            def to_dict(self):
                return {
                    "query": "x",
                    "subqueries": ["x"],
                    "results": [{"rank": 1, "title": "t", "url": "u", "snippet": "s", "score": 1.0, "source_provider": "brave", "published_at": None, "domain": "d", "citations": ["u"], "enriched_text": None}],
                    "summary": "Top findings",
                    "providers_used": ["brave"],
                    "timings_ms": {"total": 1},
                    "budget": {"max_providers_per_call": 1},
                    "errors": [],
                }

        with patch("src.emergent_planner.tools.run_search", return_value=FakeResponse()):
            out = search_web.invoke({"query": "test"})

        self.assertIn("results", out)
        self.assertIn("citations", out["results"][0])

    def test_tool_registry_select_tools(self):
        names = [tool_name(t) for t in DEFAULT_TOOLS]
        self.assertIn("search_web", names)

        selected = select_tools(DEFAULT_TOOLS, ["read_file", "search_web"])
        selected_names = [tool_name(t) for t in selected]
        self.assertEqual(selected_names, ["read_file", "search_web"])
