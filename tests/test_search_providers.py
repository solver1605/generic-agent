from unittest import TestCase
from unittest.mock import patch

from src.emergent_planner.search.providers import _map_brave, _map_tavily, run_provider_query, search_brave


class TestSearchProviders(TestCase):
    def test_map_tavily_payload(self):
        data = {
            "results": [
                {
                    "title": "T1",
                    "url": "https://example.com/1",
                    "content": "Snippet",
                    "published_date": "2026-01-01T00:00:00Z",
                }
            ]
        }
        out = _map_tavily(data, max_items=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source_provider, "tavily")
        self.assertEqual(out[0].citations, ["https://example.com/1"])

    def test_map_brave_payload(self):
        data = {
            "web": {
                "results": [
                    {
                        "title": "B1",
                        "url": "https://example.com/2",
                        "description": "Snippet",
                        "age": "2026-01-02T00:00:00Z",
                    }
                ]
            }
        }
        out = _map_brave(data, max_items=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source_provider, "brave")
        self.assertEqual(out[0].citations, ["https://example.com/2"])

    def test_search_brave_recency_mapping(self):
        captured = {}

        def fake_http_json(method, url, headers, payload, timeout_s):
            captured["url"] = url
            return {"web": {"results": []}}

        with patch("src.emergent_planner.search.providers._http_json", side_effect=fake_http_json):
            search_brave("query", "key", top_k=5, recency_days=2, timeout_s=3.0)

        self.assertIn("freshness=pw", captured["url"])

    def test_run_provider_query_missing_api_key_raises(self):
        with patch("src.emergent_planner.search.providers.os.environ.get", return_value=""):
            with self.assertRaises(Exception):
                run_provider_query(
                    "tavily",
                    query="test",
                    top_k=3,
                    recency_days=None,
                    timeout_s=1.0,
                    api_key_env="TAVILY_API_KEY",
                )
