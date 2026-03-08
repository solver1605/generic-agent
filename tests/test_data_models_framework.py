import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import ToolMessage

from src.emergent_planner.config import AgentProfileConfig, ProfileDataModelPolicyConfig, default_agent_config
from src.emergent_planner.data_models import (
    DataModelValidationError,
    RegisteredDataModel,
    build_data_model_catalog,
    build_record,
    load_persisted_records,
    persist_record,
    resolve_data_models_for_profile,
    validate_instance,
)
from src.emergent_planner.nodes import activate_data_models_from_tool_result_node
from src.emergent_planner.tools import (
    get_data_model,
    list_data_models,
    request_data_model_fields,
    upsert_data_model,
)


class TestDataModelsFramework(TestCase):
    def test_builtin_user_profile_in_catalog(self):
        cfg = default_agent_config()
        catalog = build_data_model_catalog(cfg)
        ids = [m.id for m in catalog]
        self.assertIn("user_profile", ids)

    def test_resolve_data_models_for_profile_allow_and_deny(self):
        cfg = default_agent_config()
        catalog = build_data_model_catalog(cfg)
        profile = AgentProfileConfig(
            id="p1",
            data_models=ProfileDataModelPolicyConfig(allow=["user_profile"], deny=[]),
        )
        resolved = resolve_data_models_for_profile(catalog, profile)
        self.assertEqual([m.id for m in resolved], ["user_profile"])

    def test_custom_data_model_import_with_allowlist(self):
        cfg = default_agent_config()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg = root / "custom_models"
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "registry.py").write_text(
                """
from pydantic import BaseModel, ConfigDict
from src.emergent_planner.data_models import RegisteredDataModel

class ResearchContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    topic: str


def build_models():
    return [RegisteredDataModel(id='research_context', description='ctx', schema_cls=ResearchContext, context_fields=['topic'])]
""".strip(),
                encoding="utf-8",
            )

            sys.path.insert(0, root.as_posix())
            try:
                cfg.data_model_catalog.allow_module_prefixes = ["custom_models"]
                cfg.data_model_catalog.custom_imports = ["custom_models.registry:build_models"]
                catalog = build_data_model_catalog(cfg)
                ids = [m.id for m in catalog]
                self.assertIn("research_context", ids)
            finally:
                sys.path.remove(root.as_posix())

    def test_validate_and_persist_roundtrip(self):
        cfg = default_agent_config()
        catalog = build_data_model_catalog(cfg)
        model = next(m for m in catalog if m.id == "user_profile")

        out = validate_instance(model, {"name": "R", "place": "P"}, strict=True)
        self.assertEqual(out["name"], "R")

        with self.assertRaises(DataModelValidationError):
            validate_instance(model, {"name": "R", "unknown": "x"}, strict=True)

        record = build_record("user_profile", out, {"required_fields": []})
        with tempfile.TemporaryDirectory() as td:
            p = persist_record(record, user_id="default", root=Path(td))
            self.assertTrue(p.exists())
            values, updated = load_persisted_records(["user_profile"], user_id="default", root=Path(td))
            self.assertEqual(values["user_profile"]["name"], "R")
            self.assertTrue(updated["user_profile"])

    def test_data_model_tools_contract(self):
        state = {
            "runtime": {
                "config_path": "agent_config.yaml",
                "agent_profile_id": "default",
                "active_user_id": "default",
                "data_model_values": {},
                "data_model_meta": {},
                "data_model_last_updated": {},
            }
        }
        up = upsert_data_model.func(
            model_id="user_profile",
            payload={"name": "Alice", "place": "Paris"},
            merge=True,
            state=state,
        )
        self.assertEqual(up.get("__tool"), "data_model_upsert")
        rec = dict(up.get("record", {}) or {})
        model_values = {"user_profile": dict(rec.get("data", {}) or {})}

        state2 = {
            "runtime": {
                **state["runtime"],
                "data_model_values": model_values,
                "data_model_last_updated": {"user_profile": rec.get("updated_at")},
            }
        }
        got = get_data_model.func("user_profile", state=state2)
        self.assertEqual(got["data"]["name"], "Alice")

        listed = list_data_models.func(state=state2)
        self.assertGreaterEqual(listed["count"], 1)

    def test_request_data_model_fields_emits_and_upserts(self):
        state = {
            "runtime": {
                "config_path": "agent_config.yaml",
                "agent_profile_id": "default",
                "active_user_id": "default",
                "data_model_values": {},
                "data_model_meta": {},
                "data_model_last_updated": {},
            }
        }
        with patch("src.emergent_planner.tools.interrupt", return_value={"values": {"name": "Bob"}}):
            out = request_data_model_fields.func("user_profile", fields=["name"], state=state)
        self.assertEqual(out.get("__tool"), "data_model_upsert")
        self.assertEqual(out["record"]["data"]["name"], "Bob")

    def test_unknown_data_model_returns_structured_error(self):
        state = {
            "runtime": {
                "config_path": "agent_config.yaml",
                "agent_profile_id": "default",
                "active_user_id": "default",
                "data_model_values": {},
                "data_model_meta": {},
                "data_model_last_updated": {},
            }
        }
        got = get_data_model.func("finance_retirement_inputs", state=state)
        self.assertEqual(got.get("__tool"), "data_model_error")
        self.assertEqual(got.get("status"), "unknown_model")
        self.assertIn("user_profile", got.get("available_models", []))

        up = upsert_data_model.func(
            model_id="finance_retirement_inputs",
            payload={"name": "Alice"},
            merge=True,
            state=state,
        )
        self.assertEqual(up.get("__tool"), "data_model_error")
        self.assertEqual(up.get("status"), "unknown_model")

    def test_activate_data_models_node_merges_runtime(self):
        payload = {
            "__tool": "data_model_upsert",
            "model_id": "user_profile",
            "record": {
                "model_id": "user_profile",
                "data": {"name": "Zed"},
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "partial",
                "missing_required": [],
            },
        }
        state = {
            "history": [ToolMessage(content=json.dumps(payload), tool_call_id="tc1")],
            "runtime": {},
        }
        out = activate_data_models_from_tool_result_node(state)
        rt = out["runtime"]
        self.assertEqual(rt["data_model_values"]["user_profile"]["name"], "Zed")
        self.assertEqual(rt["data_model_last_updated"]["user_profile"], "2026-01-01T00:00:00+00:00")
