"""Catalog-driven tool-call repair feedback; no model inference or automatic execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from lifemate.pipeline import call_repair
from test_call_protocol import session
from test_workflow import Env, call


def tools(domain="records"):
    return {
        domain + ".get": {"name": domain + ".get", "kind": "read", "parameters": {
            "type": "object", "properties": {"record_id": {"type": "string", "x-ref": domain + ".items",
                "description": "ID from " + domain + ".search"}}, "required": ["record_id"]}},
        domain + ".search": {"name": domain + ".search", "kind": "read", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}}},
        "messages.send": {"name": "messages.send", "kind": "write", "parameters": {
            "type": "object", "properties": {"contact_id": {"type": "string"}, "phone": {"type": "string"},
                "content": {"type": "string"}}, "required": ["content"], "x-one-of": [["contact_id"], ["phone"]]}},
        "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {
            "type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}},
            "required": ["tool_name", "params"]}},
    }


class CallRepairTests(unittest.TestCase):
    def test_missing_required_fields_and_typo_are_explained_from_schema(self):
        catalog = tools()
        report = call_repair.feedback({"name": "records.get", "arguments": {"record": "wrong-key"}}, catalog,
                                      message="Missing record_id")
        self.assertEqual(report["code"], "missing_required_arguments")
        self.assertEqual(report["call_schema"]["missing_required_params"], ["record_id"])
        self.assertEqual(report["call_schema"]["unexpected_argument_names"], ["record"])
        self.assertEqual(report["call_schema"]["lookup_hints"][0]["lookup_tools"], ["records.search"])
        self.assertFalse(report["executed"])

    def test_alternative_argument_groups_are_preserved_without_selecting_values(self):
        report = call_repair.feedback({"name": "messages.send", "arguments": {"content": "hello"}}, tools(), message="Missing recipient")
        self.assertEqual(report["code"], "missing_argument_alternative")
        self.assertEqual(report["call_schema"]["argument_alternatives"], [["contact_id"], ["phone"]])
        self.assertNotIn("corrected_call", report)

    def test_value_constraints_and_enums_are_preserved_but_examples_are_not(self):
        catalog = tools()
        catalog["records.get"]["parameters"]["properties"]["record_id"].update(enum=["a", "b"], examples=["SECRET_EXAMPLE"])
        report = call_repair.feedback({"name": "records.get", "arguments": {"record_id": "c"}}, catalog, message="Invalid enum")
        self.assertEqual(report["code"], "invalid_arguments")
        field = report["call_schema"]["parameters"]["properties"]["record_id"]
        self.assertEqual(field["enum"], ["a", "b"])
        self.assertNotIn("examples", field)

    def test_tool_selection_hints_only_contain_currently_allowed_tools(self):
        catalog = tools()
        report = call_repair.feedback({"name": "records.fetch", "arguments": {}}, catalog, message="Tool not allowed")
        self.assertEqual(report["code"], "tool_not_allowed")
        self.assertEqual(report["allowed_tools"], sorted(catalog))
        self.assertTrue(set(report["name_hints"]) <= set(catalog))
        self.assertNotIn("records.fetch", report["allowed_tools"])

    def test_installed_target_with_wrong_parameter_is_not_reported_as_disallowed(self):
        for domain in ("records", "documents", "inventory"):
            with self.subTest(domain=domain):
                catalog = tools(domain)
                value = {"name": "sys.ask_missing_info", "arguments": {"tool_name": domain + ".get", "params": ["phone"]}}
                problem = call_repair.missing_info_problem(value["arguments"], catalog)
                self.assertEqual(problem["code"], "invalid_missing_info_params")
                self.assertIn("is installed", problem["message"])
                report = call_repair.feedback(value, catalog, message=problem["message"])
                self.assertEqual(report["code"], "invalid_missing_info_params")
                self.assertEqual(report["field"], "arguments.params")
                self.assertEqual(report["invalid_params"], ["phone"])
                self.assertEqual(report["target_schema"]["valid_params"], ["record_id"])
                self.assertEqual(report["target_schema"]["required_params"], ["record_id"])
                self.assertNotIn("missing_required_params", report["target_schema"])
                self.assertEqual(report["target_schema"]["lookup_hints"][0]["lookup_tools"], [domain + ".search"])

    def test_missing_info_target_and_parameter_shape_errors_are_distinct(self):
        catalog = tools()
        cases = [({"tool_name": "unknown.get", "params": ["record_id"]}, "missing_info_tool_not_allowed"),
                 ({"tool_name": "records.get", "params": []}, "missing_info_params_shape"),
                 ({"tool_name": "records.get", "params": "record_id"}, "missing_info_params_shape"),
                 ({"tool_name": "records.get", "params": [{}]}, "invalid_missing_info_params")]
        for arguments, code in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(call_repair.missing_info_problem(arguments, catalog)["code"], code)
        self.assertIsNone(call_repair.missing_info_problem({"tool_name": "records.get", "params": ["record_id"]}, catalog))

    def test_feedback_does_not_modify_catalog_arguments_or_invent_a_corrected_call(self):
        catalog = tools(); value = {"name": "records.get", "arguments": {}}
        before = deepcopy((catalog, value))
        call_repair.feedback(value, catalog, message="Missing ID")
        self.assertEqual((catalog, value), before)

    def test_model_can_repropose_a_real_missing_info_call_with_the_same_protocol(self):
        run = session(call("sys.ask_missing_info", tool_name="inventory.change", params=["phone"]),
                      call("sys.ask_missing_info", tool_name="inventory.change", params=["object_id"]),
                      results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "waiting_for_user")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(len(run.env.calls), 1)
        self.assertEqual(run.env.calls[0]["arguments"]["params"], ["object_id"])
        rejected = next(e for e in result["trace"] if e["type"] == "rejected")
        self.assertEqual(rejected["repair"]["code"], "invalid_missing_info_params")
        self.assertEqual(rejected["repair"]["invalid_params"], ["phone"])
        self.assertEqual(rejected["repair"]["budget"]["remaining_before_stop"], 1)
        seen = [json.loads(m["content"]) for m in run.backend.seen[1][0] if m["role"] == "tool"]
        self.assertEqual(seen[0]["repair"]["target_schema"]["valid_params"], ["object_id", "value"])
        self.assertFalse(any(e["type"] in ("control_reply", "recovery") for e in result["trace"]))

    def test_schema_verdict_remains_authoritative_and_feedback_does_not_execute(self):
        class Invalid(Env):
            def validate(self, name, arguments): return {"ok": False, "error": "integer value required"}
        run = session(call("inventory.change", value="bad"), call("inventory.change", value="bad"), env=Invalid())
        result = run.run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])
        rejected = [e for e in result["trace"] if e["type"] == "rejected"]
        self.assertEqual([e["repair"]["budget"]["remaining_before_stop"] for e in rejected], [1, 0])
        self.assertEqual(rejected[-1]["repair"]["validator_message"], "Sai schema: integer value required")

    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Optional offline missing-info corpus check")
    def test_official_ask_calls_keep_their_target_and_input_field_semantics(self):
        root = Path(os.environ["LIFEMATE_DATA_ROOT"]) / "train"
        catalog = json.loads((root / "train_tools_catalog.json").read_text(encoding="utf-8"))
        all_tools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
        count = 0
        with (root / "train.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                row = json.loads(line)
                allowed = {n for app in row["context"]["installed_apps"].values() for n in app["tools"]} | {"sys.ask_missing_info"}
                selected = {n: all_tools[n] for n in allowed}
                for value in row["label"]["tool_calls"]:
                    if value["name"] == "sys.ask_missing_info":
                        self.assertIsNone(call_repair.missing_info_problem(value["arguments"], selected), row["id"])
                        count += 1
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
