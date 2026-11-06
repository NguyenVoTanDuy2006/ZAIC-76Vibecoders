"""Unified BTC protocol checks. Scripted backends are not real-model accuracy."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from lifemate import agent, workflow
from test_workflow import Backend, Env, call
from test_predictions import fixture
from lifemate.asr.predictions import write_speech_file


def session(*replies, domain="inventory", results=None, env=None, **options):
    tools = [
        {"name": domain + ".inspect", "kind": "read", "parameters": {"type": "object", "properties": {}}},
        {"name": domain + ".change", "kind": "write", "parameters": {"type": "object", "properties": {
            "value": {"type": "integer"}, "object_id": {"type": "string", "x-ref": domain + ".objects",
                "description": "ID from " + domain + ".inspect"}}}},
        {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object", "properties": {
            "tool_name": {"type": "string"}, "params": {"type": "array"}}}},
    ]
    request = {"id": "synthetic", "history": [], "context": {"installed_apps": {
        domain: {"tools": [t["name"] for t in tools[:-1]]}}}}
    speech = {"id": "synthetic", "current_text": "Inspect the requested object and apply the requested change if eligible.",
              "history_user_texts": []}
    # This suite isolates call tickets/repair. Terminal checkpoints have their own suite.
    options.setdefault("terminal_check_enabled", False)
    options.setdefault("task_reasoning_enabled", False)
    return agent.AgentSession(request, speech, tools, env or Env(results), Backend(*replies), agent.AgentConfig(**options))


class CallProtocolTests(unittest.TestCase):
    def test_unused_control_output_budget_cannot_block_unified_protocol(self):
        config = agent.AgentConfig(max_context_tokens=1024, max_new_tokens=128)
        self.assertEqual(workflow.protocol_version(config), workflow.UNIFIED_PROTOCOL_VERSION)
        with self.assertRaisesRegex(ValueError, "Control output"):
            agent.AgentConfig(unified_protocol=False, max_context_tokens=1024, max_new_tokens=128)

    def test_default_runner_saves_canonical_output_with_btc_terminal_checkpoint(self):
        with fixture() as (root, _, __, ___):
            speech = write_speech_file(root / "speech.jsonl", [
                {"id": "a", "current_text": "Previous sample", "history_user_texts": [
                    {"history_index": 1, "text": "Earlier request"}]},
                {"id": "b", "current_text": "List current events", "history_user_texts": []}])
            env = Env([{"ok": True, "events": []}])
            created = []
            def make_env(cid):
                created.append(cid)
                return env
            report = agent.run_full_pipeline(root, sample_ids=["b"], output_dir=root / "out",
                asr_predictions_path=speech, backend=Backend("List current events through calendar.list_events; no writes.",
                    call("calendar.list_events"), "Done", "Done"),
                toolenv_loader=lambda *_: make_env)
            self.assertEqual(created, ["b"])
            self.assertEqual(report["statuses"], {"b": "finished"})
            trace = json.loads(next(Path(report["trace_dir"]).glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(trace["diagnostics"]["workflow_protocol"], "btc-tool-call-v1")
            self.assertEqual(trace["model_calls"], 4)
            self.assertEqual(report["records"][0]["tool_calls"], env.get_tool_calls())
            self.assertEqual(set(report["records"][0]), {"id", "tool_calls", "state_hash"})

    def test_checkpoint_disabled_uses_only_actions_and_never_enters_old_controller(self):
        for domain in ("inventory", "documents", "transport"):
            with self.subTest(domain=domain), ExitStack() as stack:
                for target in ("lifemate.pipeline.adaptive.plan", "lifemate.pipeline.adaptive.approve",
                               "lifemate.pipeline.adaptive.decision", "lifemate.pipeline.recovery.report",
                               "lifemate.pipeline.controller.plan", "lifemate.pipeline.controller.review"):
                    stack.enter_context(patch(target, side_effect=AssertionError("Unexpected controller generation")))
                run = session(call(domain + ".inspect"), call(domain + ".change", object_id="observed-1", value=7), "Done",
                    domain=domain, results=[{"ok": True, "objects": [{"id": "observed-1"}]}, {"ok": True}])
                result = run.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual(result["model_calls"], 3)
                self.assertIsNone(run.controller_state)
                self.assertEqual(result["diagnostics"]["workflow_protocol"], "btc-tool-call-v1")
                self.assertEqual(result["submission"]["tool_calls"], run.env.get_tool_calls())
                self.assertFalse(any(e["type"] in ("control_reply", "recovery", "task_plan", "write_approval") for e in run.trace))
                for messages, tools, config in run.backend.seen:
                    self.assertEqual({t["function"]["name"] for t in tools}, set(run.tools))
                    self.assertEqual(config.max_new_tokens, 256)
                    self.assertNotIn("CONTROL PHASE", messages[0]["content"])

    def test_canonical_json_body_and_native_wrapper_have_identical_effect(self):
        canonical = {"name": "inventory.inspect", "arguments": {}}
        for raw in (json.dumps(canonical), "<tool_call>" + json.dumps(canonical) + "</tool_call>"):
            with self.subTest(raw=raw):
                run = session(raw, "Lookup complete", results=[{"ok": True, "objects": []}])
                result = run.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual(result["model_calls"], 2)
                self.assertEqual(result["submission"]["tool_calls"], [canonical])
                self.assertEqual(result["invalid_replies"], 0)

    def test_preview_reuses_the_ticket_and_execution_consumes_it_once(self):
        run = session(call("inventory.change", value=4), results=[{"ok": True}])
        first, again = run.propose(), run.propose()
        self.assertEqual(first, again)
        self.assertEqual(run.model_calls, 1)
        run.execute_pending()
        with self.assertRaises(agent.AgentInputError):
            run.execute_pending()
        self.assertEqual(len(run.env.calls), 1)
        self.assertEqual(run.call_state.receipts[0].execution_id, first["execution_id"])

    def test_changed_pending_call_never_reaches_toolenv(self):
        run = session(call("inventory.change", value=4), results=[{"ok": True}])
        run.propose()
        run.pending["call"]["arguments"]["value"] = 9
        with self.assertRaisesRegex(agent.AgentInputError, "Pending call was changed"):
            run.execute_pending()
        self.assertEqual(run.env.calls, [])

    def test_schema_and_old_controller_objects_have_one_bounded_rejection_path(self):
        for raw in ('{"goals":[],"steps":[],"reason":"plan"}',
                    '{"name":"inventory.inspect","arguments":"{}"}',
                    '{"name":"inventory.inspect","name":"other","arguments":{}}',
                    '{"name":"inventory.inspect","arguments":{"value":NaN}}',
                    '{"name":"inventory.inspect","arguments":{"value":1e999}}'):
            with self.subTest(raw=raw):
                run = session(raw, raw)
                result = run.run()
                self.assertEqual(result["status"], "invalid_output")
                self.assertEqual(result["model_calls"], 2)
                self.assertEqual(run.env.calls, [])
                self.assertFalse(any(e["type"] == "control_reply" for e in run.trace))

    def test_schema_rejection_uses_actual_validator_and_never_executes_bad_call(self):
        class Invalid(Env):
            def validate(self, name, arguments): return {"ok": False, "error": "value must be integer"}
        run = session(call("inventory.change", value="bad"), call("inventory.change", value="bad"), env=Invalid())
        result = run.run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["invalid_replies"], 2)
        self.assertEqual(run.env.calls, [])

    def test_lookup_can_follow_missing_reference_without_replan_json(self):
        run = session(call("inventory.change", object_id="guessed"), call("inventory.inspect"),
            call("inventory.change", object_id="real-id"), "Done", results=[
                {"ok": True, "objects": [{"id": "real-id"}]}, {"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["invalid_replies"], 1)
        self.assertEqual([c["name"] for c in run.env.calls], ["inventory.inspect", "inventory.change"])

    def test_asking_for_missing_data_is_an_official_call_and_stops_the_sample(self):
        run = session(call("sys.ask_missing_info", tool_name="inventory.change", params=["object_id"]),
                      results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "waiting_for_user")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(run.env.calls[0]["name"], "sys.ask_missing_info")

    def test_unknown_missing_parameter_is_rejected_before_execution(self):
        raw = call("sys.ask_missing_info", tool_name="inventory.change", params=["made_up"])
        run = session(raw, raw)
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])

    def test_no_op_write_cannot_be_promoted_to_finished_by_final_text(self):
        run = session(call("inventory.change", value=7), "Done", results=[{"ok": True, "done": False}])
        result = run.run()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(run.env.calls), 1)
        self.assertEqual(run.call_state.receipts[0].outcome, "no_op")

    def test_read_empty_result_is_data_and_does_not_create_an_automatic_write(self):
        run = session(call("inventory.inspect"), "No matching object found", results=[{"ok": True, "objects": []}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(len(run.env.calls), 1)

    def test_success_on_an_unrelated_write_does_not_erase_a_failed_write(self):
        run = session(call("inventory.change", value=7), call("inventory.change", value=9), "Done",
                      results=[{"ok": True, "done": False}, {"ok": True}])
        self.assertEqual(run.run()["status"], "blocked")
        self.assertEqual(len(run.call_state.unresolved_writes), 1)

    def test_confirmed_success_of_the_same_call_clears_its_known_no_op(self):
        run = session(call("inventory.change", value=7), call("inventory.change", value=7), "Done",
                      results=[{"ok": True, "done": False}, {"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.call_state.unresolved_writes, {})

    def test_repeated_identical_results_are_bounded_without_a_recovery_generation(self):
        run = session(*[call("inventory.inspect")] * 3,
                      results=[{"ok": True, "objects": []}] * 3)
        self.assertEqual(run.run()["status"], "no_progress")
        self.assertEqual(run.model_calls, 3)

    def test_uncertain_write_stops_before_any_blind_retry(self):
        class Uncertain(Env):
            def call(self, name, arguments):
                self.calls.append({"name": name, "arguments": deepcopy(arguments)})
                raise RuntimeError("Write outcome unavailable")
        run = session(call("inventory.change", value=7), call("inventory.change", value=7), env=Uncertain())
        result = run.run()
        self.assertEqual(result["status"], "execution_uncertain")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(len(run.env.calls), 1)
        self.assertEqual(result["submission"]["tool_calls"], run.env.get_tool_calls())

    def test_nonfinite_write_result_is_uncertain_and_receipts_remain_json_safe(self):
        run = session(call("inventory.change", value=7), results=[{"ok": True, "value": float("nan")}])
        result = run.run()
        self.assertEqual(result["status"], "execution_uncertain")
        json.dumps(result, allow_nan=False)

    def test_bad_canonical_call_log_is_detected(self):
        class BadLog(Env):
            def call(self, name, arguments): return {"ok": True}
        run = session(call("inventory.inspect"), env=BadLog())
        self.assertEqual(run.run()["status"], "execution_error")

    def test_receipts_are_immutable_and_diagnostics_do_not_expose_mutable_views(self):
        run = session(call("inventory.inspect"), "Done", results=[{"ok": True}])
        run.run()
        receipt = run.call_state.receipts[0]
        with self.assertRaises(FrozenInstanceError):
            receipt.outcome = "invented"
        snapshot = run.call_state.snapshot(); snapshot["receipts"][0]["result"]["ok"] = False
        self.assertTrue(receipt.snapshot()["result"]["ok"])

    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Optional official schema corpus check")
    def test_every_official_train_label_call_round_trips_through_the_same_parser(self):
        root = Path(os.environ["LIFEMATE_DATA_ROOT"])
        count = 0
        with (root / "train/train.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                for value in json.loads(line)["label"]["tool_calls"]:
                    source = json.dumps(value, ensure_ascii=False, allow_nan=False)
                    self.assertEqual(agent.parse_model_reply(source)["call"], value)
                    self.assertEqual(agent.parse_model_reply("<tool_call>" + source + "</tool_call>")["call"], value)
                    count += 1
        self.assertGreater(count, 0)

    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Optional official ToolEnv check")
    def test_unified_protocol_uses_one_real_toolenv_without_controller_generations(self):
        from datetime import datetime, timedelta
        root = Path(os.environ["LIFEMATE_DATA_ROOT"])
        row = agent.load_split_items(root, "train")[0]
        request = agent.prepare_request(row)
        tools = agent.select_available_tools(request, agent.load_catalog(root, "train"))
        if "weather.get_forecast" not in {t["name"] for t in tools}:
            self.skipTest("First sample has no weather lookup")
        speech = {"id": row["id"], "current_text": "Read the forecast for tomorrow.",
                  "history_user_texts": [{"history_index": i, "text": "Synthetic context"}
                     for i, turn in enumerate(request["history"]) if turn["role"] == "user"]}
        tomorrow = (datetime.fromisoformat(request["context"]["current_time"]) + timedelta(days=1)).date().isoformat()
        backend = Backend("Read forecast for tomorrow, no writes.", call("weather.get_forecast", date=tomorrow),
                          "Forecast received", "Forecast received")
        env = agent.load_toolenv(root, "train")(row["id"])
        result = agent.run_agent(request, speech, tools, env, backend=backend)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(len(env.get_tool_calls()), 1)
        self.assertEqual(result["submission"]["tool_calls"], env.get_tool_calls())
        self.assertEqual(result["submission"]["state_hash"], env.get_state_hash())


if __name__ == "__main__":
    unittest.main()
