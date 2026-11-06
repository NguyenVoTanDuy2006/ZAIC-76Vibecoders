"""Task/argument/dependency reasoning integration; scripted models, not accuracy."""
from copy import deepcopy
import json
import unittest

from lifemate import agent
from lifemate.pipeline import task_reasoning
from lifemate.pipeline.tools import qwen_tools
from test_workflow import Backend, Env, call
from test_call_protocol import session


def reasoned(*replies, **options):
    return session(*replies, task_reasoning_enabled=True, **options)


class TaskReasoningTests(unittest.TestCase):
    def test_brief_is_generated_once_and_is_only_advice_in_actor_inputs(self):
        note = "1. Inspect inventory.inspect, then change value from the request after readiness succeeds."
        run = reasoned(note, call("inventory.inspect"), "Done", results=[{"ok": True}])
        first = run.propose()
        self.assertEqual(first["type"], "task_brief")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(run.backend.seen[0][2].max_new_tokens, 512)
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(run.call_state.plan_note, note)
        phases = [e["phase"] for e in run.trace if e["type"] == "model_reply"]
        self.assertEqual(phases, ["task_plan", "act", "act"])
        work = run.backend.seen[1][0][-1]["content"]
        self.assertIn(note, work)
        self.assertIn('"brief_is_evidence": false', work)
        self.assertNotIn(note, run.known_strings)
        self.assertEqual(run.first_model_input["messages"], run.backend.seen[0][0])

    def test_invalid_or_truncated_plan_falls_back_without_contract_repair(self):
        for bad, status in (("{bad json}", "unavailable"),
                            (agent.ModelReply("unfinished note", truncated=True), "truncated"),
                            ("", "unavailable")):
            with self.subTest(status=status):
                run = reasoned(bad, call("inventory.inspect"), "Done", results=[{"ok": True}])
                self.assertEqual(run.run()["status"], "finished")
                self.assertEqual(run.call_state.plan_status, status)
                self.assertEqual(run.invalid_replies, 0)
                self.assertEqual(run.call_state.plan_note, None)
                self.assertEqual(len(run.env.calls), 1)

    def test_native_call_instead_of_a_note_is_still_an_unexecuted_validated_proposal(self):
        run = reasoned(call("inventory.inspect"), "Done", results=[{"ok": True}])
        pending = run.propose()
        self.assertEqual(pending["type"], "call")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(run.call_state.plan_status, "call_proposed")
        self.assertEqual(run.propose(), pending)
        run.execute_pending()
        self.assertEqual(run.run()["status"], "finished")

    def test_wrong_planning_call_cannot_bypass_allowed_tools_or_provenance(self):
        run = reasoned(call("not.installed"), call("inventory.change", object_id="made-up"))
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])

    def test_brief_invented_id_is_not_a_value_source(self):
        run = reasoned("Change object_id=FAKE-3 after inspection.",
            call("inventory.change", object_id="FAKE-3"), call("inventory.inspect"),
            "Cannot perform the requested change", results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.invalid_replies, 1)
        self.assertNotIn("FAKE-3", run.known_strings)
        self.assertEqual([c["name"] for c in run.env.calls], ["inventory.inspect"])

    def test_missing_request_argument_can_trigger_lookup_without_executing_candidate(self):
        for domain in ("inventory", "documents", "transport"):
            with self.subTest(domain=domain):
                # value is schema-optional, but the user's desired effect needs it.
                run = reasoned("Inspect the requested value, then apply it.",
                    call(domain + ".change"), call(domain + ".inspect"),
                    call(domain + ".change", value=17), call(domain + ".change", value=17), "Done",
                    domain=domain, results=[{"ok": True, "requested_value": 17}, {"ok": True}])
                result = run.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual([c["name"] for c in run.env.calls], [domain + ".inspect", domain + ".change"])
                self.assertEqual(run.env.calls[1]["arguments"], {"value": 17})
                checks = [e for e in run.trace if e["type"] == "write_check"]
                self.assertEqual(len(checks), 2)
                context = run.backend.seen[2][0][-1]["content"]
                self.assertIn("valid_params", context)
                self.assertIn("request-required", context)
                self.assertEqual(run.call_state.write_checks, 2)

    def test_dependency_check_replaces_notification_with_prerequisite_then_notifies(self):
        run = reasoned("Change first; notify only after successful change.")
        notify = {"name": "inventory.notify", "kind": "write", "parameters": {
            "type": "object", "properties": {"content": {"type": "string"}}}}
        run.tools[notify["name"]] = notify
        run.request["context"]["installed_apps"]["inventory"]["tools"].append(notify["name"])
        run.native_tools = qwen_tools(list(run.tools.values()))
        run.backend = Backend("Change first; notify only after successful change.",
            call("inventory.notify", content="changed"), call("inventory.change", value=9),
            call("inventory.notify", content="changed"), call("inventory.notify", content="changed"), "Done")
        run.env = Env([{"ok": True, "value": 9}, {"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual([c["name"] for c in run.env.calls], ["inventory.change", "inventory.notify"])
        reviewed = json.dumps(run.backend.seen[4][0])
        self.assertIn("exec:1", reviewed)
        self.assertIn("success", reviewed)
        self.assertEqual(run.call_state.write_checks, 2)

    def test_write_replacement_still_passes_all_validation_and_does_not_recurse(self):
        run = reasoned("Change a known object only.", call("inventory.change", value=9),
                       call("inventory.change", object_id="invented"), call("absent.tool"))
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.call_state.write_checks, 1)
        self.assertEqual(run.env.calls, [])

    def test_write_preview_is_one_review_and_one_execution(self):
        run = reasoned("Apply the explicit requested change.", call("inventory.change", value=5),
                       call("inventory.change", value=5), results=[{"ok": True}])
        run.propose()  # brief
        pending = run.propose()
        self.assertEqual(pending, run.propose())
        self.assertEqual(run.model_calls, 3)
        self.assertEqual(run.call_state.write_checks, 1)
        self.assertEqual(run.env.calls, [])
        run.execute_pending()
        self.assertEqual(len(run.env.calls), 1)
        with self.assertRaises(agent.AgentInputError):
            run.execute_pending()

    def test_plan_and_write_review_share_global_generation_limit(self):
        for replies, maximum in ((("Brief",), 1), (("Brief", call("inventory.change", value=3)), 2)):
            run = reasoned(*replies, max_steps=maximum)
            self.assertEqual(run.run()["status"], "step_limit")
            self.assertEqual(run.model_calls, maximum)
            self.assertEqual(run.env.calls, [])

    def test_review_loop_cannot_exceed_call_budget_when_replacement_is_lookup(self):
        run = reasoned("Inspect and change.", call("inventory.change", value=3), call("inventory.inspect"),
                       call("inventory.change", value=3), max_calls=1, results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "call_limit")
        self.assertEqual(run.call_state.write_checks, 1)
        self.assertEqual(len(run.env.calls), 1)

    def test_terminal_checkpoint_uses_brief_and_receipts_and_does_not_double_review_write(self):
        run = reasoned("Inspect then change; successful change is required.", call("inventory.inspect"),
                       "Done", call("inventory.change", value=8), "Done", "Done",
                       terminal_check_enabled=True, results=[{"ok": True}, {"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(run.call_state.terminal_checks, 2)
        self.assertEqual(run.call_state.write_checks, 0)
        self.assertIn("TASK_WORK_CONTEXT", json.dumps(run.backend.seen[3][0]))
        self.assertFalse(result["diagnostics"]["call_protocol"]["semantic_completion_verified"])
        self.assertEqual(set(result["submission"]), {"id", "tool_calls", "state_hash"})

    def test_argument_view_is_catalog_derived_and_immutable(self):
        run = reasoned("Brief")
        candidate = {"name": "inventory.change", "arguments": {"value": 3}}
        before = deepcopy(run.tools)
        view = task_reasoning.argument_view(run, candidate)
        self.assertEqual(view["schema"]["required_params"], [])
        self.assertEqual(view["schema"]["valid_params"], ["object_id", "value"])
        view["call"]["arguments"]["value"] = 4
        self.assertEqual(candidate["arguments"]["value"], 3)
        self.assertEqual(run.tools, before)


if __name__ == "__main__":
    unittest.main()
