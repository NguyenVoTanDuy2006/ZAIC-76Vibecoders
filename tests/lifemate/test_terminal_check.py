"""Terminal checks reuse BTC calls and the ordinary validator; scripted tests only."""
import json
import unittest

from lifemate import agent
from test_call_protocol import session
from test_workflow import call


def checked(*replies, **kwargs):
    return session(*replies, terminal_check_enabled=True, **kwargs)


class TerminalCheckTests(unittest.TestCase):
    def test_default_enables_bounded_checkpoint_without_old_control_json(self):
        self.assertTrue(agent.AgentConfig().terminal_check_enabled)
        run = checked("I will apply the change", call("inventory.change", value=8), "Done", "Done",
                      results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(run.env.calls), 1)
        self.assertEqual(run.call_state.terminal_checks, 2)
        self.assertEqual([e["phase"] for e in run.trace if e["type"] == "model_reply"],
                         ["act", "terminal_check", "act", "terminal_check"])
        self.assertTrue(run.trace[-1]["reviewed"])
        self.assertFalse(run.trace[-1]["semantic_completion_verified"])
        self.assertFalse(any(e["type"] == "control_reply" for e in run.trace))

    def test_valid_ask_can_be_replaced_with_available_lookup_before_execution(self):
        ask = call("sys.ask_missing_info", tool_name="inventory.change", params=["object_id"])
        run = checked(ask, call("inventory.inspect"), ask, ask,
                      results=[{"ok": True, "objects": []}, {"ok": True, "waiting_for_user": True}])
        self.assertEqual(run.run()["status"], "waiting_for_user")
        self.assertEqual([c["name"] for c in run.env.calls], ["inventory.inspect", "sys.ask_missing_info"])

    def test_invalid_ask_schema_uses_repair_before_spending_checkpoint_budget(self):
        bad = call("sys.ask_missing_info", tool_name="inventory.change", params=["phone"])
        good = call("sys.ask_missing_info", tool_name="inventory.change", params=["object_id"])
        run = checked(bad, good, good, results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "waiting_for_user")
        self.assertEqual(run.call_state.terminal_checks, 1)
        rejected = next(e for e in run.trace if e["type"] == "rejected")
        self.assertEqual(rejected["repair"]["code"], "invalid_missing_info_params")
        self.assertEqual(rejected["repair"]["version"], "catalog-call-repair-v1")

    def test_replacement_is_subject_to_provenance_and_schema_validation(self):
        run = checked("Done", call("inventory.change", object_id="guessed"), call("not.installed"))
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(run.invalid_replies, 2)
        self.assertEqual(run.call_state.terminal_checks, 1)

    def test_ticket_preview_is_reused_after_replacement_without_duplicate_generation(self):
        run = checked("Done", call("inventory.change", value=3), results=[{"ok": True}])
        first = run.propose()
        self.assertEqual(first, run.propose())
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.env.calls, [])
        run.execute_pending()
        self.assertEqual(len(run.env.calls), 1)

    def test_checkpoint_output_final_does_not_recursively_review_itself(self):
        run = checked("Done", "No requested writes were executed")
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.call_state.terminal_checks, 1)

    def test_exhausted_budget_stops_without_accepting_unchecked_final(self):
        run = checked("Done", call("inventory.inspect"), "Done", max_terminal_checks=1,
                      results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "checkpoint_limit")
        self.assertEqual(run.model_calls, 3)
        self.assertFalse(any(e["type"] == "finish" for e in run.trace))

    def test_checkpoint_shares_generation_budget(self):
        run = checked("Done", max_steps=1)
        self.assertEqual(run.run()["status"], "step_limit")
        self.assertEqual(run.model_calls, 1)
        self.assertEqual(run.env.calls, [])

    def test_checkpoint_timeout_and_truncation_cannot_execute_candidate(self):
        run = checked("Done", agent.ModelReply("partial", truncated=True), max_steps=2)
        self.assertEqual(run.run()["status"], "step_limit")
        self.assertEqual(run.env.calls, [])
        from test_workflow import Backend
        class TimedOut:
            def __init__(self):
                self.actor = Backend("Done")
                self.calls = 0
            def generate(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return self.actor.generate(*args, **kwargs)
                raise TimeoutError("elapsed")
        run = checked("Done")
        run.backend = TimedOut()
        self.assertEqual(run.run()["status"], "timeout")
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.env.calls, [])

    def test_prompt_contains_true_sources_and_never_promotes_candidate_to_evidence(self):
        run = checked("I used id FAKE-99", "Unable to finish")
        result = run.run()
        messages, tools, _ = run.backend.seen[1]
        self.assertIn("NOT executed", messages[0]["content"])
        self.assertIn(run.speech["current_text"], json.dumps(messages))
        self.assertIn("FAKE-99", messages[-1]["content"])
        self.assertNotIn("FAKE-99", run.known_strings)
        self.assertEqual({t["function"]["name"] for t in tools}, set(run.tools))
        self.assertEqual(result["submission"]["tool_calls"], [])


if __name__ == "__main__":
    unittest.main()
