"""Planner contract regressions; scripted CPU checks, no model inference."""
from copy import deepcopy
import json
import unittest

from lifemate.pipeline import adaptive, planning, proposals
from test_recovery_controller import (TEXT, approve, call, condition, decide,
                                      finish, goal, session, step)


def concise(steps, goals=None):
    return json.dumps({"reason": "Next grounded steps", "goals": goals or [goal()], "steps": steps})


def minimal(sid="read", kind="read", domain="inventory", deps=None):
    value = {"id": sid, "goal_id": "g", "kind": kind,
             "tool": domain + (".inspect" if kind == "read" else ".change") if kind in ("read", "write") else None}
    if deps is not None:
        value["depends_on"] = deps
    return value


class ProposalTests(unittest.TestCase):
    def test_minimal_read_reaches_execution_without_model_generated_metadata(self):
        run = session(concise([minimal()], [goal(kind="read")]), call("inventory.inspect"),
                      review_finish=False, results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(len(run.env.calls), 1)
        self.assertFalse(any(e["type"] == "recovery" for e in run.trace))
        installed = run.controller_state.steps["read"]
        self.assertEqual(installed, step())
        self.assertIn("base_version", run.trace[1]["compiled_defaults"])
        payload = json.loads(run.backend.seen[0][0][-1]["content"])["payload"]
        self.assertEqual(payload["proposal_contract"]["base_version_owned_by_code"], 0)
        self.assertNotIn("base_version", payload["proposal_contract"]["required"])

    def test_write_identity_is_generated_and_preserved_on_revision(self):
        raw = json.loads(concise([minimal("write", "write")]))
        before = deepcopy(raw)
        compiled, defaults = proposals.compile_proposal(raw, plan_version=0)
        op = compiled["steps"][0]["operation_id"]
        self.assertTrue(op.startswith("op-"))
        self.assertEqual(raw, before)
        self.assertIn("steps[0].operation_id", defaults)
        prior = {"write": {**compiled["steps"][0], "operation_id": "older-explicit-identity"}}
        revised, _ = proposals.compile_proposal(raw, plan_version=3, previous_steps=prior)
        self.assertEqual(revised["base_version"], 3)
        self.assertEqual(revised["steps"][0]["operation_id"], "older-explicit-identity")

    def test_generated_new_identity_cannot_replay_a_successful_write(self):
        run = session(concise([minimal("first", "write")]), call("inventory.change", value=7),
                      approve("first"), results=[{"ok": True}])
        run.step(); run.step(); run.step()
        state = run.controller_state
        value, _ = proposals.compile_proposal(json.loads(concise([
            minimal("first", "write"), minimal("new", "write")])),
            plan_version=1, previous_steps=state.steps)
        goals, steps = planning.parse_plan(value, sources=state.sources, tools=run.tools,
                                          max_tasks=12, previous=state)
        state.install(goals, steps, "Add a different operation")
        with self.assertRaisesRegex(planning.ControllerError, "cannot replay"):
            state.guard("new", {"name": "inventory.change", "arguments": {"value": 7}}, 2)
        self.assertEqual(len(run.env.calls), 1)

    def test_explicit_stale_or_mistyped_version_is_never_coerced(self):
        for version in ("0", "sample_id", True, 2):
            value = json.loads(concise([minimal()]))
            value["base_version"] = version
            with self.subTest(version=version), self.assertRaises(planning.ControllerError) as caught:
                proposals.compile_proposal(value, plan_version=0)
            self.assertEqual(caught.exception.code, "stale_version")

    def test_trace_shape_faults_are_reported_together_without_installation(self):
        bad = minimal()
        bad.update(depends_on=None, postcondition="The lookup returned its data", operation_id="op-read")
        value = json.loads(concise([bad], [goal(kind="read")]))
        value["base_version"] = "request-id"
        before = deepcopy(value)
        with self.assertRaises(planning.ControllerError) as caught:
            proposals.compile_proposal(value, plan_version=0)
        issues = caught.exception.as_dict()["violations"]
        self.assertEqual({i["field"] for i in issues}, {
            "base_version", "steps[0].depends_on", "steps[0].postcondition", "steps[0].operation_id"})
        self.assertIn("use []", str(caught.exception))
        self.assertEqual(value, before)
        run = session(json.dumps(value))
        run.step()
        self.assertEqual(run.controller_state.version, 0)
        self.assertEqual(run.controller_state.steps, {})
        self.assertEqual(run.env.calls, [])
        self.assertEqual(run.trace[-1]["issue"]["violations"], issues)

    def test_bad_optional_values_are_rejected_instead_of_discarded(self):
        for deps in (None, "read", ["read", "read"], [None]):
            value = json.loads(concise([{**minimal(), "depends_on": deps}]))
            with self.subTest(deps=deps), self.assertRaises(planning.ControllerError):
                proposals.compile_proposal(value, plan_version=0)
        bad = minimal("write", "write")
        bad["when"] = condition(value=True)
        bad["when"]["predicate"]["right"] = True
        with self.assertRaises(planning.ControllerError) as caught:
            proposals.compile_proposal(json.loads(concise([minimal(), bad])), plan_version=0)
        self.assertEqual(caught.exception.field, "steps[1].when.predicate")

    def test_compiler_defaults_do_not_bypass_semantic_validation(self):
        cases = [
            (minimal(domain="invented"), goal(kind="read")),
            ({**minimal(), "when": condition()}, goal(kind="read")),
            (minimal(deps=["absent"]), goal(kind="read")),
            (minimal(), goal(kind="write")),
        ]
        for candidate, target in cases:
            with self.subTest(candidate=candidate):
                run = session(concise([candidate], [target]))
                run.step()
                self.assertEqual(run.controller_state.version, 0)
                self.assertEqual(run.env.calls, [])
                self.assertEqual(run.trace[-1]["type"], "recovery")

    def test_missing_fields_and_unknown_fields_have_precise_paths(self):
        candidate = minimal(); candidate.pop("tool"); candidate["status"] = "done"
        with self.assertRaises(planning.ControllerError) as caught:
            proposals.compile_proposal(json.loads(concise([candidate])), plan_version=0)
        self.assertEqual(caught.exception.field, "steps[0]")
        self.assertIn("tool", str(caught.exception))
        self.assertIn("status", str(caught.exception))


class IncrementalPlanningTests(unittest.TestCase):
    def test_initial_distinct_deferred_operations_retain_their_reserved_identities(self):
        first = [minimal(), minimal("one", "deferred", deps=["read"]),
                 minimal("two", "deferred", deps=["read"])]
        later = [minimal(), minimal("one", "write", deps=["read"]),
                 minimal("two", "write", deps=["one"])]
        run = session(concise(first), call("inventory.inspect"), decide("replan"), concise(later),
            call("inventory.change", value=7), approve("one", 2),
            call("inventory.change", value=7), approve("two", 2), finish(2),
            results=[{"ok": True}, {"ok": True}, {"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(len(run.env.calls), 3)
        self.assertEqual(run.controller_state.steps["one"]["operation_id"], proposals.operation_id("one"))
        self.assertEqual(run.controller_state.steps["two"]["operation_id"], proposals.operation_id("two"))

    def test_deferred_goal_waits_for_observations_then_becomes_a_guarded_write(self):
        for domain in ("inventory", "documents", "transport"):
            with self.subTest(domain=domain):
                first = [minimal(domain=domain), minimal("write", "deferred", deps=["read"])]
                later = [minimal(domain=domain), {**minimal("write", "write", domain, ["read"]),
                                                  "when": condition(), "expected_arguments": {"value": 7}}]
                run = session(concise(first), call(domain + ".inspect"), decide("replan"),
                    concise(later), call(domain + ".change", value=7), approve(version=2), finish(2),
                    domain=domain, results=[{"ok": True, "eligible": True}, {"ok": True}])
                run.step(); run.step()
                self.assertEqual(run.controller_state.statuses, {"read": "done", "write": "deferred"})
                self.assertEqual(run.controller_state.ready(), [])
                self.assertEqual(run.controller_state.goal_statuses(), {"g": "pending"})
                self.assertEqual(len(run.env.calls), 1)
                result = run.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual([c["name"] for c in run.env.calls], [domain + ".inspect", domain + ".change"])
                self.assertEqual(run.controller_state.plan_revisions, 1)
                self.assertEqual(run.controller_state.receipts["obs:1"].version, 1)

    def test_deferred_work_cannot_be_executed_or_finished_even_without_final_review(self):
        run = session(concise([minimal(), minimal("write", "deferred", deps=["read"])]),
                      call("inventory.inspect"), results=[{"ok": True}], review_finish=False)
        run.step(); run.step()
        with self.assertRaisesRegex(planning.ControllerError, "no longer ready"):
            run.controller_state.guard("write", {"name": "inventory.change", "arguments": {}}, 1)
        with self.assertRaisesRegex(planning.ControllerError, "Unresolved steps"):
            adaptive.finish(run, {"message": "Done", "covered_goal_ids": ["g"], "answer_steps": []}, reviewed=False)
        self.assertEqual(run.status, "running")
        self.assertEqual(len(run.env.calls), 1)

    def test_observed_false_condition_skips_concrete_branch_without_writing(self):
        first = [minimal(), minimal("write", "deferred", deps=["read"])]
        later = [minimal(), {**minimal("write", "write", deps=["read"]), "when": condition()}]
        run = session(concise(first), call("inventory.inspect"), decide("replan"),
                      concise(later), finish(2), results=[{"ok": True, "eligible": False}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.controller_state.statuses["write"], "skipped")
        self.assertEqual(len(run.env.calls), 1)

    def test_deferred_step_cannot_hide_concrete_arguments_or_replace_a_prerequisite(self):
        for candidate in (minimal("later", "deferred"),
                          {**minimal("later", "deferred", deps=["read"]), "expected_arguments": {"value": 3}},
                          {**minimal("later", "deferred", deps=["read"]), "tool": "inventory.change"}):
            with self.subTest(candidate=candidate):
                run = session(concise([minimal(), candidate]))
                run.step()
                self.assertEqual(run.controller_state.version, 0)
                self.assertEqual(run.env.calls, [])


if __name__ == "__main__":
    unittest.main()
