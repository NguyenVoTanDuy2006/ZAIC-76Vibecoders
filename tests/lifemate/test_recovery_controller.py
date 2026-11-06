"""Domain-independent fault injection for recovery-v3; no inference or train labels."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import unittest

from lifemate import agent, workflow
from lifemate.pipeline import adaptive, planning, predicates
from lifemate.pipeline.state import ControllerState
from test_workflow import Backend, Env, call


TEXT = "Inspect the requested object and apply the requested change if eligible."


def goal(gid="g", kind="write", description="Requested outcome"):
    return {"id": gid, "description": description, "kind": kind,
            "source": {"ref": "request", "quote": TEXT}}


def step(sid="read", kind="read", domain="inventory", gid="g", deps=None,
         when=None, expected=None, post=None, operation=None):
    return {"id": sid, "goal_id": gid, "kind": kind,
            "tool": domain + (".inspect" if kind == "read" else ".change") if kind in ("read", "write") else None,
            "depends_on": deps or [], "when": when, "expected_arguments": expected or {},
            "postcondition": post, "operation_id": (operation or "op-" + sid) if kind == "write" else None}


def expression(sid="read", path=None, value=True, op="eq"):
    return {"op": op, "left": {"step": sid, "path": path or ["eligible"]}, "right": {"value": value}}


def condition(sid="read", path=None, value=True, op="eq"):
    return {"source": {"ref": "request", "quote": "if eligible"},
            "predicate": expression(sid, path, value, op)}


def plan(steps, goals=None, version=0):
    return json.dumps({"base_version": version, "reason": "Grounded plan or repair",
                       "goals": goals or [goal()], "steps": steps})


def decide(choice, version=1, **fields):
    return json.dumps({"type": choice, "plan_version": version, "reason": "Evidence-based decision", **fields})


def finish(version=1, gids=None, answers=None):
    return decide("finish", version, message="Completed requested outcomes",
                  covered_goal_ids=gids or ["g"], answer_steps=answers or [])


def approve(sid="write", version=1, choice="approve"):
    return json.dumps({"decision": choice, "plan_version": version,
                       "step_id": sid, "reason": "Grounded proposed action; not yet executed"})


def session(*replies, domain="inventory", results=None, env=None, **options):
    tools = [{"name": domain + ".inspect", "kind": "read", "parameters": {"type": "object", "properties": {}}},
             {"name": domain + ".change", "kind": "write", "parameters": {"type": "object", "properties": {
                 "value": {"type": "integer"}, "object_id": {"type": "string"}}}},
             {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object", "properties": {
                 "tool_name": {"type": "string"}, "params": {"type": "array"}}}}]
    request = {"id": "synthetic", "history": [], "context": {"installed_apps": {
        domain: {"tools": [t["name"] for t in tools[:-1]]}}}}
    speech = {"id": "synthetic", "current_text": TEXT, "history_user_texts": []}
    return agent.AgentSession(request, speech, tools, env or Env(results), Backend(*replies),
                              agent.AgentConfig(unified_protocol=False, **options))


class PredicateTests(unittest.TestCase):
    def test_unknown_empty_missing_and_false_are_distinct(self):
        expr = expression(path=["items", 0, "enabled"])
        predicates.validate(expr, {"read"})
        for result in ({}, {"items": []}, {"items": [{}]}):
            self.assertEqual(predicates.evaluate(expr, {"read": result}), "unknown")
        self.assertEqual(predicates.evaluate(expr, {"read": {"items": [{"enabled": False}]}}), "false")

    def test_three_valued_logic_types_and_numeric_thresholds(self):
        unknown = expression("missing")
        false = expression(value=False)
        results = {"read": {"eligible": True}}
        self.assertEqual(predicates.evaluate({"op": "all", "items": [unknown, false]}, results), "false")
        self.assertEqual(predicates.evaluate({"op": "any", "items": [unknown, false]}, results), "unknown")
        self.assertEqual(predicates.evaluate({"op": "not", "item": unknown}, results), "unknown")
        self.assertEqual(predicates.evaluate(expression(value=1), results), "false")
        self.assertEqual(predicates.evaluate(expression(value={"a": 1}), {"read": {"eligible": {"a": True}}}), "false")
        self.assertEqual(predicates.evaluate(expression(value=1.0), {"read": {"eligible": 1}}), "true")
        self.assertEqual(predicates.evaluate(expression(value=True, op="contains"), {"read": {"eligible": [1]}}), "false")
        expr = {"op": "gte", "left": {"step": "balance", "path": ["amount"]},
                "right": {"step": "invoice", "path": ["amount"]}}
        self.assertEqual(predicates.evaluate(expr, {"balance": {"amount": 80}, "invoice": {"amount": 100}}), "false")
        self.assertEqual(predicates.evaluate(expr, {"balance": {"amount": "80"}, "invoice": {"amount": 100}}), "unknown")

    def test_no_python_execution_or_unknown_operators(self):
        for expr in ("__import__('os')", {"op": "exec", "left": {}, "right": {}},
                     expression("absent"), expression(path=[-1])):
            with self.subTest(expr=expr), self.assertRaises(predicates.PredicateError):
                predicates.validate(expr, {"read"})

    def test_large_json_integer_comparison_never_overflows(self):
        expr = expression(path=["amount"], value=10 ** 400, op="gte")
        predicates.validate(expr, {"read"})
        self.assertEqual(predicates.evaluate(expr, {"read": {"amount": 10 ** 401}}), "true")

    def test_literal_tautology_cannot_claim_observed_evidence(self):
        with self.assertRaisesRegex(predicates.PredicateError, "observed step data"):
            predicates.validate({"op": "eq", "left": {"value": 1}, "right": {"value": 1}}, set())


class PlanContractTests(unittest.TestCase):
    def validate(self, run, value, **kw):
        return planning.parse_plan(json.loads(value), sources=run.controller_state.sources,
            tools=run.tools, max_tasks=12, **kw)

    def test_lookup_cannot_depend_on_its_own_unknown_condition(self):
        run = session()
        with self.assertRaisesRegex(planning.ControllerError, "lookup cannot depend"):
            self.validate(run, plan([step(when=condition()), step("write", "write", deps=["read"])]))

    def test_sources_coverage_unknown_tools_and_cycles_are_validated(self):
        run = session()
        bad_goal = goal(); bad_goal["source"]["quote"] = "Invented user instruction"
        cases = [plan([step()], [bad_goal]), plan([step()], [goal("g", "read"), goal("missing", "read")]),
                 plan([step(domain="unknown")], [goal(kind="read")]),
                 plan([step(deps=["write"]), step("write", "write", deps=["read"])])]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(planning.ControllerError):
                self.validate(run, value)

    def test_read_receipt_cannot_cover_a_write_goal_without_a_write_step(self):
        with self.assertRaisesRegex(planning.ControllerError, "matching effect step"):
            self.validate(session(), plan([step()]))

    def test_maintain_requires_a_real_observed_dependency(self):
        constant = {"op": "eq", "left": {"value": 1}, "right": {"value": 1}}
        for expr in (constant, expression("keep")):
            with self.subTest(expr=expr), self.assertRaises(planning.ControllerError):
                self.validate(session(), plan([step("keep", "maintain", post=expr)], [goal(kind="maintain")]))

    def test_mixed_invalid_revision_is_atomic_and_stale_revision_is_rejected(self):
        run = session(plan([step()], [goal(kind="read")]), call("inventory.inspect"),
                      results=[{"ok": True, "eligible": True}])
        run.step(); run.step()
        before = deepcopy(run.controller_state.snapshot())
        bad = json.loads(plan([step(), step("write", "write")], version=1))
        bad["goals"][0]["kind"] = "write"
        bad["steps"][1]["depends_on"] = ["absent"]
        with self.assertRaises(planning.ControllerError):
            planning.parse_plan(bad, sources=run.controller_state.sources, tools=run.tools,
                max_tasks=12, previous=run.controller_state, reinterpret=True)
        self.assertEqual(run.controller_state.snapshot(), before)
        with self.assertRaisesRegex(planning.ControllerError, "base_version=1"):
            self.validate(run, plan([step()], [goal(kind="read")]), previous=run.controller_state)


class RecoveryFlowTests(unittest.TestCase):
    def test_optional_finish_review_flag_skips_model_review_but_keeps_code_completion_checks(self):
        run = session(plan([step()], [goal(kind="read")]), call("inventory.inspect"),
            results=[{"ok": True}], review_finish=False)
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_calls"], 2)
        self.assertFalse(next(e for e in result["trace"] if e["type"] == "finish")["reviewed"])

    def test_control_feedback_is_structured_and_request_sources_are_not_mutable_views(self):
        run = session("bad JSON", plan([step()], [goal(kind="read")]), call("inventory.inspect"), finish(),
            results=[{"ok": True}])
        changed = run.controller_state.sources
        changed["request"] = "overwrite"
        self.assertEqual(run.run()["status"], "finished")
        retry_input = json.loads(run.backend.seen[1][0][-1]["content"])
        self.assertEqual(retry_input["sources"]["request"], TEXT)
        self.assertEqual(retry_input["controller"]["issue"]["code"], "control_json")
        self.assertEqual(retry_input["controller"]["previous_bad_reply"], "bad JSON")

    def test_read_condition_write_flow_uses_same_controller_across_domains_without_full_reviews(self):
        for domain in ("inventory", "documents", "transport", "subscriptions"):
            with self.subTest(domain=domain):
                steps = [step(domain=domain), step("write", "write", domain, deps=["read"],
                    when=condition(), expected={"value": 7})]
                run = session(plan(steps), call(domain + ".inspect"), call(domain + ".change", value=7),
                    approve(), finish(), domain=domain, results=[{"ok": True, "eligible": True}, {"ok": True, "value": 7}])
                result = run.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual(result["model_calls"], 5)
                self.assertEqual(result["submission"]["tool_calls"], run.env.get_tool_calls())
                self.assertEqual(result["submission"]["state_hash"], run.env.get_state_hash())
                self.assertEqual(result["diagnostics"]["workflow_protocol"], "recovery-json-v3")
                self.assertEqual([r["step_id"] for r in result["diagnostics"]["controller"]["observations"]], ["read", "write"])
                self.assertFalse(any(e.get("phase") == "review" for e in result["trace"]))

    def test_false_condition_skips_branch_but_completes_the_unconditional_lookup(self):
        steps = [step(), step("write", "write", deps=["read"], when=condition())]
        run = session(plan(steps), call("inventory.inspect"), finish(), results=[{"ok": True, "eligible": False}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.controller_state.statuses, {"read": "done", "write": "skipped"})
        self.assertEqual(len(run.env.calls), 1)

    def test_unknown_predicate_never_authorizes_write_or_skip_and_can_be_replanned(self):
        bad = [step(), step("write", "write", deps=["read"], when=condition(path=["missing"]))]
        fixed = [step(), step("write", "write", deps=["read"], when=condition())]
        run = session(plan(bad), call("inventory.inspect"), decide("replan"), plan(fixed, version=1),
            call("inventory.change", value=1), approve(version=2), finish(2), results=[{"ok": True, "eligible": True}, {"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(run.controller_state.version, 2)
        self.assertEqual(len(run.env.calls), 2)
        self.assertEqual(run.controller_state.receipts["obs:1"].version, 1)

    def test_interpretation_can_be_corrected_without_erasing_initial_request_or_execution(self):
        original = goal(kind="read", description="Mistaken read-only interpretation")
        corrected = goal(kind="write", description="Corrected requested change")
        run = session(plan([step()], [original]), call("inventory.inspect"), decide("reinterpret"),
            plan([step(), step("write", "write", deps=["read"])], [corrected], version=1),
            call("inventory.change", value=2), approve(version=2), finish(2),
            results=[{"ok": True}, {"ok": True}])
        frozen_request = run.controller_state.request_json
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.controller_state.interpretation_version, 2)
        self.assertEqual(run.controller_state.request_json, frozen_request)
        self.assertEqual(run.controller_state.revisions[0]["goals"]["g"], original)
        self.assertEqual(run.controller_state.revisions[1]["goals"]["g"], corrected)

    def test_deviating_actor_cannot_execute_another_tool_before_its_dependency(self):
        steps = [step(), step("write", "write", deps=["read"])]
        run = session(plan(steps), call("inventory.change", value=2), decide("retry", step_id="read"),
            call("inventory.inspect"), call("inventory.change", value=2), approve(), finish(),
            results=[{"ok": True}, {"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual([c["name"] for c in run.env.calls], ["inventory.inspect", "inventory.change"])
        self.assertTrue(any(e.get("issue", {}).get("code") == "wrong_step_tool" for e in run.trace))

    def test_small_approval_contract_does_not_require_a_result_before_execution(self):
        run = session(plan([step("write", "write")]), call("inventory.change", value=1),
            '{"tasks":[],"complete":true}', approve(), finish(), results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(run.env.calls), 1)
        self.assertTrue(any(e.get("issue", {}).get("code") == "approval_contract" for e in run.trace))
        prompt = run.backend.seen[2][0][0]["content"]
        self.assertIn("Execution evidence is NOT", prompt)
        self.assertNotIn("tasks", json.loads(run.backend.seen[2][0][-1]["content"])["payload"])

    def test_malformed_control_is_bounded_without_any_execution(self):
        run = session("<tool_call>{}</tool_call>", "<tool_call>{}</tool_call>")
        result = run.run()
        self.assertEqual(result["status"], "control_error")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(run.env.calls, [])

    def test_finished_claim_cannot_drop_unexecuted_goals(self):
        run = session(plan([step("write", "write")]), "Already done", finish(),
            decide("retry", step_id="write"), call("inventory.change", value=4), approve(), finish(),
            results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(run.env.calls), 1)
        self.assertTrue(any(e.get("issue", {}).get("code") == "incomplete_goals" for e in run.trace))

    def test_maintaining_state_needs_evidence_but_needs_no_write(self):
        read = step()
        keep = step("keep", "maintain", deps=["read"], when=condition(value=False))
        run = session(plan([read, keep], [goal(kind="maintain")]), call("inventory.inspect"), finish(),
            results=[{"ok": True, "eligible": False}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.controller_state.statuses["keep"], "done")
        self.assertEqual(len(run.env.calls), 1)

    def test_missing_user_information_goes_through_real_executor_and_waits(self):
        run = session(plan([step("write", "write")]), '<controller_request>{"type":"replan","reason":"Need input"}</controller_request>',
            plan([step("write", "write")], version=1), results=[])
        # Force a recovery point with the same version, without creating any env call.
        run.step()
        run.controller_state.phase = "recover"
        run.backend = Backend(decide("ask", tool_name="inventory.change", params=["object_id"]))
        run.env = Env([{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "waiting_for_user")
        self.assertEqual(result["submission"]["tool_calls"][0]["name"], "sys.ask_missing_info")

    def test_unsupported_goal_cannot_be_finished_as_a_successful_answer(self):
        run = session(plan([step("explain", "answer")], [goal(kind="unsupported")]),
            finish(answers=["explain"]), decide("stop"), decide("stop"))
        result = run.run()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(run.env.calls, [])

    def test_false_condition_can_skip_an_unavailable_action_without_claiming_it_executed(self):
        steps = [step(), step("unsupported", "answer", deps=["read"], when=condition())]
        run = session(plan(steps, [goal(kind="unsupported")]), call("inventory.inspect"), finish(),
            results=[{"ok": True, "eligible": False}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.controller_state.statuses["unsupported"], "skipped")

    def test_no_op_budget_survives_irrelevant_success_and_changing_failure_text(self):
        goals = [goal("g", "write"), goal("other", "read")]
        steps = [step("write", "write"), step("other_read", gid="other")]
        run = session(plan(steps, goals), call("inventory.change", value=1), approve(),
            decide("retry", step_id="other_read"), call("inventory.inspect"),
            call("inventory.change", value=2), approve(),
            results=[{"ok": True, "done": False, "message": "First failure"},
                     {"ok": True, "objects": [1]}, {"ok": True, "done": False, "message": "Different failure"}],
            max_no_progress=2)
        result = run.run()
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(run.controller_state.stalls["g"], 2)
        self.assertEqual(len(run.env.calls), 3)

    def test_total_generation_budget_applies_across_reinterpretation(self):
        run = session(plan([step()], [goal(kind="read")]), call("inventory.inspect"), decide("reinterpret"),
            max_steps=3, results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(result["model_calls"], 3)

    def test_dependent_answers_can_be_recorded_incrementally_without_premature_finish(self):
        steps = [step("first", "answer"), step("second", "answer", deps=["first"])]
        run = session(plan(steps, [goal(kind="answer")]),
            decide("answer", step_id="first", message="First grounded answer", evidence=["request"]),
            finish(answers=["second"]))
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(run.controller_state.statuses, {"first": "done", "second": "done"})

    def test_replan_budget_exhausts_even_when_new_ids_make_plans_look_different(self):
        first, second = [step()], [step("another")]
        run = session(plan(first, [goal(kind="read")]),
            '<controller_request>{"type":"replan","reason":"Change route"}</controller_request>',
            plan(second, [goal(kind="read")], version=1),
            '<controller_request>{"type":"replan","reason":"Change route again"}</controller_request>',
            max_plan_revisions=1)
        self.assertEqual(run.run()["status"], "recovery_exhausted")
        self.assertEqual(run.env.calls, [])


class ExecutionIdentityTests(unittest.TestCase):
    def test_unlogged_write_cannot_be_reconciled_or_reported_as_finished(self):
        class MissingLog(Env):
            def call(self, name, arguments):
                return {"ok": True}
        run = session(plan([step("write", "write")]), call("inventory.change", value=1), approve(), env=MissingLog())
        result = run.run()
        self.assertEqual(result["status"], "execution_error")
        self.assertEqual(result["submission"]["tool_calls"], [])
        self.assertEqual(run.controller_state.statuses["write"], "uncertain")

    def test_prewrite_observation_cannot_reconcile_a_later_uncertain_effect(self):
        read = step("verify")
        write = step("write", "write", deps=["verify"], post=expression("write", ["value"], 3))
        run = session(plan([read, write]), call("inventory.inspect"), call("inventory.change", value=3), approve(),
            results=[{"ok": True, "value": 3}, {"ok": True, "value": 4}])
        run.step(); run.step(); run.step(); run.step()
        self.assertEqual(run.controller_state.statuses["write"], "uncertain")
        with self.assertRaisesRegex(planning.ControllerError, "after the uncertain"):
            adaptive.reconcile(run, "write", "verify")

    def test_uncertainty_blocks_new_operation_ids_even_with_different_arguments(self):
        write = step("write", "write", post=expression("write", ["value"], 3))
        run = session(plan([write]), call("inventory.change", value=3), approve(), results=[{"ok": True, "value": 4}])
        run.step(); run.step(); run.step()
        state = run.controller_state
        goals, steps = planning.parse_plan(json.loads(plan([write, step("new", "write")], version=1)),
            sources=state.sources, tools=run.tools, max_tasks=12, previous=state)
        state.install(goals, steps, "Propose another operation")
        with self.assertRaisesRegex(planning.ControllerError, "outstanding write uncertainty"):
            state.guard("new", {"name": "inventory.change", "arguments": {"value": 4}}, 2)

    def test_reconcile_does_not_use_an_unused_or_negated_equality_as_argument_binding(self):
        expr = {"op": "any", "items": [expression("write", ["value"], 3), expression("write", ["ready"], True)]}
        write, verify = step("write", "write", post=expr), step("verify")
        run = session(plan([write]), call("inventory.change", value=3), approve(), results=[{"ok": True, "value": 4, "ready": False}])
        run.step(); run.step(); run.step()
        state = run.controller_state
        goals, steps = planning.parse_plan(json.loads(plan([write, verify], version=1)),
            sources=state.sources, tools=run.tools, max_tasks=12, previous=state)
        state.install(goals, steps, "Verify result")
        state.record("verify", {"name": "inventory.inspect", "arguments": {}},
                     {"ok": True, "value": 4, "ready": True}, "obs:2", 2)
        with self.assertRaisesRegex(planning.ControllerError, "positive observed equality"):
            adaptive.reconcile(run, "write", "verify")

    def test_repeated_prelog_read_exceptions_get_unique_receipts_and_stop_at_budget(self):
        class ReadFailure(Env):
            def call(self, *args):
                raise RuntimeError("Failure before appending the canonical log")
        run = session(plan([step()], [goal(kind="read")]), call("inventory.inspect"),
            decide("retry", step_id="read"), call("inventory.inspect"), env=ReadFailure(), max_no_progress=2)
        self.assertEqual(run.run()["status"], "no_progress")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(set(run.controller_state.receipts), {"obs:1", "obs:2"})

    def test_nonobject_write_result_is_uncertain_and_cannot_finish(self):
        for bad_result in (None, {"ok": True, "value": float("nan")}):
            with self.subTest(result=bad_result):
                run = session(plan([step("write", "write")]), call("inventory.change", value=1), approve(),
                    finish(), decide("stop"), decide("stop"), results=[bad_result])
                result = run.run()
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(run.controller_state.statuses["write"], "uncertain")
                self.assertEqual(len(run.env.calls), 1)

    def test_replan_cannot_rename_completed_step_or_replay_write_under_new_operation(self):
        completed = step("write", "write")
        run = session(plan([completed]), call("inventory.change", value=7), approve(), results=[{"ok": True}])
        run.step(); run.step(); run.step()
        state = run.controller_state
        renamed = step("renamed", "write", operation="op-write")
        with self.assertRaisesRegex(planning.ControllerError, "cannot be removed"):
            planning.parse_plan(json.loads(plan([renamed], version=1)), sources=state.sources,
                tools=run.tools, max_tasks=12, previous=state)
        proposed = [completed, step("replay", "write")]
        goals, steps = planning.parse_plan(json.loads(plan(proposed, version=1)), sources=state.sources,
            tools=run.tools, max_tasks=12, previous=state)
        state.install(goals, steps, "Append a pending step")
        with self.assertRaisesRegex(planning.ControllerError, "cannot replay"):
            state.guard("replay", {"name": "inventory.change", "arguments": {"value": 7}}, 2)
        self.assertEqual(len(run.env.calls), 1)
        receipt = next(iter(state.receipts.values()))
        with self.assertRaises(FrozenInstanceError):
            receipt.outcome = "error"
        copy = receipt.snapshot(); copy["result"]["ok"] = False
        self.assertTrue(receipt.snapshot()["result"]["ok"])

    def test_two_intentionally_declared_distinct_operations_can_have_identical_arguments(self):
        steps = [step("first", "write"), step("second", "write", deps=["first"])]
        run = session(plan(steps), call("inventory.change", value=1), approve("first"),
            call("inventory.change", value=1), approve("second"), finish(), results=[{"ok": True}, {"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(len(run.env.calls), 2)

    def test_pending_proposal_is_invalidated_by_a_new_plan_version_before_execution(self):
        run = session(plan([step("write", "write")]), call("inventory.change", value=1), approve())
        run.step(); run.propose(); run.propose()
        self.assertIsNotNone(run.pending)
        state = run.controller_state
        state.install(state.goals, state.steps, "Install a newer revision")
        event = run.execute_pending()
        self.assertEqual(event["issue"]["code"], "stale_execution")
        self.assertIsNone(run.pending)
        self.assertEqual(run.env.calls, [])

    def test_uncertain_write_is_verified_by_read_before_completion_without_reexecution(self):
        class MutatingEnv(Env):
            def call(self, name, arguments):
                self.calls.append({"name": name, "arguments": deepcopy(arguments)})
                if name.endswith(".change"):
                    self.value = arguments["value"]
                    raise RuntimeError("Response lost after mutation")
                return {"ok": True, "value": self.value}
        original = step("write", "write", post=expression("write", ["value"], 3))
        verification = step("verify")
        run = session(plan([original]), call("inventory.change", value=3), approve(),
            decide("replan"), plan([original, verification], version=1), call("inventory.inspect"),
            decide("reconcile", 2, step_id="write", verification_step="verify"), finish(2), env=MutatingEnv())
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual([c["name"] for c in run.env.calls], ["inventory.change", "inventory.inspect"])
        self.assertEqual(next(iter(run.controller_state.receipts.values())).outcome, "uncertain")
        self.assertTrue(any(e["type"] == "execution_reconciled" for e in run.trace))


class OfficialRecoveryIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Set LIFEMATE_DATA_ROOT for optional official-runtime check")
    def test_recovery_controller_with_official_toolenv_and_scripted_model(self):
        from datetime import datetime, timedelta
        root = Path(os.environ["LIFEMATE_DATA_ROOT"])
        row = agent.load_split_items(root, "train")[0]
        request = agent.prepare_request(row)
        tools = agent.select_available_tools(request, agent.load_catalog(root, "train"))
        if "weather.get_forecast" not in {t["name"] for t in tools}:
            self.skipTest("First sample does not allow the read-only weather tool")
        speech = {"id": row["id"], "current_text": TEXT,
                  "history_user_texts": [{"history_index": i, "text": "Synthetic history"}
                                         for i, turn in enumerate(request["history"]) if turn["role"] == "user"]}
        read = step(); read["tool"] = "weather.get_forecast"
        tomorrow = (datetime.fromisoformat(row["context"]["current_time"]) + timedelta(days=1)).date().isoformat()
        backend = Backend(plan([read], [goal(kind="read")]), call("weather.get_forecast", date=tomorrow), finish())
        env = agent.load_toolenv(root, "train")(row["id"])
        result = agent.run_agent(request, speech, tools, env, backend=backend,
                                 config=agent.AgentConfig(unified_protocol=False))
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["submission"]["tool_calls"], env.get_tool_calls())
        self.assertEqual(result["submission"]["state_hash"], env.get_state_hash())


if __name__ == "__main__":
    unittest.main()
