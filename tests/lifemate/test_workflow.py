"""Controller behavior across invented domains; scripted replies, not model accuracy."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

from lifemate import agent, workflow
from lifemate import train_eval
from unittest.mock import patch
from test_predictions import fixture


def task(tid="read", kind="read", depends=None, condition=""):
    return {"id": tid, "goal": "User requested outcome " + tid, "kind": kind,
            "depends_on": depends or [], "condition": condition}


def plan(tasks, names):
    return json.dumps({"tasks": tasks, "tool_names": names})


def update(tid="read", status="pending", refs=None, condition="unknown", condition_refs=None):
    return {"id": tid, "status": status, "evidence": refs or [], "reason": "Observed outcome",
            "condition_status": condition, "condition_evidence": condition_refs or []}


def review(updates, complete=False, **extra):
    return json.dumps({"tasks": updates, "complete": complete, "next_tools": [],
                       "message": "Assessment summary", **extra})


def call(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


class Backend:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.seen = []

    def generate(self, messages, tools, config, remaining_seconds=None):
        self.seen.append((deepcopy(messages), deepcopy(tools), config))
        value = next(self.replies)
        return value if isinstance(value, agent.ModelReply) else agent.ModelReply(value)


class Env:
    def __init__(self, results=None):
        self.calls = []
        self.results = iter(results or [])

    def validate(self, name, arguments):
        return {"ok": True}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": deepcopy(arguments)})
        return deepcopy(next(self.results))

    def get_tool_calls(self):
        return deepcopy(self.calls)

    def get_state_hash(self):
        return "scripted-state-hash"


def session(backend, domain="warehouse", results=None, config=None):
    # These regression scripts exercise the preserved v2 full-ledger protocol.
    # The default BTC protocol and recovery-v3 have their own suites.
    config = replace(config or agent.AgentConfig(), unified_protocol=False, adaptive_controller=False)
    tools = [
        {"name": f"{domain}.inspect", "kind": "read", "description": "Inspect requested objects",
         "parameters": {"type": "object", "properties": {}}},
        {"name": f"{domain}.change", "kind": "write", "description": "Apply requested change",
         "parameters": {"type": "object", "properties": {"object_id": {
             "type": "string", "x-ref": f"{domain}.objects", "description": f"ID from {domain}.inspect"}}}},
        {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object"}},
    ]
    request = {"id": "invented", "history": [], "context": {
        "current_time": "2030-06-01T08:00:00+07:00", "installed_apps": {
            domain: {"tools": [t["name"] for t in tools[:-1]]}}}}
    speech = {"id": "invented", "current_text": "Inspect then change the requested object if eligible",
              "history_user_texts": []}
    return agent.AgentSession(request, speech, tools, Env(results), backend, config)


class PlanTests(unittest.TestCase):
    def test_duplicate_missing_and_cyclic_dependencies_are_rejected(self):
        for tasks in ([task(), task()], [task(depends=["absent"])],
                      [task("a", depends=["b"]), task("b", depends=["a"])], []):
            with self.subTest(tasks=tasks), self.assertRaises(workflow.WorkflowError):
                workflow.task_plan({"tasks": tasks})

    def test_control_json_rejects_duplicate_nonfinite_and_native_calls(self):
        for text in ('{"tasks":[],"tasks":[]}', '{"x":NaN}', '[]', call("invented.inspect")):
            with self.subTest(text=text), self.assertRaises(workflow.WorkflowError):
                workflow.parse_control(text)

    def test_unknown_tool_in_plan_never_executes_and_has_bounded_repair(self):
        invalid = plan([task()], ["absent.inspect"])
        run = session(Backend(invalid, invalid))
        result = run.run()
        self.assertEqual(result["status"], "control_error")
        self.assertEqual(result["submission"]["tool_calls"], [])
        self.assertEqual(result["model_calls"], 2)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tasks = workflow.task_plan({"tasks": [task("write", "write")]})
        self.evidence = {"request": {"source": "request"},
                         "obs:1": {"source": "tool", "kind": "write", "outcome": "no_op"}}

    def test_no_op_wrong_kind_and_invented_refs_cannot_prove_write(self):
        for ref, entry in [("obs:1", self.evidence["obs:1"]),
                           ("obs:1", {"source": "tool", "kind": "read", "outcome": "success"}),
                           ("invented", None), ("request", self.evidence["request"])]:
            evidence = deepcopy(self.evidence)
            if entry is not None:
                evidence[ref] = entry
            value = json.loads(review([update("write", "done", [ref])], True))
            with self.subTest(ref=ref, entry=entry), self.assertRaises(workflow.WorkflowError):
                workflow.reviewed_tasks(value, self.tasks, evidence)
        self.assertEqual(self.tasks["write"]["status"], "pending")

    def test_skip_and_dependency_checks_require_proven_conditions(self):
        tasks = workflow.task_plan({"tasks": [task("a"), task("b", "write", ["a"], "eligible")]})
        evidence = {"obs:1": {"source": "tool", "kind": "write", "outcome": "success"}}
        bad = json.loads(review([update("a"), update("b", "done", ["obs:1"], "true", ["obs:1"])], True))
        with self.assertRaises(workflow.WorkflowError):
            workflow.reviewed_tasks(bad, tasks, evidence)
        bad = json.loads(review([update("a"), update("b", "skipped", condition="false")]))
        with self.assertRaises(workflow.WorkflowError):
            workflow.reviewed_tasks(bad, tasks, evidence)

    def test_review_can_add_a_missed_task_without_erasing_existing_requirements(self):
        value = json.loads(review([update("write"), update("extra")], new_tasks=[task("extra")]))
        result = workflow.reviewed_tasks(value, self.tasks, self.evidence)
        self.assertEqual(set(result), {"write", "extra"})
        with self.assertRaises(workflow.WorkflowError):
            workflow.reviewed_tasks(json.loads(review([update("extra")], new_tasks=[task("extra")])),
                                    self.tasks, self.evidence)

    def test_successful_identical_write_cannot_be_reexecuted_for_same_task(self):
        candidate = {"name": "invented.change", "arguments": {"count": 2}}
        evidence = {"obs:1": {"task_id": "write", "outcome": "success", "call": candidate}}
        with self.assertRaisesRegex(workflow.WorkflowError, "already executed"):
            workflow.approve_write({"approve": True, "task_id": "write"}, self.tasks, evidence, candidate)

    def test_malformed_review_fields_are_control_errors_not_runner_crashes(self):
        valid = json.loads(review([update("write")]))
        for value in ({**valid, "new_tasks": "bad"}, {**valid, "tasks": [None]},
                      {**valid, "tasks": [{**update("write"), "evidence": {}}]},
                      {**valid, "complete": "true"}):
            with self.subTest(value=value), self.assertRaises(workflow.WorkflowError):
                workflow.reviewed_tasks(value, self.tasks, self.evidence)

    def test_missing_unknown_condition_metadata_is_conservatively_defaulted(self):
        tasks = workflow.task_plan({"tasks": [task()]})
        value = json.loads(review([{"id": "read", "status": "pending", "evidence": [], "reason": "Need data"}]))
        result = workflow.reviewed_tasks(value, tasks, self.evidence)
        self.assertEqual(result["read"]["condition_status"], "unknown")
        self.assertEqual(result["read"]["condition_evidence"], [])
        self.assertEqual(result["read"]["status"], "pending")
        self.assertNotIn("condition_status", value["tasks"][0])

    def test_missing_condition_evidence_never_proves_truth_or_skips_a_branch(self):
        tasks = workflow.task_plan({"tasks": [task("write", "write", condition="eligible")]})
        for condition in ("true", "false"):
            value = json.loads(review([update("write", condition=condition)]))
            del value["tasks"][0]["condition_evidence"]
            with self.subTest(condition=condition), self.assertRaises(workflow.WorkflowError):
                workflow.reviewed_tasks(value, tasks, self.evidence)

    def test_invalid_evidence_error_identifies_task_field_and_ref(self):
        value = json.loads(review([update("write", refs=["obs_1"])]))
        with self.assertRaisesRegex(workflow.WorkflowError, "Task write: unknown evidence ref 'obs_1'"):
            workflow.reviewed_tasks(value, self.tasks, self.evidence)


class ObservationDiagnosticTests(unittest.TestCase):
    def test_blank_optional_read_filters_are_flagged_without_mutation(self):
        value = {"name": "inventory.search", "arguments": {"region": "", "query": "", "limit": 0}}
        before = deepcopy(value)
        schema = {"type": "object", "required": ["query"], "properties": {
            "region": {"type": "string"}, "query": {"type": "string"}, "limit": {"type": "integer"}}}
        diagnostics = workflow.observation_diagnostics(value, schema, {"ok": True, "items": []}, "read")
        self.assertEqual(diagnostics["optional_blank_read_arguments"], ["arguments.region"])
        self.assertEqual(diagnostics["empty_collection_fields"], ["items"])
        self.assertEqual(value, before)

    def test_write_empty_values_are_not_treated_as_bad_filters(self):
        value = {"name": "documents.update", "arguments": {"content": ""}}
        schema = {"properties": {"content": {"type": "string"}}}
        self.assertEqual(workflow.observation_diagnostics(value, schema, {"ok": True}, "write"), {
            "optional_blank_read_arguments": [], "empty_collection_fields": []})


class GeneralFlowTests(unittest.TestCase):
    def test_same_controller_handles_read_lookup_write_in_unrelated_domains(self):
        for domain in ("warehouse", "documents", "subscriptions", "transport"):
            with self.subTest(domain=domain):
                inspect, change = f"{domain}.inspect", f"{domain}.change"
                tasks = [task(), task("write", "write", ["read"], "eligible")]
                ready = [update("read", "done", ["obs:1"]),
                         update("write", condition="true", condition_refs=["obs:1"])]
                done = [ready[0], update("write", "done", ["obs:2"], "true", ["obs:1"])]
                backend = Backend(plan(tasks, [change]), call(inspect), review(ready),
                                  call(change, object_id="returned-object"),
                                  review(ready, approve=True, task_id="write"),
                                  review(done, True), "Completed", review(done, True))
                run = session(backend, domain, [{"ok": True, "objects": [{"id": "returned-object"}],
                                                 "eligible": True}, {"ok": True, "done": True}])
                result = run.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual([c["name"] for c in run.env.get_tool_calls()], [inspect, change])
                self.assertEqual(result["submission"]["tool_calls"], run.env.get_tool_calls())
                self.assertEqual(result["submission"]["state_hash"], run.env.get_state_hash())
                self.assertEqual(result["model_calls"], 8)
                self.assertIn(inspect, backend.seen[1][1][0]["function"]["name"])

    def test_early_final_is_reviewed_then_work_continues(self):
        backend = Backend(plan([task()], ["warehouse.inspect"]), "I will inspect it",
                          review([update()], False), call("warehouse.inspect"),
                          review([update(status="done", refs=["obs:1"])], True), "Done",
                          review([update(status="done", refs=["obs:1"])], True))
        result = session(backend, results=[{"ok": True, "objects": []}]).run()
        self.assertEqual(result["status"], "finished")
        self.assertTrue(any(e["type"] == "finish_rejected" for e in result["trace"]))
        self.assertEqual(len(result["submission"]["tool_calls"]), 1)

    def test_false_condition_skips_write_without_any_mutation(self):
        tasks = [task(), task("write", "write", ["read"], "eligible")]
        updates = [update(status="done", refs=["obs:1"]),
                   update("write", "skipped", ["obs:1"], "false", ["obs:1"])]
        backend = Backend(plan(tasks, ["warehouse.inspect", "warehouse.change"]), call("warehouse.inspect"),
                          review(updates, True), "No change needed", review(updates, True))
        result = session(backend, results=[{"ok": True, "eligible": False}]).run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual([c["name"] for c in result["submission"]["tool_calls"]], ["warehouse.inspect"])

    def test_write_before_dependencies_or_condition_is_rejected_without_execution(self):
        tasks = [task(), task("write", "write", ["read"], "eligible")]
        bad = review([update(), update("write")], approve=True, task_id="write")
        backend = Backend(plan(tasks, ["warehouse.change"]), call("warehouse.change"), bad, bad)
        result = session(backend).run()
        self.assertEqual(result["status"], "control_error")
        self.assertEqual(result["submission"]["tool_calls"], [])

    def test_runtime_no_op_cannot_be_promoted_to_success(self):
        tasks = [task("write", "write")]
        pending = [update("write")]
        false_done = review([update("write", "done", ["obs:1"])], True)
        backend = Backend(plan(tasks, ["warehouse.change"]), call("warehouse.change"),
                          review(pending, approve=True, task_id="write"), false_done, false_done)
        result = session(backend, results=[{"ok": True, "done": False, "message": "Unavailable"}]).run()
        self.assertEqual(result["status"], "control_error")
        self.assertEqual(result["diagnostics"]["workflow"]["tasks"][0]["status"], "pending")
        self.assertEqual(len(result["submission"]["tool_calls"]), 1)

    def test_runtime_failure_can_recover_with_a_different_valid_action(self):
        tasks = [task("write", "write")]
        pending = [update("write")]
        done = [update("write", "done", ["obs:2"])]
        backend = Backend(plan(tasks, ["warehouse.change"]), call("warehouse.change", mode="first"),
                          review(pending, approve=True, task_id="write"), review(pending),
                          call("warehouse.change", mode="alternative"),
                          review(pending, approve=True, task_id="write"), review(done, True),
                          "Done", review(done, True))
        result = session(backend, results=[{"ok": True, "done": False}, {"ok": True}]).run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(result["submission"]["tool_calls"]), 2)
        self.assertEqual(result["diagnostics"]["workflow"]["tasks"][0]["evidence"], ["obs:2"])

    def test_repeated_identical_no_op_is_bounded(self):
        tasks = [task("write", "write")]
        pending = [update("write")]
        replies = [plan(tasks, ["warehouse.change"])]
        for _ in range(3):
            replies += [call("warehouse.change"), review(pending, approve=True, task_id="write"), review(pending)]
        result = session(Backend(*replies), results=[{"ok": True, "done": False}] * 3).run()
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(len(result["submission"]["tool_calls"]), 3)

    def test_missing_info_waits_without_claiming_completion(self):
        backend = Backend(plan([task("write", "write")], []),
                          call("sys.ask_missing_info", tool_name="warehouse.change", params=["object_id"]))
        result = session(backend, results=[{"ok": True}]).run()
        self.assertEqual(result["status"], "waiting_for_user")

    def test_answer_only_request_can_finish_without_tools(self):
        backend = Backend(plan([task("answer", "answer")], []), "An explanation",
                          review([update("answer", "done", ["request"])], True))
        result = session(backend).run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["submission"]["tool_calls"], [])

    def test_routing_expands_schemas_without_executing_or_logging_a_tool(self):
        backend = Backend(plan([task()], []), '<tool_request>{"all":true}</tool_request>',
                          call("warehouse.inspect"), review([update(status="done", refs=["obs:1"])], True),
                          "Done", review([update(status="done", refs=["obs:1"])], True))
        run = session(backend, results=[{"ok": True, "objects": []}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(run.env.calls), 1)
        self.assertEqual(len(backend.seen[1][1]), 1)
        self.assertEqual(len(backend.seen[2][1]), 3)

    def test_call_to_allowed_but_unopened_schema_discovers_without_penalizing_or_executing(self):
        backend = Backend(plan([task()], []), call("warehouse.inspect"), call("warehouse.inspect"))
        run = session(backend, results=[{"ok": True, "objects": []}])
        run.step()  # Plan initially opens only the missing-information tool.
        event = run.step()
        self.assertEqual(event["type"], "tool_route")
        self.assertEqual(run.invalid_replies, 0)
        self.assertEqual(run.env.calls, [])
        self.assertFalse(json.loads(run.messages[-1]["content"])["executed"])
        run.step()
        self.assertEqual(len(run.env.calls), 1)

    def test_blocked_final_rechecks_all_tools_before_stopping(self):
        blocked = review([update(status="blocked", refs=["request"])], True)
        backend = Backend(plan([task()], []), "Unsupported", blocked, blocked)
        result = session(backend).run()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["model_calls"], 4)
        self.assertTrue(any(e["type"] == "finish_rejected" for e in result["trace"]))

    def test_truncated_control_and_all_phases_share_generation_budget(self):
        backend = Backend(agent.ModelReply("{}", truncated=True), plan([task()], []), "Done")
        result = session(backend, config=agent.AgentConfig(max_steps=3)).run()
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(result["submission"]["tool_calls"], [])
        self.assertEqual(result["model_calls"], 3)

    def test_planner_and_reviewer_receive_no_gold_and_use_same_backend(self):
        backend = Backend(plan([task("answer", "answer")], []), "Explanation",
                          review([update("answer", "done", ["request"])], True))
        run = session(backend)
        run.run()
        self.assertEqual([e["phase"] for e in run.trace if e["type"] in ("model_reply", "control_reply")],
                         ["plan", "act", "review"])
        for messages, tools, _ in (backend.seen[0], backend.seen[-1]):
            self.assertEqual(tools, [])
            self.assertIn("CONTROL PHASE", messages[0]["content"])
        self.assertEqual(backend.seen[0][2].max_new_tokens, run.config.control_max_new_tokens)

    def test_control_prompt_excludes_actor_protocol_and_retry_includes_rejected_output(self):
        bad = "<final>Cannot decide yet</final>"
        done = [update("answer", "done", ["request"])]
        backend = Backend(plan([task("answer", "answer")], []), "Explanation", bad, review(done, True))
        run = session(backend)
        result = run.run()
        self.assertEqual(result["status"], "finished")
        for index in (0, 2, 3):
            messages = backend.seen[index][0]
            system = messages[0]["content"]
            self.assertIn(workflow.CONTROL_SYSTEM_PROMPT, system)
            self.assertNotIn("<tool_call>", system)
            self.assertNotIn("<tool_request>", system)
            self.assertNotIn("<final>", system)
            self.assertEqual(json.loads(messages[-1]["content"])["request_text"], run.speech["current_text"])
        feedback = json.loads(backend.seen[3][0][-1]["content"])
        self.assertEqual(feedback["previous_control_reply"], bad)
        self.assertIn("Invalid control JSON", feedback["previous_control_error"])
        self.assertIsNone(run.control_bad_reply)

    def test_observed_format_failures_recover_and_empty_lookup_can_retry_without_blank_filter(self):
        pending = update()
        del pending["condition_evidence"]  # Same missing field as the reported Kaggle reply.
        done = [update(status="done", refs=["obs:2"])]
        backend = Backend(plan([task()], ["warehouse.inspect"]), call("warehouse.inspect", region=""),
                          "<final>No items means nothing is available</final>", review([pending]),
                          call("warehouse.inspect"), review(done, True), "Found current data", review(done, True))
        run = session(backend, results=[{"ok": True, "items": []}, {"ok": True, "items": [{"available": False}]}])
        run.tools["warehouse.inspect"]["parameters"]["properties"]["region"] = {"type": "string"}
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len([e for e in result["trace"] if e["type"] == "control_rejected"]), 1)
        self.assertEqual([c["arguments"] for c in run.env.calls], [{"region": ""}, {}])
        self.assertEqual(result["diagnostics"]["workflow"]["observations"][0]["diagnostics"], {
            "optional_blank_read_arguments": ["arguments.region"], "empty_collection_fields": ["items"]})
        actor_state = backend.seen[4][0][0]["content"]
        self.assertIn('"optional_blank_read_arguments": ["arguments.region"]', actor_state)
        self.assertIn(workflow.ACT_ARGUMENT_GUIDANCE, actor_state)

    def test_controller_gets_only_snapshot_messages_and_full_candidate_schema(self):
        pending = [update("write")]
        backend = Backend(plan([task("write", "write")], ["warehouse.change"]),
                          call("warehouse.change"), review(pending, approve=True, task_id="write"),
                          review([update("write", "done", ["obs:1"])], True), "Done",
                          review([update("write", "done", ["obs:1"])], True))
        run = session(backend, results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["diagnostics"]["workflow_protocol"], workflow.LEGACY_PROTOCOL_VERSION)
        for index in (0, 2, 3, 5):
            messages, tools, _ = backend.seen[index]
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertEqual(tools, [])
            self.assertTrue(all("tool_calls" not in m for m in messages))
        preflight = json.loads(backend.seen[2][0][-1]["content"])
        self.assertEqual(preflight["candidate_tool_schema"], run.native_tools[1]["function"])
        self.assertEqual(preflight["observations"], [])
        after_execution = json.loads(backend.seen[3][0][-1]["content"])
        self.assertEqual(after_execution["observations"][0]["ref"], "obs:1")
        self.assertEqual(after_execution["observations"][0]["result"], {"ok": True})

    def test_write_review_example_requires_approval_while_other_phases_do_not(self):
        preflight = workflow.review_instruction({"candidate_call": {}})
        for field in ('"approve": false', '"task_id":'):
            self.assertIn(field, preflight)
            self.assertNotIn(field, workflow.review_instruction({"observation_ref": "obs:1"}))
            self.assertNotIn(field, workflow.review_instruction({"candidate_final": "Done"}))
        self.assertIn("PHA WRITE_REVIEW", preflight)

    def test_wrong_phase_call_is_never_executed_even_if_it_is_a_valid_action(self):
        proposed = call("warehouse.change")
        backend = Backend(plan([task("write", "write")], ["warehouse.change"]),
                          proposed, proposed, proposed)
        run = session(backend)
        result = run.run()
        self.assertEqual(result["status"], "control_error")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(result["submission"]["tool_calls"], [])

    def test_missing_approval_is_repaired_without_claiming_execution_or_duplicating_a_write(self):
        pending = [update("write")]
        done = [update("write", "done", ["obs:1"])]
        backend = Backend(plan([task("write", "write")], ["warehouse.change"]),
                          call("warehouse.change"), review(pending),
                          review(pending, approve=True, task_id="write"),
                          review(done, True), "Done", review(done, True))
        run = session(backend, results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(len(run.env.calls), 1)
        error = next(e for e in result["trace"] if e["type"] == "control_rejected")
        self.assertIn("boolean approve", error["error"])
        feedback = json.loads(backend.seen[3][0][-1]["content"])
        self.assertEqual(feedback["previous_control_reply"], review(pending))
        approvals = [e for e in result["trace"] if e["type"] == "task_review" and "approved" in e]
        self.assertEqual(len(approvals), 1)
        self.assertTrue(approvals[0]["approved"])

    def test_json_followed_by_final_text_does_not_bypass_review_contract(self):
        done = [update("answer", "done", ["request"])]
        mixed = review(done, True) + "\n<final>Done</final>"
        backend = Backend(plan([task("answer", "answer")], []), "Done", mixed, review(done, True))
        run = session(backend)
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(run.env.calls, [])
        self.assertIn("Extra data", next(e["error"] for e in result["trace"] if e["type"] == "control_rejected"))

    def test_tool_call_in_plan_never_creates_or_modifies_state(self):
        proposed = call("warehouse.change", object_id="invented")
        run = session(Backend(proposed, proposed))
        result = run.run()
        self.assertEqual(result["status"], "control_error")
        self.assertEqual(run.env.calls, [])
        self.assertEqual(result["model_calls"], 2)


class AblationTests(unittest.TestCase):
    def test_variants_reuse_backend_and_same_ids_but_get_independent_runs(self):
        with fixture() as (root, _, __, ___):
            backend = object()
            configs = {"first": agent.AgentConfig(workflow_enabled=False), "second": agent.AgentConfig()}
            def runner(*args, **kwargs):
                name = Path(args[2]).name
                return {"test_submition_path": name + ".json", "trace_dir": name + "-traces"}
            from pathlib import Path
            with patch.object(train_eval, "run_train_agent", side_effect=runner) as run, \
                 patch.object(train_eval, "evaluate_submission", return_value=({"ordered_exact_accuracy": 0}, [])) as evaluate:
                result = train_eval.run_ablation(root, root / "speech", root / "ablation", backend=backend,
                                                 sample_ids=["b", "a"], variants=configs)
            self.assertEqual(set(result), set(configs))
            self.assertEqual(run.call_count, 2)
            for invocation in run.call_args_list:
                self.assertIs(invocation.kwargs["backend"], backend)
                self.assertEqual(invocation.kwargs["sample_ids"], ["b", "a"])
            self.assertEqual(evaluate.call_count, 2)
            for invocation in evaluate.call_args_list:
                self.assertTrue(invocation.kwargs["require_finished"])
            saved = json.loads((root / "ablation/ablation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_ids"], ["b", "a"])
            self.assertFalse(saved["variants"]["first"]["config"]["workflow_enabled"])


if __name__ == "__main__":
    unittest.main()
