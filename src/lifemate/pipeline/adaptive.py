"""Recoverable controller: model proposals, code-owned transitions and receipts."""
from __future__ import annotations

from copy import deepcopy
import json

from lifemate import workflow
from . import planning, predicates, proposals, recovery
from .planning import ControllerError
from .state import encoded


PLAN_PROMPT = """Return one concise JSON proposal with reason, goals, steps only.
Follow payload.proposal_contract. Code supplies base_version and operation_id;
omit both. reason is one short sentence, not a narrative. Omit unchanged optional
step fields to use their defaults. Never output statuses or execution evidence.

This is a JSON SHAPE example, not a plan to copy:
{"reason":"short reason","goals":[{"id":"g1","description":"requested outcome",
"kind":"read","source":{"ref":"request","quote":"exact span from sources"}}],
"steps":[{"id":"s1","goal_id":"g1","kind":"read","tool":"exact allowed tool name"}]}
Goal kinds: read, write, answer, maintain, unsupported. Keep EVERY requested outcome
including conditional ones. Honor the user's final correction. Source quotes must
occur verbatim in sources. An unsupported goal needs an answer step and ask/stop,
never successful finish. Preserving current state is maintain/answer, not a write.

Each executable step implements one tool. tool must match its read/write kind;
answer/maintain/deferred use null. depends_on is an ARRAY of step IDs, e.g. [] or
["s1"], never null or a string. Use lookup prerequisites for missing IDs/amounts.
expected_arguments is {} or a grounded subset; actor fills the rest. Never invent
IDs, times, paths or arguments. Notify about a change only AFTER it succeeds.

Plan incrementally: when later work needs results not observed yet, retain its
goal and add a kind=deferred step with tool=null and depends_on pointing to the
lookup steps. Deferred is NOT executable or completed; it must be planned later.
It has empty expected_arguments, postcondition=null. Do not invent a future
result schema to express a condition. Resolve deferred work from real observations
and expand it under the same step ID to preserve its logical operation identity.

when is null unless the USER specified a condition whose result fields are known.
Lookup prerequisites are not business conditions; never gate a lookup on itself.
when has source and predicate. postcondition is null or a JSON predicate, NOT prose.
Example expression shape: {"op":"eq","left":{"step":"s1","path":["field",0,"flag"]},
"right":{"value":true}}. Replace example fields using actual observed data.
Operators: eq, ne, lt, lte, gt, gte, contains; all/any use items:[expressions]; not
uses item:expression. Operands are {"value":literal} or {"step":id,"path":[keys]}.
Every comparison needs observed step data. when refs must be dependencies;
postcondition may also reference its own tool step. Missing data stays unknown.
Numeric thresholds use the requested payment amount, not simply zero.

On revisions, preserve prior goal IDs/source anchors and every done/uncertain step
EXACTLY. Preserve existing operation IDs from the snapshot. Only reinterpret may
change prior goal meanings against the immutable request. Never replay history.
"""

DECISION_PROMPT = """Return one JSON decision with type, plan_version, reason and the
fields for that type. Do not rewrite task statuses or invent execution evidence.
type=retry: add step_id (a ready step), select a different grounded action/arguments.
type=replan: repair missing lookup steps, wrong dependencies or the unfinished route.
type=reinterpret: correct mistaken meaning/conditions/goals against immutable request.
type=ask: add tool_name and params (list of missing parameter names); these must
match a real available tool. This executes sys.ask_missing_info through the executor.
type=stop: no supported path; reports blocked after checking full allowed catalog.
type=finish: add message, covered_goal_ids (all IDs), answer_steps (ready answer IDs).
type=answer: add step_id, message, evidence (source/receipt refs); record an
intermediate ready answer without finishing the whole request. Use it when later
steps depend on an answer. Unsupported goals cannot be answered as fulfilled.
Finish only when all requested outcomes are satisfied or their condition is proven
false. A successful lookup is not a successful write. Empty collections are data,
not proof of arbitrary conditions. Do not finish an unsupported/missing action as an
answer: use ask/stop or replan. When a final answer reveals an omitted request, use
reinterpret to add the missing goal. When plans repeatedly fail, reconsider meaning,
not just JSON format. Tool failures/no-ops need retry/lookup/alternative/replan.
type=reconcile: add step_id (uncertain), verification_step (successful read for that
same goal). Only declared postconditions can prove its outcome; never retry an
uncertain write blindly. If verification is unavailable, ask or stop.
"""

APPROVAL_PROMPT = """PRE-EXECUTION APPROVAL. Return only
{decision:approve|reject|replan|reinterpret,plan_version:INTEGER,step_id:ACTIVE_ID,reason:TEXT}.
The candidate has NOT executed. Approve if it implements the user's intended
operation with grounded arguments and a ready step. Execution evidence is NOT
required to APPROVE; only after execution may code confirm completion. Do not
mark steps done or return the full ledger. Reject unsupported arguments/objects;
replan/reinterpret when the plan or its meaning is wrong. Use candidate_tool_schema.
"""


def control_messages(session, instruction, payload=None):
    state = session.controller_state
    data = {"immutable_input": json.loads(state.request_json), "sources": state.sources,
            "controller": state.snapshot(), "tool_directory": session.tool_directory,
            "payload": deepcopy(payload or {})}
    if payload and "candidate_call" in payload:
        name = payload["candidate_call"]["name"]
        data["candidate_tool_schema"] = next(deepcopy(t["function"]) for t in session.native_tools if t["function"]["name"] == name)
    system = workflow.CONTROL_SYSTEM_PROMPT + "\nCONTROL PHASE: " + instruction
    return [{"role": "system", "content": system}, {"role": "user", "content": encoded(data)}]


def actor_messages(session):
    messages = deepcopy(session.messages)
    state = session.controller_state
    guidance = ("\nUse the controller active_step only. One native tool call for its exact tool. "
                "To do another operation, reply <controller_request>{\"type\":\"replan\",\"reason\":\"...\"}"
                "</controller_request>, or use type reinterpret if the request was misunderstood. "
                "Use sys.ask_missing_info only for genuinely missing user input. Do not repeat completed "
                "operations. A plain final is only a proposal.\n")
    messages[0]["content"] += guidance + workflow.ACT_ARGUMENT_GUIDANCE + "\n" + encoded(state.snapshot())
    if session.config.include_examples:
        messages[0]["content"] += "\nA lookup that determines a condition is unconditional; its dependent action is conditional. Approval authorizes a proposed write; only a later result can confirm it. Keeping state unchanged needs no write. A no-op or missing result requires recovery."
    return messages


def generate(session, phase, instruction, payload=None):
    reply = session._generate_reply(phase, control_messages(session, instruction, payload), [], True)
    if reply is None:
        return None
    try:
        if reply.truncated:
            raise ControllerError("truncated", "Control output was truncated; nothing executed", category="format")
        value = workflow.parse_control(reply.text)
    except workflow.WorkflowError as error:
        return recovery.report(session, ControllerError("control_json", str(error), category="format"),
                               raw=reply.text, retry_phase=session.controller_state.phase)
    except ControllerError as error:
        return recovery.report(session, error, raw=reply.text, retry_phase=session.controller_state.phase)
    return value, reply.text


def plan(session):
    state = session.controller_state
    version = state.version
    result = generate(session, state.phase, PLAN_PROMPT,
                      {"proposal_contract": proposals.contract(version)})
    if not isinstance(result, tuple):
        return result or {"type": "stopped", "status": session.status}
    value, raw = result
    try:
        if state.version != version:
            raise ControllerError("stale_version", "Controller changed while generating the proposal")
        value, defaults = proposals.compile_proposal(value, plan_version=version,
                                                     previous_steps=state.steps)
        goals, steps = planning.parse_plan(value, sources=state.sources, tools=session.tools,
            max_tasks=session.config.max_tasks, previous=state if state.version else None,
            reinterpret=state.phase == "reinterpret")
        unchanged = state.goals == goals and state.steps == steps
        if unchanged:
            raise ControllerError("unchanged_plan", "A repair must change the failed interpretation/plan; choose another recovery strategy")
    except ControllerError as error:
        return recovery.report(session, error, raw=raw, retry_phase=state.phase)
    state.install(goals, steps, value["reason"], reinterpret=state.phase == "reinterpret")
    session.tasks = {sid: {**deepcopy(s), "status": state.statuses[sid]} for sid, s in state.steps.items()}
    names = list(dict.fromkeys(s["tool"] for s in steps.values() if s["tool"]))
    session.active_names = set() if session.config.route_tools else set(session.tools)
    session._set_route(names)
    event = {"type": "plan_revision", "plan_version": state.version,
             "interpretation_version": state.interpretation_version, "reason": value["reason"],
             "compiled_defaults": defaults,
             "goals": list(deepcopy(goals).values()), "steps": list(deepcopy(steps).values())}
    session.trace.append(event)
    return event


def approve(session):
    state = session.controller_state
    pending = state.pending_review
    result = generate(session, "approve", APPROVAL_PROMPT, {"candidate_call": pending["call"]})
    if not isinstance(result, tuple):
        return result or {"type": "stopped", "status": session.status}
    value, raw = result
    try:
        if set(value) != {"decision", "plan_version", "step_id", "reason"} or not isinstance(value["reason"], str):
            raise ControllerError("approval_contract", "Approval needs decision, plan_version, step_id, reason only", category="format")
        if type(value["plan_version"]) is not int or value["plan_version"] != state.version or value["step_id"] != pending["task_id"]:
            raise ControllerError("stale_approval", "Approval must reference the current plan and pending step", category="format")
        choice = value["decision"]
        if choice in ("replan", "reinterpret"):
            return recovery.revise(session, reinterpret=choice == "reinterpret", reason=value["reason"])
        if choice == "reject":
            state.pending_review = None
            return recovery.report(session, ControllerError("write_rejected", value["reason"],
                step_id=value["step_id"], category="action"), raw=raw)
        if choice != "approve":
            raise ControllerError("approval_contract", "Unknown approval decision", category="format")
        state.guard(pending["task_id"], pending["call"], pending["plan_version"])
    except ControllerError as error:
        return recovery.report(session, error, raw=raw, retry_phase="approve")
    session.pending = deepcopy(pending)
    state.pending_review, state.phase = None, "act"
    session.trace.append({"type": "write_approval", "approved": True, "step_id": value["step_id"],
                          "plan_version": state.version, "reason": value["reason"]})
    return deepcopy(session.pending)


def decision(session):
    state = session.controller_state
    result = generate(session, state.phase, DECISION_PROMPT, {"candidate_final": state.candidate_final})
    if not isinstance(result, tuple):
        return result or {"type": "stopped", "status": session.status}
    value, raw = result
    try:
        if type(value.get("plan_version")) is not int or value["plan_version"] != state.version or not isinstance(value.get("reason"), str):
            raise ControllerError("decision_contract", "Decision needs current plan_version and reason", category="format")
        choice = value.get("type")
        extras = {"retry": {"step_id"}, "replan": set(), "reinterpret": set(),
                  "ask": {"tool_name", "params"}, "stop": set(),
                  "finish": {"message", "covered_goal_ids", "answer_steps"},
                  "answer": {"step_id", "message", "evidence"},
                  "reconcile": {"step_id", "verification_step"}}
        if not isinstance(choice, str) or choice not in extras or set(value) != {"type", "plan_version", "reason"} | extras[choice]:
            raise ControllerError("decision_contract", "Unknown decision type or fields", category="format")
        if choice in ("replan", "reinterpret"):
            return recovery.revise(session, reinterpret=choice == "reinterpret", reason=value["reason"])
        if choice == "retry":
            sid = value["step_id"]
            if not isinstance(sid, str) or sid not in state.ready():
                raise ControllerError("not_ready", "Retry must choose a ready step", field="step_id")
            state.active_id, state.phase = sid, "act"
        elif choice == "ask":
            name, params = value["tool_name"], value["params"]
            if not isinstance(name, str) or name not in session.tools or not isinstance(params, list) or not params or any(not isinstance(p, str) or p not in session.tools[name]["parameters"].get("properties", {}) for p in params):
                raise ControllerError("missing_input", "Ask must name an allowed tool and real missing parameter fields")
            call = {"name": "sys.ask_missing_info", "arguments": {"tool_name": name, "params": params}}
            if not session.env.validate(call["name"], call["arguments"])["ok"]:
                raise ControllerError("ask_schema", "Missing-information call failed schema validation")
            if session._expected_log_count >= session.config.max_calls:
                session.status = "call_limit"
                return {"type": "stopped", "status": session.status}
            session.pending = {"type": "call", "call": call, "content": "", "schema_ok": True,
                               "task_id": None, "plan_version": state.version}
            return deepcopy(session.pending)
        elif choice == "stop":
            if session.active_names != set(session.tools):
                session.active_names = set(session.tools)
                state.phase = "recover"
                event = {"type": "catalog_recheck", "executed": False, "reason": value["reason"]}
                session.trace.append(event)
                return event
            session.status = "blocked"
        elif choice == "reconcile":
            reconcile(session, value["step_id"], value["verification_step"])
            state.phase = "act"
        elif choice == "finish":
            finish(session, value)
        elif choice == "answer":
            sid, refs = value["step_id"], value["evidence"]
            if (not isinstance(sid, str) or sid not in state.ready() or state.steps[sid]["kind"] != "answer"
                    or state.goals[state.steps[sid]["goal_id"]]["kind"] == "unsupported"
                    or not isinstance(value["message"], str) or not value["message"].strip()
                    or not isinstance(refs, list) or not refs
                    or any(not isinstance(r, str) or r not in set(state.sources) | set(state.receipts) for r in refs)
                    or predicates.evaluate(state.steps[sid]["postcondition"], state.results()) != "true"):
                raise ControllerError("answer_evidence", "Intermediate answer needs a ready answer step and actual source/receipt evidence")
            state.statuses[sid], state.phase = "done", "act"
            session.messages.append({"role": "assistant", "content": value["message"]})
    except (ControllerError, TypeError) as error:
        if not isinstance(error, ControllerError):
            error = ControllerError("decision_contract", str(error), category="format")
        return recovery.report(session, error, raw=raw, retry_phase=state.phase)
    state.issue, state.bad_reply = None, None
    event = {"type": "controller_decision", "decision": choice, "plan_version": state.version,
             "reason": value["reason"], "status": session.status}
    session.trace.append(event)
    return event


def finish(session, value, *, reviewed=True):
    state = session.controller_state
    covered, answers = value["covered_goal_ids"], value["answer_steps"]
    if not isinstance(value["message"], str) or not value["message"].strip() or not isinstance(covered, list) or any(not isinstance(g, str) for g in covered) or len(covered) != len(set(covered)) or set(covered) != set(state.goals):
        raise ControllerError("goal_coverage", "Finish must cover every interpreted goal", field="covered_goal_ids")
    if not isinstance(answers, list) or any(not isinstance(s, str) for s in answers) or len(answers) != len(set(answers)):
        raise ControllerError("answer_steps", "answer_steps must contain unique IDs", category="format")
    candidate_statuses = deepcopy(state.statuses)
    if any(g["kind"] == "unsupported" and any(s["goal_id"] == gid and s["kind"] == "answer"
            and state.statuses[sid] != "skipped" for sid, s in state.steps.items()) for gid, g in state.goals.items()):
        raise ControllerError("unsupported_goal", "Unsupported outcomes require ask/stop, not success")
    for sid in answers:
        step = state.steps.get(sid, {})
        if (step.get("kind") != "answer" or candidate_statuses.get(sid) != "pending"
                or any(candidate_statuses[d] != "done" for d in step["depends_on"])
                or predicates.evaluate(step["when"]["predicate"] if step["when"] else None, state.results()) != "true"):
            raise ControllerError("answer_evidence", "Only ready answer steps can be finalized", step_id=sid)
        candidate_statuses[sid] = "done"
        if predicates.evaluate(state.steps[sid]["postcondition"], state.results()) != "true":
            raise ControllerError("answer_evidence", "Answer postcondition is not verified", step_id=sid)
    unresolved = [sid for sid, status in candidate_statuses.items() if status not in ("done", "skipped")]
    if unresolved:
        raise ControllerError("incomplete_goals", "Unresolved steps: " + ", ".join(unresolved), field="steps")
    state.statuses = candidate_statuses
    session.status = "finished"
    session.messages.append({"role": "assistant", "content": value["message"]})
    session.trace.append({"type": "finish", "text": value["message"], "reviewed": reviewed,
                          "plan_version": state.version})


def reconcile(session, sid, verification_sid):
    state = session.controller_state
    if not isinstance(sid, str) or not isinstance(verification_sid, str) or state.statuses.get(sid) != "uncertain":
        raise ControllerError("reconcile", "Select an uncertain step")
    original, verification = state.steps[sid], state.steps.get(verification_sid, {})
    if verification.get("kind") != "read" or verification.get("goal_id") != original["goal_id"] or state.statuses.get(verification_sid) != "done" or original["postcondition"] is None:
        raise ControllerError("reconcile", "Reconciliation needs a successful same-goal read and a predeclared postcondition", step_id=sid)
    receipts = list(state.receipts.values())
    original_index = max((i for i, r in enumerate(receipts) if r.step_id == sid), default=-1)
    verification_index = max((i for i, r in enumerate(receipts) if r.step_id == verification_sid and r.outcome == "success"), default=-1)
    if original_index < 0 or verification_index <= original_index:
        raise ControllerError("stale_verification", "Verification must execute after the uncertain operation", step_id=sid)
    receipt = receipts[original_index]
    expression = deepcopy(original["postcondition"])
    def replace(value):
        if isinstance(value, dict):
            if value.get("step") == sid:
                value["step"] = verification_sid
            for child in value.values():
                replace(child)
        elif isinstance(value, list):
            for child in value:
                replace(child)
    replace(expression)
    results = state.results()
    if predicates.evaluate(expression, results) != "true":
        raise ControllerError("reconcile", "Verification does not prove the original postcondition", step_id=sid)
    # Only positive conjunctions of actually verified equalities can bind the
    # original action values. An unused any/not branch or a tautology cannot.
    compared = []
    def collect(expr):
        if expr["op"] == "all":
            for child in expr["items"]:
                collect(child)
        elif expr["op"] == "eq":
            left, right = expr["left"], expr["right"]
            observed = [v for v in (left, right) if v.get("step") == verification_sid and v.get("path")]
            if not observed or predicates.evaluate(expr, results) != "true":
                raise ControllerError("reconcile_binding", "Each equality must prove a field of the new verification result", step_id=sid)
            compared.extend([predicates.operand(left, results), predicates.operand(right, results)])
        else:
            raise ControllerError("reconcile_binding", "Reconciliation only accepts positive observed equality proofs", step_id=sid)
    collect(expression)
    arguments = json.loads(receipt.call_json)["arguments"]
    if any(isinstance(v, (dict, list)) or not any(type(v) is type(c) and v == c for c in compared) for v in arguments.values()):
        raise ControllerError("reconcile_binding", "Verification must bind all original action argument values", step_id=sid)
    state.statuses[sid] = "done"
    if original["kind"] == "write":
        state.successful_operations[original["operation_id"]] = receipt
    state.refresh()
    session.trace.append({"type": "execution_reconciled", "step_id": sid, "verification_step": verification_sid,
                          "original_receipt": receipt.ref, "plan_version": state.version})


def propose(session):
    state = session.controller_state
    if state.phase in ("plan", "replan", "reinterpret"):
        return plan(session)
    if state.phase == "approve":
        return approve(session)
    if state.phase in ("recover", "finish"):
        return decision(session)
    ready = state.ready()
    resolved = all(s in ("done", "skipped") for s in state.statuses.values())
    if resolved and not session.config.review_finish:
        try:
            finish(session, {"message": "Đã hoàn tất các bước theo kế hoạch đã xác minh.",
                             "covered_goal_ids": list(state.goals), "answer_steps": []}, reviewed=False)
            return {"type": "finish", "reviewed": False}
        except ControllerError as error:
            return recovery.report(session, error)
    if not ready or resolved:
        state.phase = "finish" if resolved else "recover"
        if state.issue is None and state.phase == "recover":
            state.issue = {"code": "no_ready_step", "category": "plan", "message": "Dependencies/conditions have no executable next step"}
        return decision(session)
    if state.active_id not in ready:
        state.active_id = ready[0]
    step = state.steps[state.active_id]
    if step["kind"] == "answer":
        state.phase = "finish"
        return decision(session)
    from .actor import propose_action
    return propose_action(session)


def prepare_call(session, call, content, raw):
    state = session.controller_state
    sid = state.active_id
    if call["name"] == "sys.ask_missing_info":
        args = call["arguments"]
        name, params = args.get("tool_name"), args.get("params")
        if not isinstance(name, str) or name not in session.tools or not isinstance(params, list) or not params or any(not isinstance(p, str) or p not in session.tools[name]["parameters"].get("properties", {}) for p in params):
            return recovery.report(session, ControllerError("ask_schema", "Ask needs an installed tool and missing parameter fields", category="action"), raw=raw)
        sid = None
    else:
        try:
            state.guard(sid, call, state.version)
        except ControllerError as error:
            return recovery.report(session, error, raw=raw)
    pending = {"type": "call", "call": deepcopy(call), "content": content,
               "schema_ok": True, "task_id": sid, "plan_version": state.version}
    if sid and state.steps[sid]["kind"] == "write" and session.config.review_writes:
        state.pending_review, state.phase = pending, "approve"
        return {"type": "write_review_needed", "step_id": sid, "plan_version": state.version}
    session.pending = pending
    return deepcopy(pending)


def on_final(session, text):
    state = session.controller_state
    state.candidate_final, state.phase = text, "finish"
    if not session.config.review_finish:
        ready_answers = [s for s in state.ready() if state.steps[s]["kind"] == "answer"]
        try:
            finish(session, {"message": text, "covered_goal_ids": list(state.goals), "answer_steps": ready_answers}, reviewed=False)
            return {"type": "finish", "text": text}
        except ControllerError as error:
            return recovery.report(session, error)
    return {"type": "finish_review_needed"}


def before_execute(session, pending):
    state = session.controller_state
    try:
        if pending.get("plan_version") != state.version:
            raise ControllerError("stale_execution", "Pending call belongs to an older plan")
        if pending["task_id"] is not None:
            state.guard(pending["task_id"], pending["call"], pending["plan_version"])
    except ControllerError as error:
        session.pending = None
        return recovery.report(session, error)
    return None


def observation(session, pending, result, *, uncertain=False):
    state = session.controller_state
    sid = pending["task_id"]
    if sid is None:
        return None
    # An exception may happen before the environment appends a call. Receipt
    # identity must therefore not depend on the environment's call count.
    ref = f"{'uncertain' if uncertain else 'obs'}:{len(state.receipts) + 1}"
    receipt = state.record(sid, pending["call"], result, ref, pending["plan_version"], uncertain=uncertain)
    session.observations[ref] = receipt.snapshot()
    session.tasks = {i: {**deepcopy(s), "status": state.statuses[i]} for i, s in state.steps.items()}
    step = state.steps[sid]
    if receipt.outcome != "success" or state.statuses[sid] != "done":
        state.issue = {"code": "uncertain_execution" if uncertain else "outcome_unverified",
                       "category": "uncertain" if state.statuses[sid] == "uncertain" else "runtime",
                       "step_id": sid, "message": "Inspect result/postcondition; retry reads, revise plan or reconcile before retrying a write"}
        state.phase = "recover"
        if state.stalls[step["goal_id"]] >= session.config.max_no_progress:
            session.status = "no_progress"
            state.exhausted.append(deepcopy(state.issue))
    else:
        state.phase, state.issue, state.bad_reply = "act", None, None
    session.trace.append({"type": "observation_verified", "step_id": sid, "goal_id": step["goal_id"],
                          "plan_version": state.version, "outcome": receipt.outcome,
                          "step_status": state.statuses[sid], "observation_ref": ref})
    return ref
