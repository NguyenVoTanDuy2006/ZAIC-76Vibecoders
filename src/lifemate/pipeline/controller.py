"""Plan and review transitions. Receives the live session context; never executes tools."""
from __future__ import annotations

from copy import deepcopy

from lifemate import workflow


def control_reject(session, error, raw=None, phase=None):
    session.control_failures += 1
    session.control_feedback = str(error)
    session.control_bad_reply = raw
    event = {"type": "control_rejected", "phase": phase, "error": str(error), "executed": False}
    session.trace.append(event)
    if session.control_failures >= session.config.max_invalid_replies:
        session.status = "control_error"
    return event


def plan(session):
    reply = session._generate_reply("plan", session._workflow_messages(workflow.PLAN_INSTRUCTION), [], True)
    if reply is None:
        return {"type": "stopped", "status": session.status}
    try:
        if reply.truncated:
            raise workflow.WorkflowError("Truncated task plan; no tools executed")
        value = workflow.parse_control(reply.text)
        tasks = workflow.task_plan(value, session.config.max_tasks)
        names = workflow.validate_names(value.get("tool_names"), session.tools)
    except workflow.WorkflowError as error:
        return session._control_reject(error, reply.text, "plan")
    session.tasks = tasks
    if session.config.route_tools:
        session.active_names = set()
        session._set_route(names)
    session.control_failures = 0
    session.control_feedback = ""
    session.control_bad_reply = None
    event = {"type": "task_plan", "tasks": deepcopy(list(tasks.values())),
             "active_tools": sorted(session.active_names)}
    session.trace.append(event)
    return event


def review(session):
    context = session.review_context
    reply = session._generate_reply("review", session._workflow_messages(workflow.review_instruction(context), context), [], True)
    if reply is None:
        return {"type": "stopped", "status": session.status}
    try:
        if reply.truncated:
            raise workflow.WorkflowError("Truncated review; no completion or write accepted")
        value = workflow.parse_control(reply.text)
        tasks = workflow.reviewed_tasks(value, session.tasks, session.evidence, session.config.max_tasks)
        names = workflow.validate_names(value.get("next_tools", []), session.tools)
        task_id = workflow.approve_write(value, tasks, session.evidence, context["candidate_call"]) if "candidate_call" in context else None
    except workflow.WorkflowError as error:
        return session._control_reject(error, reply.text, "review")
    session.tasks = tasks
    session._set_route(names)
    session.control_failures = 0
    session.control_feedback = ""
    session.control_bad_reply = None
    session.review_context = None
    event = {"type": "task_review", "tasks": deepcopy(list(tasks.values())),
             "complete": value["complete"], "message": value["message"]}
    if "candidate_call" in context:
        event.update(approved=task_id is not None, task_id=task_id, candidate_call=deepcopy(context["candidate_call"]))
    session.trace.append(event)
    if "candidate_call" in context:
        if task_id is not None:
            session.pending = {"type": "call", "call": context["candidate_call"],
                            "content": context["content"], "schema_ok": True, "task_id": task_id}
            return deepcopy(session.pending)
        return session._reject(context["raw"], "Write review rejected: " + value["message"])
    if "candidate_final" in context and value["complete"]:
        if any(t["status"] == "blocked" for t in tasks.values()) and session.active_names != set(session.tools):
            session.active_names = set(session.tools)
            session.review_context = context
            event = {"type": "finish_rejected", "reason": "Recheck blocked tasks against full tool catalog"}
            session.trace.append(event)
            return event
        session.status = "finished" if all(t["status"] != "blocked" for t in tasks.values()) else "blocked"
        finish = {"type": "finish", "text": value["message"], "reviewed": True}
        session.messages.append({"role": "assistant", "content": value["message"]})
        session.trace.append(finish)
        return finish
    if "candidate_final" in context:
        event = {"type": "finish_rejected", "reason": value["message"], "executed": False}
        session.trace.append(event)
        # A bad initial route must not make an unsupported-task conclusion final.
        session.active_names = set(session.tools)
    return event
