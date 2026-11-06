"""Typed, bounded recovery policy. Counters survive plan/interpretation revisions."""
from __future__ import annotations

from copy import deepcopy


def report(session, error, *, raw=None, retry_phase="recover"):
    state = session.controller_state
    issue = error.as_dict()
    state.issue, state.bad_reply = issue, raw
    step = state.steps.get(error.step_id or state.active_id, {})
    key = ":".join((error.category, step.get("goal_id", "request"), error.code))
    state.failures[key] += 1
    event = {"type": "recovery", "phase": state.phase, "issue": deepcopy(issue),
             "attempt": state.failures[key], "executed": False, "plan_version": state.version}
    session.trace.append(event)
    if error.category == "format":
        scope = "format:" + state.phase
        state.failures[scope] += 1
        if state.failures[scope] >= session.config.max_invalid_replies:
            if state.failures[scope] > session.config.max_invalid_replies or state.phase in ("recover", "finish") or not state.version:
                session.status = "control_error"
            else:
                state.phase = "recover"
        else:
            state.phase = retry_phase
    elif error.category == "action":
        scope = "action:" + step.get("goal_id", "request")
        state.failures[scope] += 1
        state.phase = "recover" if state.failures[scope] >= session.config.max_action_repairs else retry_phase
        if state.failures[scope] > session.config.max_action_repairs:
            session.status = "recovery_exhausted"
    elif state.phase in ("plan", "replan", "reinterpret"):
        state.failures["invalid_plan"] += 1
        if state.failures["invalid_plan"] >= session.config.max_plan_revisions:
            session.status = "control_error"
        else:
            state.phase = retry_phase
    else:
        state.phase = "recover"
        # Valid JSON that repeats an impossible decision must also be bounded.
        if state.failures[key] >= session.config.max_no_progress:
            session.status = "no_progress"
    if session.status != "running":
        state.exhausted.append(issue)
    return event


def revise(session, *, reinterpret=False, reason):
    state = session.controller_state
    key = "interpretation_revisions" if reinterpret else "plan_revisions"
    cap = session.config.max_interpretation_revisions if reinterpret else session.config.max_plan_revisions
    if getattr(state, key) >= cap:
        session.status = "recovery_exhausted"
        state.exhausted.append({"code": key, "reason": reason})
        event = {"type": "recovery_exhausted", "budget": key, "reason": reason}
    else:
        setattr(state, key, getattr(state, key) + 1)
        state.phase = "reinterpret" if reinterpret else "replan"
        state.pending_review = None
        session.pending = None
        event = {"type": "replan_requested", "reinterpret": reinterpret, "reason": reason,
                 "plan_version": state.version}
    session.trace.append(event)
    return event
