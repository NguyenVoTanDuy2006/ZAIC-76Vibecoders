"""Consume a pending call once, execute it in the existing ToolEnv and record observations."""
from __future__ import annotations

from copy import deepcopy
import json

from lifemate import workflow
from .config import AgentInputError
from .messages import assistant_call_message, tool_result_message
from .tools import _strings


def execute_pending(session):
    session._check_log()
    if session.pending is None:
        raise AgentInputError("Không có pending call. Chạy propose trước; không chạy lại execute.")
    if session.status != "running" or not session._budget_ok():
        return {"type": "stopped", "status": session.status}
    if session.call_state is not None:
        from . import protocol
        protocol.consume(session, session.pending)
    if session.controller_state is not None:
        from . import adaptive
        rejection = adaptive.before_execute(session, session.pending)
        if rejection is not None:
            return rejection
    pending = deepcopy(session.pending)
    call = session.pending["call"]
    content = session.pending["content"]
    task_id = session.pending.get("task_id")
    before_call_count = session._expected_log_count
    session.pending = None  # consume before calling; an exception cannot re-execute it
    session.messages.append(assistant_call_message(call, content=content))
    try:
        result = session.env.call(call["name"], deepcopy(call["arguments"]))
    except Exception as error:
        session._expected_log_count = len(session.env.get_tool_calls())
        if session.call_state is not None:
            from . import protocol
            uncertain = session.tools[call["name"]].get("kind") == "write"
            result = {"ok": False, "uncertain": uncertain, "error": f"{type(error).__name__}: {error}"}
            session.messages.append(tool_result_message(call["name"], result))
            protocol.observation(session, pending, result, uncertain=True)
            event = {"type": "execution_uncertain" if uncertain else "execution_error",
                     "call": deepcopy(call), "error": result["error"], "execution_id": pending["execution_id"]}
            session.trace.append(event)
            return event
        if session.controller_state is not None and task_id is not None:
            from . import adaptive
            uncertain = session.controller_state.steps[task_id]["kind"] == "write"
            result = {"ok": False, "uncertain": uncertain, "error": f"{type(error).__name__}: {error}"}
            session.messages.append(tool_result_message(call["name"], result))
            adaptive.observation(session, pending, result, uncertain=uncertain)
            if uncertain and session._expected_log_count != before_call_count + 1:
                # No canonical execution receipt means replay cannot reconcile
                # this attempt safely. Preserve the env log/hash and stop.
                session.status = "execution_error"
            event = {"type": "execution_uncertain" if uncertain else "execution_rejected", "call": deepcopy(call), "error": result["error"]}
            session.trace.append(event)
            return event
        session.status = "execution_error"
        event = {"type": "execution_error", "call": call,
                 "error": f"{type(error).__name__}: {error}"}
        session.trace.append(event)
        return event
    session._expected_log_count = len(session.env.get_tool_calls())
    if session.controller_state is not None or session.call_state is not None:
        from .state import encoded
        calls = session.env.get_tool_calls()
        if len(calls) != before_call_count + 1 or encoded(calls[-1]) != encoded(call):
            result = {"ok": False, "error": "ToolEnv canonical call log does not match the attempted execution"}
            session.messages.append(tool_result_message(call["name"], result))
            if session.call_state is not None:
                from . import protocol
                protocol.observation(session, pending, result, uncertain=True)
            else:
                from . import adaptive
                adaptive.observation(session, pending, result, uncertain=task_id is not None and session.controller_state.steps[task_id]["kind"] == "write")
            session.status = "execution_error"
            event = {"type": "execution_error", "call": deepcopy(call), "error": result["error"]}
            session.trace.append(event)
            return event
    invalid_result = not isinstance(result, dict)
    if (session.controller_state is not None or session.call_state is not None) and not invalid_result:
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            invalid_result = True
    if (session.controller_state is not None or session.call_state is not None) and invalid_result:
        uncertain = (session.tools[call["name"]].get("kind") == "write" if session.call_state is not None
                     else task_id is not None and session.controller_state.steps[task_id]["kind"] == "write")
        result = {"ok": False, "error": "Tool result is not a finite JSON object", "uncertain": uncertain}
        session.messages.append(tool_result_message(call["name"], result))
        if session.call_state is not None:
            from . import protocol
            protocol.observation(session, pending, result, uncertain=True)
        else:
            from . import adaptive
            adaptive.observation(session, pending, result, uncertain=uncertain)
        event = {"type": "execution_uncertain" if uncertain else "execution_rejected", "call": deepcopy(call), "error": result["error"]}
        session.trace.append(event)
        return event
    session.messages.append(tool_result_message(call["name"], result))
    if isinstance(result, dict) and result.get("ok") and result.get("done") is not False:
        session.known_strings.update(_strings(result))
        session.consecutive_invalid_replies = 0
    event = {"type": "tool_result", "call": deepcopy(call), "result": deepcopy(result),
             "env_call_count": session._expected_log_count}
    session.trace.append(event)
    if session.call_state is not None:
        from . import protocol
        event["execution_id"] = pending["execution_id"]
        event["observation_ref"] = protocol.observation(session, pending, result)
    elif session.controller_state is not None:
        from . import adaptive
        ref = adaptive.observation(session, pending, result)
        if ref:
            event["observation_ref"] = ref
    elif session.config.workflow_enabled:
        ref = f"obs:{session._expected_log_count}"
        observation = {"ref": ref, "source": "tool", "call": deepcopy(call),
                       "result": deepcopy(result), "outcome": workflow.outcome(result),
                       "kind": session.tools[call["name"]].get("kind", "unknown"), "task_id": task_id}
        observation["diagnostics"] = workflow.observation_diagnostics(
            call, session.tools[call["name"]]["parameters"], result, observation["kind"])
        session.observations[ref] = observation
        session.evidence[ref] = observation
        event["observation_ref"] = ref
        if observation["outcome"] != "success":
            fingerprint = json.dumps([task_id, call, result], sort_keys=True, ensure_ascii=False)
            session.no_progress[fingerprint] = session.no_progress.get(fingerprint, 0) + 1
            if session.no_progress[fingerprint] >= session.config.max_no_progress:
                session.status = "no_progress"
        else:
            session.no_progress.clear()
        session.review_context = {"observation_ref": ref}
    if call["name"] == "sys.ask_missing_info" and result.get("ok"):
        session.status = "waiting_for_user"
    else:
        session._budget_ok()
    return event
