"""Actor proposal, schema discovery, validation and repair feedback; no tool execution."""
from __future__ import annotations

from copy import deepcopy
import json
import re

from lifemate import workflow
from .config import ModelOutputError
from .messages import assistant_call_message, tool_result_message
from .replies import parse_model_reply
from .tools import reference_lookup_hints, unverified_references
from .planning import ControllerError
from . import call_repair


def reject_reply(session, raw, message, call=None, lookup_hints=None, content="", issue=None, details=None):
    session.invalid_replies += 1
    session.consecutive_invalid_replies += 1
    feedback = {"source": "agent_validation", "executed": False, "error": message,
                "instruction": "Sửa lỗi và chỉ gọi một tool mỗi lượt; chờ tool_result trước bước tiếp theo."}
    repair = None
    if session.call_state is not None:
        repair = call_repair.feedback(call, session.tools, message=message, issue=issue,
                                      lookup_hints=lookup_hints, grounding=details)
        repair["budget"] = {"consecutive_rejections": session.consecutive_invalid_replies,
                            "rejection_limit": session.config.max_invalid_replies,
                            "remaining_before_stop": max(0, session.config.max_invalid_replies - session.consecutive_invalid_replies)}
        feedback["repair"] = repair
        feedback["instruction"] = repair["next_step"] + " " + repair["expected_output"]
        if issue is None:
            issue = ControllerError(repair["code"], message, category=repair["category"])
    if details:
        feedback["grounding"] = deepcopy(details)
    if lookup_hints:
        feedback["lookup_required"] = deepcopy(lookup_hints)
        feedback["instruction"] = (
            "Call bị chặn, chưa thực thi. Bước tiếp theo chỉ gọi một lookup_tool được liệt kê "
            "để tìm đúng đối tượng theo yêu cầu, chờ kết quả và dùng ID thật trả về. "
            "Không lặp lại ID vừa bị từ chối hoặc đoán ID khác. Nếu không có tool tra cứu "
            "phù hợp hoặc kết quả không xác định được đối tượng, hỏi thông tin còn thiếu."
        )
    if call is None:
        session.messages.append({"role": "assistant", "content": raw})
        session.messages.append({"role": "user", "content": json.dumps(feedback, ensure_ascii=False)})
    else:
        session.messages.append(assistant_call_message(call, content=content))
        session.messages.append(tool_result_message(call["name"], feedback))
    event = {"type": "rejected", "raw": raw, "error": message, "executed": False}
    if repair is not None:
        event["repair"] = deepcopy(repair)
    if lookup_hints:
        event["lookup_required"] = deepcopy(lookup_hints)
    if details:
        event["grounding"] = deepcopy(details)
    session.trace.append(event)
    if session.call_state is not None:
        from . import protocol
        protocol.reject(session, issue, message)
        event["category"] = session.call_state.proposal_issue["category"]
    elif session.controller_state is not None:
        from . import recovery
        issue = issue or ControllerError("action_rejected", message,
            step_id=session.controller_state.active_id, category="action")
        recovery.report(session, issue, raw=raw, retry_phase="act")
    elif session.consecutive_invalid_replies >= session.config.max_invalid_replies:
        session.status = "invalid_output"
    return event


def propose_action(session):
    """Generate and validate one actor proposal after the coordinator selects this phase."""
    messages = session._workflow_messages() if session.config.workflow_enabled else deepcopy(session.messages)
    reply = session._generate_reply("act", messages, session._routed_tools())
    if reply is None:
        return {"type": "stopped", "status": session.status}
    return accept_reply(session, reply)


def accept_reply(session, reply, *, terminal_checked=False, write_checked=False):
    """Validate either the actor output or one checkpoint replacement identically."""
    if reply.truncated:
        return session._reject(reply.text, "Output bị cắt bởi max_new_tokens; không thực thi",
            issue=ControllerError("actor_truncated", "Actor output was truncated", category="format"))
    if session.controller_state is not None and reply.text.strip().startswith("<controller_request>"):
        from . import recovery
        try:
            match = re.fullmatch(r"<controller_request>\s*(.*?)\s*</controller_request>", reply.text.strip(), re.S)
            value = workflow.parse_control(match.group(1)) if match else {}
            if set(value) != {"type", "reason"} or value["type"] not in ("replan", "reinterpret") or not isinstance(value["reason"], str):
                raise workflow.WorkflowError("controller_request needs type=replan|reinterpret and reason")
        except workflow.WorkflowError as error:
            return session._reject(reply.text, str(error), issue=ControllerError("controller_request", str(error), category="format"))
        return recovery.revise(session, reinterpret=value["type"] == "reinterpret", reason=value["reason"])
    if (session.config.workflow_enabled and session.call_state is None and isinstance(reply.text, str)
            and reply.text.strip().startswith("<tool_request>")):
        previous_names = set(session.active_names)
        try:
            match = re.fullmatch(r"<tool_request>\s*(.*?)\s*</tool_request>", reply.text.strip(), re.S)
            if not match:
                raise workflow.WorkflowError("Incomplete tool_request")
            request = workflow.parse_control(match.group(1))
            if request == {"all": True}:
                session.active_names = set(session.tools)
            elif set(request) == {"names"}:
                session._set_route(request["names"])
            else:
                raise workflow.WorkflowError("Use names array or all=true")
        except workflow.WorkflowError as error:
            return session._reject(reply.text, str(error))
        if session.controller_state is not None and session.active_names == previous_names:
            from . import recovery
            return recovery.report(session, ControllerError("schema_stall", "These schemas are already open; execute a ready step or revise the plan"), raw=reply.text)
        event = {"type": "tool_route", "active_tools": sorted(session.active_names), "executed": False}
        session.trace.append(event)
        return event
    try:
        parsed = parse_model_reply(reply.text)
    except ModelOutputError as error:
        return session._reject(reply.text, str(error), issue=ControllerError("actor_json", str(error), category="format"))
    if parsed["type"] == "finish":
        if session.call_state is not None:
            if session.config.terminal_check_enabled and not terminal_checked:
                from .checkpoint import review
                return review(session, {"type": "finish", "text": parsed["text"]})
            from .protocol import finish
            return finish(session, parsed["text"], reviewed=terminal_checked)
        if session.controller_state is not None:
            from .adaptive import on_final
            return on_final(session, parsed["text"])
        if session.config.workflow_enabled and session.config.review_finish:
            session.review_context = {"candidate_final": parsed["text"]}
            return {"type": "finish_review_needed"}
        session.messages.append({"role": "assistant", "content": reply.text})
        session.status = "finished"
        session.trace.append(parsed)
        return parsed
    call = parsed["call"]
    content = parsed.get("content", "")
    if call["name"] not in session.tools:
        return session._reject(reply.text, "Tool không được phép", call, content=content)
    if session.call_state is None and call["name"] not in session.active_names:
        session._set_route([call["name"]])
        session.messages.append(assistant_call_message(call, content=content))
        session.messages.append(tool_result_message(call["name"], {
            "source": "schema_discovery", "executed": False,
            "instruction": "Schema vừa được mở; đọc schema rồi đề xuất lại call. Chưa có kết quả thực thi.",
        }))
        event = {"type": "tool_route", "active_tools": sorted(session.active_names),
                 "proposed_call": deepcopy(call), "executed": False}
        session.trace.append(event)
        return event
    verdict = session.env.validate(call["name"], call["arguments"])
    if not verdict["ok"]:
        return session._reject(reply.text, f"Sai schema: {verdict.get('error')}", call, content=content)
    if session.config.check_reference_provenance:
        unknown = unverified_references(call["arguments"], session.tools[call["name"]]["parameters"],
                                        session.known_strings, session.utterances)
        if unknown:
            hints = [hint for hint in reference_lookup_hints(list(session.tools.values()),
                         session.tools[call["name"]]["parameters"])
                     if any(re.sub(r"\[\d+\]", "[]", path) == hint["parameter"] for path in unknown)]
            return session._reject(reply.text, "ID x-ref chưa được quan sát: " + ", ".join(unknown)
                                + ". Hãy tra cứu qua công cụ trước; không tự đoán ID.", call,
                                    lookup_hints=hints, content=content,
                                    issue=ControllerError("missing_reference", "Find the missing IDs through a planned lookup",
                                        step_id=session.controller_state.active_id if session.controller_state else None,
                                        field=", ".join(unknown), category="data"))
    if session.call_state is not None and session.config.grounding_enabled:
        from . import grounding
        violations = grounding.check_call(call, session.tools[call["name"]]["parameters"],
            current_text=session.speech["current_text"], request=session.request,
            observations=session.observations, known_strings=session.known_strings,
            history_user_texts=session.speech["history_user_texts"], tool_kind=session.tools[call["name"]].get("kind"))
        if violations:
            first = violations[0]
            return session._reject(reply.text, first["message"], call, content=content,
                issue=ControllerError(first["code"], first["message"], field=first["field"], category="data"),
                details=violations)
    if session._expected_log_count >= session.config.max_calls:
        session.status = "call_limit"
        return {"type": "stopped", "status": session.status}
    if session.call_state is not None:
        if call["name"] == "sys.ask_missing_info":
            problem = call_repair.missing_info_problem(call["arguments"], session.tools)
            if problem:
                return session._reject(reply.text, problem["message"], call, content=content,
                    issue=ControllerError(problem["code"], problem["message"], field=problem["field"], category="schema"))
            if session.config.terminal_check_enabled and not terminal_checked:
                from .checkpoint import review
                return review(session, {"type": "call", "call": call})
        if (session.config.task_reasoning_enabled and not write_checked
                and session.tools[call["name"]].get("kind") == "write"):
            from .task_reasoning import review_write
            return review_write(session, call)
        from .protocol import prepare_call
        return prepare_call(session, call, content, reply.text)
    if session.controller_state is not None:
        from .adaptive import prepare_call
        return prepare_call(session, call, content, reply.text)
    if (session.config.workflow_enabled and session.config.review_writes
            and session.tools[call["name"]].get("kind") != "read"
            and call["name"] != "sys.ask_missing_info"):
        session.review_context = {"candidate_call": deepcopy(call), "content": content, "raw": reply.text}
        return {"type": "write_review_needed"}
    session.pending = {"type": "call", "call": deepcopy(call), "schema_ok": True, "content": content}
    return deepcopy(session.pending)
