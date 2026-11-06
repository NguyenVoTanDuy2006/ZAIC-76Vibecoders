"""Canonical session snapshots and exclusive submission serialization."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json

from lifemate import workflow
from .config import AgentInputError


def session_result(session):
    session._check_log()
    if session.controller_state is not None:
        session.tasks = {sid: {**deepcopy(step), "status": session.controller_state.statuses[sid]}
                         for sid, step in session.controller_state.steps.items()}
    return {
        "status": session.status, "elapsed_seconds": session.elapsed(),
        "model_calls": session.model_calls, "invalid_replies": session.invalid_replies,
        "consecutive_invalid_replies": session.consecutive_invalid_replies,
        "trace": deepcopy(session.trace),
        "diagnostics": {
            "initial_messages": deepcopy(session.initial_messages),
            "tools": deepcopy(session.native_tools),
            "agent_config": asdict(session.config),
            "workflow_protocol": workflow.protocol_version(session.config),
            "backend_type": type(session.backend).__name__,
            "model_source": getattr(getattr(session.backend, "model", None), "name_or_path", None),
            "workflow": {"tasks": deepcopy(list((session.tasks or {}).values())),
                         "observations": deepcopy(list(session.observations.values())),
                         "active_tools": sorted(session.active_names)},
            "controller": ({**session.controller_state.snapshot(),
                            "plan_history": deepcopy(session.controller_state.revisions)}
                           if session.controller_state is not None else None),
            "call_protocol": session.call_state.snapshot() if session.call_state is not None else None,
            "first_model_input": deepcopy(session.first_model_input),
        },
        "submission": {
            "id": session.request["id"],
            "tool_calls": deepcopy(session.env.get_tool_calls()),
            "state_hash": session.env.get_state_hash(),
        },
    }


def write_submission(path, records, expected_ids=None):
    """Write only the three official fields; keep debug trace in a different file."""
    records = list(records)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise AgentInputError("Submission có ID trùng")
    if expected_ids is not None:
        expected_ids = list(expected_ids)
        if len(expected_ids) != len(set(expected_ids)) or set(ids) != set(expected_ids):
            raise AgentInputError("Submission thiếu/thừa ID")
    for record in records:
        if set(record) != {"id", "tool_calls", "state_hash"}:
            raise AgentInputError("Dòng nộp phải có đúng id, tool_calls, state_hash")
        if not isinstance(record["tool_calls"], list) or not isinstance(record["state_hash"], str):
            raise AgentInputError("Sai kiểu dữ liệu submission")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: do not accidentally overwrite an earlier experiment.
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return path
