"""Prediction structure/schema/runtime checks, separate from score computation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from .io import call_error, loads, read_json, read_rows


SUCCESS_STATUSES = {"finished", "waiting_for_user"}


def read_predictions(path, expected_ids):
    expected = set(expected_ids)
    records, issues, by_id = [], [], {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("row must have a nonempty string id")
        except ValueError as error:
            issues.append({"line": number, "kind": "invalid_json", "error": str(error)})
            continue
        records.append(row)
        if row["id"] not in expected:
            issues.append({"line": number, "id": row["id"], "kind": "foreign_id"})
    counts = Counter(r["id"] for r in records)
    for row in records:
        sid = row["id"]
        if sid in expected:
            if counts[sid] > 1:
                by_id[sid] = {"id": sid, "_invalid": "duplicate_prediction_id"}
            else:
                by_id[sid] = row
    issues.extend({"id": sid, "kind": "duplicate_id", "count": count} for sid, count in counts.items() if count > 1)
    return by_id, issues


def load_traces(path):
    if path is None:
        return None
    path = Path(path)
    rows = [read_json(p) for p in sorted(path.glob("*.json"))] if path.is_dir() else read_rows(path)
    indexed = {}
    for row in rows:
        sid = row.get("id", row.get("submission", {}).get("id"))
        if not isinstance(sid, str) or sid in indexed:
            raise ValueError(f"Invalid/duplicate trace ID: {sid}")
        indexed[sid] = row
    return indexed


def validate_prediction(sample_id, prediction, item, runtime, traces=None):
    """Check submission integrity independently of diagnostic runtime health.

    BTC scores the submitted calls/hash, not the internal Agent status. ``valid``
    remains the strict validation result for callers checking run health;
    ``submission_valid`` is the integrity gate used by the local score.
    """
    errors, submission_errors, runtime_errors = [], [], []
    report = {"id": sample_id, "valid": False, "submission_valid": False,
              "errors": errors, "submission_errors": submission_errors,
              "runtime_errors": runtime_errors, "runtime_valid": None,
              "schema_errors": [], "replay": None, "trace": None,
              "trace_status_measured": traces is not None}

    def submission_error(error):
        submission_errors.append(error)
        errors.append(error)

    def finish():
        report["submission_valid"] = not submission_errors and report["replay"] is not None
        report["valid"] = report["submission_valid"] and not runtime_errors
        return report

    if traces is not None:
        trace = traces.get(sample_id)
        report["trace"] = trace
        if trace is None:
            runtime_errors.append({"kind": "missing_trace"})
        elif trace.get("status") not in SUCCESS_STATUSES:
            runtime_errors.append({"kind": "runtime_failure", "status": trace.get("status"), "error": trace.get("error")})
        errors.extend(runtime_errors)
        report["runtime_valid"] = not runtime_errors
    if prediction is None:
        submission_error({"kind": "missing_prediction"})
        return finish()
    if prediction.get("_invalid"):
        submission_error({"kind": prediction["_invalid"]})
        return finish()
    if set(prediction) != {"id", "tool_calls", "state_hash"}:
        submission_error({"kind": "invalid_fields", "fields": sorted(prediction)})
    calls = prediction.get("tool_calls")
    if not isinstance(calls, list):
        submission_error({"kind": "tool_calls_not_list"})
    elif len(calls) > 1000:
        submission_error({"kind": "too_many_calls", "limit": 1000})
    else:
        for index, call in enumerate(calls):
            error = call_error(call)
            if error:
                submission_error({"kind": "invalid_call", "call_index": index, "error": error})
    if not isinstance(prediction.get("state_hash"), str) or not re.fullmatch(r"[a-f0-9]{64}", prediction["state_hash"]):
        submission_error({"kind": "invalid_state_hash_format"})
    if submission_errors:
        return finish()
    if traces is not None and report["trace"] is not None and report["trace"].get("submission") != prediction:
        submission_error({"kind": "trace_submission_mismatch"})
    allowed = {t for app in item["context"]["installed_apps"].values() for t in app["tools"]} | {"sys.ask_missing_info"}
    report["schema_errors"] = [{"call_index": i, "name": call["name"], "error": error}
        for i, call in enumerate(calls)
        if (error := runtime.schema_error(call["name"], call["arguments"], runtime.tools, allowed))]
    # Schema-invalid attempted calls remain in K and in replay. Never filter them.
    try:
        replay = runtime.replay(sample_id, calls)
        report["replay"] = replay
        if replay["tool_calls"] != calls:
            submission_error({"kind": "replay_call_log_mismatch"})
        if replay["state_hash"] != prediction["state_hash"]:
            submission_error({"kind": "submitted_hash_not_reproducible", "submitted": prediction["state_hash"],
                           "replayed": replay["state_hash"]})
    except Exception as error:
        submission_error({"kind": "replay_failure", "error": f"{type(error).__name__}: {error}"})
    return finish()
