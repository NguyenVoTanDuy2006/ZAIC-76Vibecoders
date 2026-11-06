"""Prepare a real train subset and evaluate canonical agent tool calls offline.

Oracle transcripts are an explicit opt-in requested for this agent-only experiment;
they are never a fallback for missing ASR. No fake tools or environment-file reads.
Action labels are used only by evaluate_submission(), after inference has finished.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path


def first_train_items(data_root, count=10):
    from lifemate.agent import load_split_items

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    rows = load_split_items(data_root, "train")
    if len(rows) < count:
        raise ValueError(f"Train has {len(rows)} records; requested exactly {count}")
    return rows[:count]


def select_train_items(data_root, count=10, sample_ids=None):
    """Select by explicit IDs for held-out runs, or preserve the first-N default.

    Explicit IDs define the complete ordered subset; count is only used when IDs
    are omitted. Selection depends on metadata IDs, never labels or outcomes.
    """
    if sample_ids is None:
        return first_train_items(data_root, count)
    from lifemate.agent import load_split_items

    if isinstance(sample_ids, (str, bytes)):
        raise ValueError("sample_ids must be a sequence of IDs, not one string")
    ids = list(sample_ids)
    if not ids or any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must contain unique nonempty strings")
    indexed = {row["id"]: row for row in load_split_items(data_root, "train")}
    missing = set(ids) - indexed.keys()
    if missing:
        raise ValueError("Unknown train IDs: " + ", ".join(sorted(missing)))
    return [indexed[cid] for cid in ids]


def create_oracle_asr(data_root, output_path, count=10, *, sample_ids=None):
    """Explicit oracle-ASR experiment: copy current/history training transcripts.

    This does not perform ASR inference. Never invoke this as a fallback when a
    real ASR file is incomplete. The caller must deliberately select oracle mode.
    """
    from lifemate.asr.predictions import validate_speech_rows, write_speech_file

    selected = select_train_items(data_root, count, sample_ids)
    speech = []
    for item in selected:
        current = item.get("audio_text")
        if not isinstance(current, str) or not current.strip():
            raise ValueError(f"Missing oracle current transcript: {item['id']}")
        history = []
        for index, turn in enumerate(item.get("history", [])):
            if turn.get("role") == "user":
                text = turn.get("audio_text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing oracle history transcript: {item['id']}/{index}")
                history.append({"history_index": index, "text": text})
        speech.append({"id": item["id"], "current_text": current, "history_user_texts": history})
    validate_speech_rows(speech, selected)
    output_path = write_speech_file(output_path, speech)
    metadata = {
        "asr_source": "oracle_train_audio_text", "asr_inference": False,
        "purpose": "Agent evaluation with ground-truth transcripts, not ASR quality measurement.",
        "samples": len(selected), "history_audio_count": sum(len(row["history_user_texts"]) for row in speech),
        "selected_ids": [row["id"] for row in selected],
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def prepare_asr_file(data_root, asr_path, output_dir, count=10, *, sample_ids=None):
    """Filter explicit ASR/oracle records by the selected ordered train subset."""
    from lifemate.asr.predictions import read_jsonl, validate_speech_rows, write_speech_file

    selected = select_train_items(data_root, count, sample_ids)
    expected = {row["id"] for row in selected}
    asr_path = Path(asr_path).resolve()
    if not asr_path.is_file():
        raise FileNotFoundError(f"ASR predictions not found: {asr_path}")
    indexed = {}
    for row in read_jsonl(asr_path):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("ASR must use id/current_text/history_user_texts records")
        if row["id"] in indexed:
            raise ValueError(f"Duplicate ASR id: {row['id']}")
        indexed[row["id"]] = row
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"Missing ASR for {len(missing)} selected train IDs: {', '.join(sorted(missing)[:8])}")
    speech = [indexed[row["id"]] for row in selected]
    validate_speech_rows(speech, selected)  # Complete original user-history indices required.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"asr_predictions_first{count}.jsonl" if sample_ids is None else "asr_predictions_selected.jsonl"
    filtered = output_dir / filename
    write_speech_file(filtered, speech)
    ids_file = output_dir / "selected_ids.json"
    ids_file.write_text(json.dumps([row["id"] for row in selected], indent=2), encoding="utf-8")
    return {"asr_file": str(filtered), "selected_ids": [row["id"] for row in selected],
            "count": len(selected), "source_asr_file": str(asr_path),
            "history_audio_count": sum(len(row["history_user_texts"]) for row in speech)}


def run_train_agent(data_root, prepared_asr_file, output_dir, *, count=10, model_config=None,
                    backend=None, agent_config=None, sample_ids=None):
    """Use the existing Qwen runner and real load_toolenv hook, once for each selected sample.

    backend may be a preloaded real Qwen backend held in a notebook variable,
    or an injected backend in CPU unit tests. If omitted, the runner loads Qwen.
    """
    from lifemate.agent import run_full_pipeline
    from lifemate.asr.predictions import FileTranscriber

    selected = select_train_items(data_root, count, sample_ids)
    # The normal full-split ASR hook checks every train ID. This batch-specific
    # FileTranscriber validates exactly the requested subset instead.
    transcriber = FileTranscriber(prepared_asr_file, selected)
    result = run_full_pipeline(
        data_root=data_root, split="train", sample_ids=[row["id"] for row in selected], output_dir=output_dir,
        asr_loader=lambda *_: transcriber, asr_predictions_path=prepared_asr_file,
        model_config=model_config, backend=backend, agent_config=agent_config,
    )
    if [record["id"] for record in result["records"]] != [row["id"] for row in selected]:
        raise ValueError("Runner output does not match the requested train order/IDs")
    # Keep the user's requested filename; the JSONL emitted by the runner is also retained.
    target = Path(result["submission_path"]).parent / "test_submition.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result["records"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    result["test_submition_path"] = str(target)
    return result


def ablation_configs():
    """Isolate grounding, then the bounded terminal checkpoint on the same BTC grammar."""
    from lifemate.agent import AgentConfig

    return {
        "btc_without_grounding": AgentConfig(grounding_enabled=False, terminal_check_enabled=False, task_reasoning_enabled=False),
        "btc_with_grounding": AgentConfig(terminal_check_enabled=False, task_reasoning_enabled=False),
        "btc_with_terminal_check": AgentConfig(task_reasoning_enabled=False),
        "btc_with_task_reasoning": AgentConfig(),
    }


def run_ablation(data_root, prepared_asr_file, output_dir, *, backend, count=10,
                 sample_ids=None, variants=None, asr_source="predictions_file"):
    """Reuse preloaded weights, fresh ToolEnv per sample per variant; never train.

    Freeze the IDs before running. Use separate development/held-out IDs to avoid
    choosing prompts based on held-out labels. Results are diagnostics, not an
    official score or proof that all added phases improve the model.
    """
    from dataclasses import asdict

    selected = select_train_items(data_root, count, sample_ids)
    ids = [row["id"] for row in selected]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    variants = ablation_configs() if variants is None else variants
    if not variants or any(not isinstance(name, str) or not name.isidentifier() for name in variants):
        raise ValueError("Variant names must be nonempty safe identifiers")
    results = {}
    for name, config in variants.items():
        report = run_train_agent(data_root, prepared_asr_file, output_dir / name, backend=backend,
                                 agent_config=config, sample_ids=ids)
        summary, _ = evaluate_submission(data_root, report["test_submition_path"], sample_ids=ids,
                                         trace_dir=report["trace_dir"], asr_source=asr_source,
                                         require_finished=True)
        results[name] = {"config": asdict(config), "summary": summary,
                         "submission_path": report["test_submition_path"], "trace_dir": report["trace_dir"]}
        # Persist completed variants even if a later run fails.
        (output_dir / "ablation_summary.json").write_text(json.dumps({
            "selected_ids": ids, "asr_source": asr_source, "variants": results,
            "not_official_leaderboard_score": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _call_key(call):
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("Each tool call must have exactly name and arguments")
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        raise ValueError("Invalid tool call name/arguments types")
    return json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _overlap_metrics(gold, predicted):
    """Multiset call matching: duplicate calls remain separate requirements."""
    gold, predicted = Counter(gold), Counter(predicted)
    matched = sum((gold & predicted).values())
    required, emitted = sum(gold.values()), sum(predicted.values())
    precision = matched / emitted if emitted else float(required == 0)
    recall = matched / required if required else 1.0
    return {"matched": matched, "gold": required, "predicted": emitted,
            "missing": required - matched, "extra": emitted - matched,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _sum_metrics(metrics):
    """Micro-average from matched/gold/predicted counts, not mean sample F1."""
    matched = sum(row["matched"] for row in metrics)
    gold = sum(row["gold"] for row in metrics)
    predicted = sum(row["predicted"] for row in metrics)
    precision = matched / predicted if predicted else float(gold == 0)
    recall = matched / gold if gold else 1.0
    return {"matched": matched, "gold": gold, "predicted": predicted,
            "missing": gold - matched, "extra": predicted - matched,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _trace_diagnostics(trace_dir, indexed, details):
    """Read saved agent traces only; never open application/environment state.

    Early-finish flags are candidates based on missing gold tool *names*, not a
    semantic verdict. Exact-call disagreement can also reflect equivalent text.
    Canonical call lists/hash must match the evaluated prediction to avoid using
    an old run's trace. Unrelated IDs are counted and ignored.
    """
    path = Path(trace_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Trace directory not found: {path}")
    by_id = {row["id"]: row for row in details}
    seen, statuses, phases, protocols = set(), Counter(), Counter(), Counter()
    counts = Counter()
    trace_fields = ("rejected_events", "error_events", "grounding_rejections", "truncated_replies", "runtime_noop_results",
                    "runtime_error_results", "empty_list_results", "early_finish_candidate")
    for row in details:
        row.update({"trace_available": False, "trace_status": None})
        row.update({key: None for key in trace_fields})
    for file in sorted(path.glob("*.json")):
        trace = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(trace, dict) or not isinstance(trace.get("submission"), dict):
            raise ValueError(f"Trace has no canonical submission: {file.name}")
        record = trace["submission"]
        cid = record.get("id")
        if not isinstance(cid, str):
            raise ValueError(f"Trace submission ID must be text: {file.name}")
        if cid not in by_id:
            counts["unrelated_trace_files"] += 1
            continue
        if cid in seen:
            raise ValueError(f"Duplicate trace ID: {cid}")
        prediction = indexed.get(cid)
        if (prediction is None or set(record) != {"id", "tool_calls", "state_hash"}
                or record["state_hash"] != prediction["state_hash"]
                or not isinstance(record["tool_calls"], list)
                or [_call_key(call) for call in record["tool_calls"]]
                   != [_call_key(call) for call in prediction["tool_calls"]]):
            raise ValueError(f"Trace does not match evaluated prediction: {cid}")
        seen.add(cid)
        status = trace.get("status")
        if not isinstance(status, str):
            raise ValueError(f"Trace status must be text: {cid}")
        events = trace.get("trace")
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError(f"Trace events must be objects: {cid}")
        values = Counter()
        diagnostics = trace.get("diagnostics") or {}
        if isinstance(diagnostics.get("workflow_protocol"), str):
            protocols[diagnostics["workflow_protocol"]] += 1
        protocol_data = diagnostics.get("call_protocol") or {}
        first_call = protocol_data.get("first_tool_call_seconds")
        if type(first_call) in (int, float) and first_call >= 0:
            counts["first_call_seconds"] += first_call
            counts["first_call_measured_samples"] += 1
        for event in events:
            kind = event.get("type")
            if kind in ("model_reply", "control_reply"):
                phases[event.get("phase", "act")] += 1
                counts["model_generations"] += 1
                counter = ("control_generations" if kind == "control_reply" else
                           event["phase"] + "_generations" if event.get("phase") in ("terminal_check", "task_plan", "write_check")
                           else "action_generations")
                counts[counter] += 1
                for key in ("input_tokens", "output_tokens", "generation_seconds"):
                    number = event.get(key)
                    if isinstance(number, (int, float)) and not isinstance(number, bool):
                        counts[key] += number
                        counts[key + "_measured_replies"] += 1
            if kind in ("rejected", "control_rejected", "finish_rejected", "recovery"):
                values["rejected_events"] += 1
            if kind == "rejected" and event.get("grounding"):
                values["grounding_rejections"] += 1
            if kind in ("error", "execution_error", "execution_uncertain", "execution_rejected"):
                values["error_events"] += 1
            if kind in ("model_reply", "control_reply") and event.get("truncated") is True:
                values["truncated_replies"] += 1
            if kind == "tool_result" and isinstance(event.get("result"), dict):
                result = event["result"]
                values["runtime_noop_results"] += result.get("done") is False
                values["runtime_error_results"] += result.get("ok") is False
                lists = [value for value in result.values() if isinstance(value, list)]
                values["empty_list_results"] += bool(lists) and all(not value for value in lists)
        early = status == "finished" and by_id[cid]["missing_tool_name_calls"] > 0
        row = by_id[cid]
        row.update({"trace_available": True, "trace_status": status,
                    **{key: values[key] for key in trace_fields if key != "early_finish_candidate"},
                    "early_finish_candidate": early})
        statuses[status] += 1
        counts.update(values)
        counts["early_finish_candidates"] += early
    return {"evaluated_traces": len(seen), "missing_traces": len(details) - len(seen),
            "statuses": dict(statuses),
            "protocols": dict(protocols),
            "control_generations": counts["control_generations"],
            "action_generations": counts["action_generations"],
            "terminal_check_generations": counts["terminal_check_generations"],
            "task_plan_generations": counts["task_plan_generations"],
            "write_check_generations": counts["write_check_generations"],
            "mean_seconds_to_first_tool_call": (counts["first_call_seconds"] / counts["first_call_measured_samples"]
                                                if counts["first_call_measured_samples"] else None),
            "first_call_measured_samples": counts["first_call_measured_samples"],
            "phases": dict(phases), "model_generations": counts["model_generations"],
            **{key: counts[key] for key in ("input_tokens", "output_tokens", "generation_seconds",
                "input_tokens_measured_replies", "output_tokens_measured_replies", "generation_seconds_measured_replies")},
            **{key: counts[key] for key in (*trace_fields[:-1], "early_finish_candidates", "unrelated_trace_files")},
            "early_finish_note": "Heuristic: status finished with missing gold tool-name calls; not proof of semantic failure.",
            "empty_list_note": "All top-level list fields empty; this may be the correct lookup result."}


def evaluate_submission(data_root, submission_path, *, count=10, output_dir=None,
                        asr_source="predictions_file", sample_ids=None, trace_dir=None,
                        require_finished=False):
    """Report strict exact accuracy and general call diagnostics, not leaderboard score.

    Prediction rows are joined by ID. Missing rows count as wrong in the full N
    denominator; duplicates and IDs outside the selected subset are errors.
    Gold final_state contains only changed apps and is NOT a complete state hash.
    require_finished gates sample exact matches on status == "finished" from a
    matching saved trace. Missing status never implies successful completion.
    Call-only matches/F1 remain diagnostics. The default preserves standalone
    call-list evaluation; the Kaggle notebook and ablations enable the gate.
    """
    if require_finished and trace_dir is None:
        raise ValueError("require_finished=True needs trace_dir to verify agent status")
    selected = select_train_items(data_root, count, sample_ids)
    count = len(selected)
    submission_path = Path(submission_path)
    text = submission_path.read_text(encoding="utf-8-sig")
    predictions = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(predictions, list):
        raise ValueError("Submission must be a JSON list or JSONL records")
    expected = {row["id"] for row in selected}
    indexed = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(prediction) != {"id", "tool_calls", "state_hash"}:
            raise ValueError("Submission record must have exactly id/tool_calls/state_hash")
        cid = prediction["id"]
        if not isinstance(cid, str) or cid not in expected or cid in indexed:
            raise ValueError(f"Unexpected or duplicate prediction ID: {cid}")
        if not isinstance(prediction["tool_calls"], list) or not isinstance(prediction["state_hash"], str):
            raise ValueError(f"Invalid prediction types: {cid}")
        indexed[cid] = prediction
    details = []
    for item in selected:
        cid = item["id"]
        label = item.get("label")
        if not isinstance(label, dict) or not isinstance(label.get("tool_calls"), list):
            raise ValueError(f"Missing train label.tool_calls: {cid}")
        gold_calls = label["tool_calls"]
        gold_keys = [_call_key(call) for call in gold_calls]
        prediction = indexed.get(cid)
        predicted_calls = prediction["tool_calls"] if prediction is not None else None
        predicted_keys = [_call_key(call) for call in predicted_calls] if prediction is not None else None
        gold_names = [call["name"] for call in gold_calls]
        predicted_names = [call["name"] for call in predicted_calls] if prediction is not None else None
        name_metrics = _overlap_metrics(gold_names, predicted_names or [])
        call_metrics = _overlap_metrics(gold_keys, predicted_keys or [])
        comparable_hash = isinstance(label.get("state_hash"), str)
        details.append({
            "id": cid, "prediction_present": prediction is not None,
            "gold_call_count": len(gold_keys),
            "predicted_call_count": len(predicted_keys) if predicted_keys is not None else None,
            "ordered_exact_match": predicted_keys == gold_keys,
            "unordered_exact_match": predicted_keys is not None and Counter(predicted_keys) == Counter(gold_keys),
            "call_only_ordered_exact_match": predicted_keys == gold_keys,
            "call_only_unordered_exact_match": predicted_keys is not None and Counter(predicted_keys) == Counter(gold_keys),
            "ordered_tool_name_match": predicted_names == gold_names,
            "tool_name_metrics": name_metrics, "exact_call_metrics": call_metrics,
            "missing_tool_name_calls": name_metrics["missing"], "extra_tool_name_calls": name_metrics["extra"],
            "missing_exact_calls": call_metrics["missing"], "extra_exact_calls": call_metrics["extra"],
            "argument_mismatch_calls": name_metrics["matched"] - call_metrics["matched"],
            "gold_state_hash_available": comparable_hash,
            "state_hash_match": (prediction is not None and prediction["state_hash"] == label["state_hash"]) if comparable_hash else None,
            "gold_tool_calls": deepcopy(gold_calls), "predicted_tool_calls": deepcopy(predicted_calls),
        })
    ordered = sum(row["ordered_exact_match"] for row in details)
    unordered = sum(row["unordered_exact_match"] for row in details)
    hash_rows = [row for row in details if row["gold_state_hash_available"]]
    summary = {
        "asr_source": asr_source,
        "expected_samples": count, "predicted_samples": len(indexed),
        "missing_predictions": count - len(indexed),
        "ordered_exact_matches": ordered, "ordered_exact_accuracy": ordered / count,
        "unordered_exact_matches": unordered, "unordered_exact_accuracy": unordered / count,
        "ordered_tool_name_matches": sum(row["ordered_tool_name_match"] for row in details),
        "ordered_tool_name_accuracy": sum(row["ordered_tool_name_match"] for row in details) / count,
        "tool_name_metrics": _sum_metrics([row["tool_name_metrics"] for row in details]),
        "exact_call_metrics": _sum_metrics([row["exact_call_metrics"] for row in details]),
        "argument_mismatch_calls": sum(row["argument_mismatch_calls"] for row in details),
        "call_metrics_note": "Duplicate-aware micro precision/recall/F1 ignoring order; exact calls require name and all arguments. Argument mismatch counts unmatched exact calls among matched tool names, not semantic errors. Empty gold/predicted call sets score 1; missing prediction rows still fail sample exact accuracy.",
        "selected_ids": [row["id"] for row in selected],
        "state_hash_comparable_samples": len(hash_rows),
        "state_hash_accuracy": sum(row["state_hash_match"] for row in hash_rows) / len(hash_rows) if hash_rows else None,
        "require_finished": require_finished,
        "call_only_ordered_exact_matches": ordered, "call_only_ordered_exact_accuracy": ordered / count,
        "call_only_unordered_exact_matches": unordered, "call_only_unordered_exact_accuracy": unordered / count,
        "metric": ("Exact tool name and all argument values; primary metric also requires call order. "
                   + ("Sample exact matches additionally require status == finished in a matching trace."
                      if require_finished else "Sample exact matches do not require finished status.")),
        "not_official_leaderboard_score": True,
        "state_hash_note": "Train label.final_state covers changed apps only; it is not hashed as full gold state.",
    }
    if trace_dir is not None:
        summary["trace_diagnostics"] = _trace_diagnostics(trace_dir, indexed, details)
    else:
        for row in details:
            row.update(trace_available=False, trace_status=None)
    for row in details:
        row["agent_finished"] = row["trace_available"] and row["trace_status"] == "finished"
        if require_finished:
            row["ordered_exact_match"] = row["call_only_ordered_exact_match"] and row["agent_finished"]
            row["unordered_exact_match"] = row["call_only_unordered_exact_match"] and row["agent_finished"]
    known_statuses = sum(row["trace_available"] for row in details)
    finished = sum(row["agent_finished"] for row in details)
    summary.update(
        ordered_exact_matches=sum(row["ordered_exact_match"] for row in details),
        ordered_exact_accuracy=sum(row["ordered_exact_match"] for row in details) / count,
        unordered_exact_matches=sum(row["unordered_exact_match"] for row in details),
        unordered_exact_accuracy=sum(row["unordered_exact_match"] for row in details) / count,
        finished_samples=finished, unfinished_samples=known_statuses - finished,
        unknown_status_samples=count - known_statuses,
        finished_rate=finished / count if trace_dir is not None else None,
        status_note="Only the exact status 'finished' passes; unknown status fails the gate. Finished alone does not prove correctness. This is an internal metric, not the official scorer.",
    )
    output_dir = Path(output_dir) if output_dir is not None else submission_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "accuracy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "accuracy_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "accuracy_details.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [key for key in details[0] if not key.endswith("tool_calls")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: (json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                               if isinstance(row[key], (dict, list)) else row[key]) for key in fields}
                         for row in details)
    return summary, details
