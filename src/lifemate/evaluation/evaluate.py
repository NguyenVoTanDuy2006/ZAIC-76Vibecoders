"""BTC score formula plus independent, evidence-based agent diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from .io import code_fingerprint, index_rows, sha256, write_json, write_rows
from .matching import MatchConvention, argument_differences, match_calls, maximum_matching, prf
from .runtime import state_differences
from .validate import read_predictions, validate_prediction


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _near_calls(calls, gold, pairs, tools, convention):
    matched_left, matched_right = {p[0] for p in pairs}, {p[1] for p in pairs}
    result = []
    for i, call in enumerate(calls):
        if i in matched_left:
            continue
        candidates = [{"submitted_index": i, "gold_index": j,
            "differences": argument_differences(call["arguments"], target["arguments"],
                tools.get(call["name"], {}).get("parameters", {}), convention)}
            for j, target in enumerate(gold) if j not in matched_right and call["name"] == target["name"]]
        if candidates:
            best = min(len(c["differences"]) for c in candidates)
            nearest = [c for c in candidates if len(c["differences"]) == best]
            for candidate in nearest:
                candidate["ambiguous"] = len(nearest) > 1
                candidate["evidence_level"] = "observed_argument_difference; pairing_is_diagnostic_only"
                result.append(candidate)
    return result


def evaluate(items, gold_rows, predictions_path, runtime, output, *, traces=None, convention=None):
    convention = convention or MatchConvention()
    items, gold = index_rows(items), index_rows(gold_rows)
    if not items or items.keys() != gold.keys():
        raise ValueError("Evaluation requires exactly one verified gold per selected input")
    if any(not r.get("reference_valid") for r in gold.values()):
        raise ValueError("Reference replay failed; score would not be trustworthy")
    predictions, file_issues = read_predictions(predictions_path, items)
    reports = []
    counts = Counter()
    runtime_statuses = Counter()
    loss = Counter({key: 0.0 for key in ("L_invalid", "L_hash", "L_calls", "L_OOS")})
    potentials, latency, input_tokens, output_tokens = defaultdict(float), [], [], []
    for sid, item in items.items():
        target, prediction = gold[sid], predictions.get(sid)
        checked = validate_prediction(sid, prediction, item, runtime, traces)
        gcalls = target["tool_calls"]
        raw_calls = prediction.get("tool_calls") if prediction else None
        calls = raw_calls if isinstance(raw_calls, list) and all(isinstance(c, dict) and
            isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict) for c in raw_calls) else []
        pairs = match_calls(calls, gcalls, runtime.tools, convention)
        names = maximum_matching(calls, gcalls, lambda a, b: a["name"] == b["name"])
        k, g, m = len(raw_calls) if isinstance(raw_calls, list) else 0, len(gcalls), len(pairs)
        replay = checked["replay"]
        h = int(replay is not None and replay["state_hash"] == target["state_hash"])
        c = 2 * m / (k + g) if k + g else 1.0
        # Internal stop status is diagnostic. Only submitted artifact integrity
        # gates the score; valid partial outputs may still match all gold calls.
        valid = checked["submission_valid"]
        score = (float(k == 0) if g == 0 else h * c) if valid else 0.0
        bucket = "L_invalid" if not valid else "L_OOS" if g == 0 else "L_hash" if not h else "L_calls"
        loss[bucket] += 1 - score
        left, right = {i for i, _ in pairs}, {j for _, j in pairs}
        observed = []
        if not valid:
            observed.append("invalid_submission")
        if checked["runtime_valid"] is False:
            observed.append("runtime_failure")
        if g == 0 and k:
            observed.append("out_of_scope_false_action")
        if len(names) < g:
            observed.append("missing_tool_selection")
        if len(names) < k:
            observed.append("extra_tool_selection")
        if len(names) > m:
            observed.append("arguments_mismatch")
        if checked["schema_errors"]:
            observed.append("schema_error")
        if valid and g and not h:
            observed.append("final_state_mismatch")
        results = replay["results"] if replay else []
        noops = [{"call_index": i, "result": r} for i, r in enumerate(results) if r.get("done") is False]
        if noops:
            observed.append("tool_done_false")
        missing_pred = [x for x in calls if x["name"] == "sys.ask_missing_info"]
        missing_gold = [x for x in gcalls if x["name"] == "sys.ask_missing_info"]
        missing_pairs = match_calls(missing_pred, missing_gold, runtime.tools, convention)
        counts.update({"samples": 1, "valid": int(valid), "invalid": int(not valid),
            "runtime_healthy": int(checked["runtime_valid"] is True),
            "runtime_unhealthy": int(checked["runtime_valid"] is False),
            "runtime_unmeasured": int(checked["runtime_valid"] is None),
            "strict_valid": int(checked["valid"] and not checked["schema_errors"]),
            "submitted_calls": k, "gold_calls": g, "matched_calls": m if valid else 0,
            "matched_names": len(names) if valid else 0, "schema_errors": len(checked["schema_errors"]),
            "hash_matches_nonempty": int(valid and g > 0 and h), "nonempty_gold_samples": int(g > 0),
            "out_of_scope_samples": int(g == 0), "out_of_scope_correct": int(valid and g == 0 and k == 0),
            "out_of_scope_false_actions": int(g == 0 and k > 0),
            "missing_info_submitted": len(missing_pred), "missing_info_gold": len(missing_gold),
            "missing_info_matched": len(missing_pairs) if valid else 0})
        trace = checked["trace"]
        runtime_statuses[trace.get("status", "unknown") if trace else
                         "missing_trace" if traces is not None else "not_measured"] += 1
        elapsed = trace.get("elapsed_seconds") if trace else None
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            latency.append(elapsed)
        else:
            elapsed = None
        for event in trace.get("trace", []) if trace else []:
            if event.get("type") == "model_reply":
                for field, values in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
                    value = event.get(field)
                    if type(value) is int and value >= 0:
                        values.append(value)
        diffs = state_differences(replay["state"], target["canonical_state"]) if replay else []
        report = {"id": sid, "group_id": target.get("group_id"), "valid": valid,
            "submission_valid": valid, "runtime_valid": checked["runtime_valid"],
            "strict_validation_passed": checked["valid"] and not checked["schema_errors"],
            "category": "out_of_scope" if not g else "missing_info" if missing_gold else "action",
            "apps": sorted(item["context"]["installed_apps"]), "has_history": bool(item.get("history")),
            "K": k if isinstance(raw_calls, list) else None, "G": g, "M": m, "U": k - m,
            "H": h, "C": c, "score": score, "loss_bucket": bucket,
            "matched_pairs": [{"submitted_index": i, "gold_index": j} for i, j in pairs],
            "unmatched_submitted": [{"index": i, "call": call} for i, call in enumerate(calls) if i not in left],
            "unmatched_gold": [{"index": j, "call": call} for j, call in enumerate(gcalls) if j not in right],
            "tool_selection": prf(len(names), k, g), "call_match": prf(m, k, g),
            "missing_information": {**prf(len(missing_pairs), len(missing_pred), len(missing_gold)),
                "submitted": missing_pred, "gold": missing_gold},
            "argument_differences": _near_calls(calls, gcalls, pairs, runtime.tools, convention),
            "schema_errors": checked["schema_errors"], "validation_errors": checked["errors"],
            "submission_errors": checked["submission_errors"], "runtime_errors": checked["runtime_errors"],
            "state_differences": diffs, "submitted_hash": prediction.get("state_hash") if prediction else None,
            "replayed_hash": replay["state_hash"] if replay else None, "gold_hash": target["state_hash"],
            "tool_results": results, "no_effect_results": noops, "latency_seconds": elapsed,
            "runtime_status": trace.get("status") if trace else "not_measured",
            "input_audit": trace.get("input_audit") if trace else None,
            "rejected_proposals": [e for e in trace.get("trace", []) if e.get("type") == "rejected"] if trace else [],
            "observed_errors": observed, "suspected_causes": [], "confirmed_causes": [],
            "understanding_accuracy": None, "understanding_note": "Tool/argument/state evidence only; no semantic intent annotations supplied",
            "transcript_differences": "not_measured: ASR evaluation is owned by the existing ASR pipeline"}
        reports.append(report)
        for error in set(observed):
            potentials[error] += 1 - score
    n = len(reports)
    final_score = 100 * sum(r["score"] for r in reports) / n
    slices = defaultdict(list)
    for report in reports:
        slices[f"category:{report['category']}"].append(report)
        slices[f"history:{report['has_history']}"].append(report)
        slices[f"gold_calls:{report['G']}"].append(report)
        for app in report["apps"]:
            slices[f"installed_app:{app}"].append(report)
    metrics = {"score": final_score, "score_label": "local_estimate_of_competition_score",
        "scoring_convention": convention.metadata(), "counts": dict(counts),
        "prediction_file_issues": file_issues, "submission_file_valid": not file_issues and not counts["invalid"],
        "score_policy": "Score canonical submitted calls and hash independently of Agent stop status; runtime health is reported separately.",
        "strict_validation_passed": not file_issues and counts["strict_valid"] == n,
        "runtime_health": {"measured": traces is not None, "healthy_samples": counts["runtime_healthy"],
            "unhealthy_samples": counts["runtime_unhealthy"], "unmeasured_samples": counts["runtime_unmeasured"],
            "statuses": dict(runtime_statuses)},
        "loss_decomposition": {k: 100 * v / n for k, v in loss.items()},
        "loss_sum": 100 - final_score,
        "tool_selection_micro": prf(counts["matched_names"], counts["submitted_calls"], counts["gold_calls"]),
        "call_match_micro_diagnostic_only": prf(counts["matched_calls"], counts["submitted_calls"], counts["gold_calls"]),
        "missing_information_micro": prf(counts["missing_info_matched"], counts["missing_info_submitted"], counts["missing_info_gold"]),
        "hash_accuracy_nonempty": counts["hash_matches_nonempty"] / counts["nonempty_gold_samples"] if counts["nonempty_gold_samples"] else None,
        "latency": {"coverage": len(latency), "p50_seconds": _percentile(latency, .5), "p95_seconds": _percentile(latency, .95),
            "scope": "agent loop incl. ToolEnv; cached ASR text is not ASR inference latency"},
        "token_usage": {"input": sum(input_tokens) if input_tokens else None,
            "output": sum(output_tokens) if output_tokens else None, "measured_model_replies": len(input_tokens)},
        "priority_potential_points": dict(sorted(((k, 100 * v / n) for k, v in potentials.items()), key=lambda p: -p[1])),
        "priority_note": "Overlapping error groups: potentials must not be added; they are upper bounds, not promised improvements",
        "trace_coverage_checked": traces is not None, "runtime": runtime.identity, "code": code_fingerprint(),
        "predictions_sha256": sha256(predictions_path)}
    metrics["slices"] = {key: {"samples": len(rows), "score": 100 * sum(r["score"] for r in rows) / len(rows),
        "invalid": sum(not r["valid"] for r in rows)} for key, rows in sorted(slices.items())}
    output = Path(output)
    write_json(output / "metrics.json", metrics)
    write_rows(output / "samples.jsonl", reports)
    lines = [f"# Agent evaluation\n\nLocal estimated score: {final_score:.6f}/100 ({n} samples).",
        "\nCall convention has not been verified against an official BTC scorer.",
        f"\nSubmission-valid samples: {counts['valid']}; invalid submissions: {counts['invalid']}.",
        f"\nRuntime unhealthy: {counts['runtime_unhealthy']}; unmeasured: {counts['runtime_unmeasured']}. "
        "Runtime status does not override the calls/hash score.",
        "\n| Error group | Maximum recoverable points |\n|---|---:|"]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in metrics["priority_potential_points"].items())
    lines.append("\nPotentials overlap. See samples.jsonl for call indices, argument paths, schema errors and canonical state differences.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
