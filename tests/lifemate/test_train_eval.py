"""CPU contract/evaluation tests, not model quality or GPU inference results."""
import ast
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import textwrap
import unittest
from unittest.mock import Mock, patch
import uuid
import zipfile

from lifemate import agent, train_eval
from lifemate.asr.predictions import read_jsonl


REPO = Path(__file__).resolve().parents[2]


@contextmanager
def workspace():
    base = REPO / ".test-scratch"
    root = base / uuid.uuid4().hex
    (root / "data/train").mkdir(parents=True)
    try:
        yield root
    finally:
        assert root.resolve().parent == base.resolve()
        shutil.rmtree(root)
        if not any(base.iterdir()):
            base.rmdir()


def corpus(root):
    rows = []
    for index, cid in enumerate(("train_0001", "train_0226", "train_0002")):
        rows.append({"id": cid, "audio": f"audio/{cid}.wav", "audio_text": f"ORACLE_CURRENT_{index}",
            "history": [{"role": "assistant", "text": "Hello"},
                        {"role": "user", "audio": "audio/history.wav", "audio_text": "ORACLE_HISTORY"}],
            "context": {"installed_apps": {"calendar": {"tools": ["calendar.list_events"]}}},
            "label": {"tool_calls": [], "final_state": {"GOLD_STATE_SENTINEL": {}}}})
    data_root = root / "data"
    (data_root / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    catalog = {"apps": [{"id": "calendar", "tools": [
        {"name": "calendar.list_events", "parameters": {"type": "object"}}]},
        {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "parameters": {"type": "object"}}]}]}
    (data_root / "train/train_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return data_root, rows


def predictions_file(root, rows):
    file = root / "test_submition.json"
    file.write_text(json.dumps(rows), encoding="utf-8")
    return file


class TrainEvaluationTests(unittest.TestCase):
    def test_protocol_cost_diagnostics_separate_actor_from_control_generations(self):
        with workspace() as root:
            data, rows = corpus(root)
            record = {"id": rows[0]["id"], "tool_calls": [], "state_hash": "h"}
            path = predictions_file(root, [record])
            traces = root / "traces"; traces.mkdir()
            (traces / "000000.json").write_text(json.dumps({
                "submission": record, "status": "finished", "trace": [{
                    "type": "model_reply", "phase": "act", "input_tokens": 12,
                    "output_tokens": 4, "generation_seconds": 0.5}, {
                    "type": "model_reply", "phase": "terminal_check", "input_tokens": 15,
                    "output_tokens": 3, "generation_seconds": 0.6}, {
                    "type": "model_reply", "phase": "task_plan", "input_tokens": 17,
                    "output_tokens": 10, "generation_seconds": 0.7}, {
                    "type": "model_reply", "phase": "write_check", "input_tokens": 19,
                    "output_tokens": 4, "generation_seconds": 0.8}],
                "diagnostics": {"workflow_protocol": "btc-tool-call-v1", "call_protocol": {
                    "first_tool_call_seconds": 1.25}},
            }), encoding="utf-8")
            summary, _ = train_eval.evaluate_submission(data, path, count=1, trace_dir=traces)
            diag = summary["trace_diagnostics"]
            self.assertEqual(diag["protocols"], {"btc-tool-call-v1": 1})
            self.assertEqual(diag["control_generations"], 0)
            self.assertEqual(diag["action_generations"], 1)
            self.assertEqual(diag["terminal_check_generations"], 1)
            self.assertEqual(diag["task_plan_generations"], 1)
            self.assertEqual(diag["write_check_generations"], 1)
            self.assertEqual(diag["model_generations"], 4)
            self.assertEqual(diag["input_tokens"], 63)
            self.assertEqual(diag["mean_seconds_to_first_tool_call"], 1.25)

    def test_oracle_is_explicit_and_includes_original_history_indices(self):
        with workspace() as root:
            data, rows = corpus(root)
            output = train_eval.create_oracle_asr(data, root / "oracle.jsonl", 2)
            speech = read_jsonl(output)
            self.assertEqual([row["id"] for row in speech], ["train_0001", "train_0226"])
            self.assertEqual(speech[0]["current_text"], "ORACLE_CURRENT_0")
            self.assertEqual(speech[0]["history_user_texts"], [{"history_index": 1, "text": "ORACLE_HISTORY"}])
            self.assertNotIn("GOLD_STATE_SENTINEL", output.read_text())
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertFalse(metadata["asr_inference"])
            self.assertEqual(metadata["history_audio_count"], 2)
            with self.assertRaises(FileExistsError):
                train_eval.create_oracle_asr(data, output, 2)

    def test_missing_oracle_history_is_rejected_and_real_asr_never_falls_back(self):
        with workspace() as root:
            data, rows = corpus(root)
            del rows[0]["history"][1]["audio_text"]
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history transcript"):
                train_eval.create_oracle_asr(data, root / "oracle.jsonl", 2)
            with patch.object(train_eval, "create_oracle_asr", side_effect=AssertionError("No implicit oracle")):
                with self.assertRaises(FileNotFoundError):
                    train_eval.prepare_asr_file(data, root / "missing.jsonl", root / "prepared", 2)

    def test_subset_filters_superset_asr_by_train_order_and_requires_all_history(self):
        with workspace() as root:
            data, _ = corpus(root)
            raw = train_eval.create_oracle_asr(data, root / "oracle.jsonl", 3)
            speech = read_jsonl(raw)
            raw.write_text("".join(json.dumps(row) + "\n" for row in reversed(speech)), encoding="utf-8")
            prepared = train_eval.prepare_asr_file(data, raw, root / "prepared", 2)
            self.assertEqual(prepared["selected_ids"], ["train_0001", "train_0226"])
            self.assertEqual([row["id"] for row in read_jsonl(prepared["asr_file"])], prepared["selected_ids"])
            speech[0]["history_user_texts"] = []
            raw.write_text("".join(json.dumps(row) + "\n" for row in speech), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history"):
                train_eval.prepare_asr_file(data, raw, root / "bad-prepared", 2)

    def test_qwen_loader_and_real_toolenv_hook_are_used_by_existing_runner(self):
        with workspace() as root:
            data, _ = corpus(root)
            speech = train_eval.create_oracle_asr(data, root / "oracle.jsonl", 2)
            created, seen = [], []

            class Env:
                def __init__(self, split, cid, root):
                    created.append((split, cid, root))
                def get_tool_calls(self):
                    return []
                def get_state_hash(self):
                    return "unit-test-state-hash"

            class Backend:
                def generate(self, messages, tools, config, remaining_seconds=None):
                    seen.append(deepcopy(messages))
                    return agent.ModelReply("<final>Done</final>")

            config = agent.ModelLoadConfig(model_source="local-test-checkpoint")
            with patch.object(agent, "get_tool_env_factory", return_value=Env), \
                 patch.object(agent, "load_qwen_backend", return_value=Backend()) as load, redirect_stdout(io.StringIO()):
                result = train_eval.run_train_agent(data, speech, root / "runs", count=2, model_config=config,
                    agent_config=agent.AgentConfig(workflow_enabled=False))
            load.assert_called_once()
            self.assertIs(load.call_args.args[0], config)
            self.assertEqual([cid for _, cid, _ in created], ["train_0001", "train_0226"])
            self.assertEqual(len(created), 2)
            self.assertEqual([row["id"] for row in json.loads(Path(result["test_submition_path"]).read_text())], ["train_0001", "train_0226"])
            prompt = json.dumps(seen)
            self.assertIn("ORACLE_CURRENT", prompt)
            self.assertIn("ORACLE_HISTORY", prompt)
            self.assertNotIn("GOLD_STATE_SENTINEL", prompt)
            self.assertTrue(Path(result["submission_path"]).is_file())

    def test_exact_accuracy_joins_ids_and_distinguishes_call_order(self):
        with workspace() as root:
            data, rows = corpus(root)
            a = {"name": "calendar.a", "arguments": {"x": 1, "y": 2}}
            b = {"name": "calendar.b", "arguments": {}}
            rows[0]["label"]["tool_calls"] = [a, b]
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            path = predictions_file(root, [
                {"id": "train_0226", "tool_calls": [], "state_hash": "hash2"},
                {"id": "train_0001", "tool_calls": [b, a], "state_hash": "hash1"},
            ])
            summary, details = train_eval.evaluate_submission(data, path, count=2, asr_source="oracle_train_audio_text")
            self.assertEqual(summary["ordered_exact_accuracy"], 0.5)
            self.assertEqual(summary["unordered_exact_accuracy"], 1.0)
            self.assertEqual(summary["state_hash_comparable_samples"], 0)
            self.assertIsNone(summary["state_hash_accuracy"])
            self.assertEqual(summary["asr_source"], "oracle_train_audio_text")
            self.assertEqual([row["id"] for row in details], ["train_0001", "train_0226"])

    def test_missing_predictions_count_as_wrong_even_when_gold_calls_are_empty(self):
        with workspace() as root:
            data, _ = corpus(root)
            path = predictions_file(root, [{"id": "train_0001", "tool_calls": [], "state_hash": "x"}])
            summary, _ = train_eval.evaluate_submission(data, path, count=2)
            self.assertEqual(summary["expected_samples"], 2)
            self.assertEqual(summary["missing_predictions"], 1)
            self.assertEqual(summary["ordered_exact_accuracy"], 0.5)

    def test_finished_gate_rejects_failed_or_nonterminal_runs_even_with_exact_calls(self):
        with workspace() as root:
            data, rows = corpus(root)
            record = {"id": rows[0]["id"], "tool_calls": [], "state_hash": "canonical"}
            path = predictions_file(root, [record])
            traces = root / "traces"; traces.mkdir()
            for status in ("finished", "control_error", "invalid_output", "blocked",
                           "waiting_for_user", "timeout", "finish"):
                with self.subTest(status=status):
                    (traces / "one.json").write_text(json.dumps({
                        "submission": record, "status": status, "trace": [],
                    }), encoding="utf-8")
                    summary, details = train_eval.evaluate_submission(
                        data, path, count=1, trace_dir=traces, require_finished=True)
                    self.assertEqual(summary["ordered_exact_accuracy"], float(status == "finished"))
                    self.assertEqual(summary["unordered_exact_accuracy"], float(status == "finished"))
                    self.assertEqual(summary["call_only_ordered_exact_accuracy"], 1)
                    self.assertTrue(details[0]["call_only_ordered_exact_match"])
                    self.assertEqual(details[0]["agent_finished"], status == "finished")
                    self.assertEqual(summary["finished_samples"], int(status == "finished"))
                    saved = json.loads((root / "accuracy_summary.json").read_text(encoding="utf-8"))
                    self.assertTrue(saved["require_finished"])
                    self.assertEqual(saved["ordered_exact_accuracy"], summary["ordered_exact_accuracy"])

    def test_finished_status_alone_does_not_pass_wrong_arguments_or_call_order(self):
        with workspace() as root:
            data, rows = corpus(root)
            a = {"name": "generic.read", "arguments": {"value": 1}}
            b = {"name": "generic.write", "arguments": {}}
            rows[0]["label"]["tool_calls"] = [a, b]
            rows[1]["label"]["tool_calls"] = [a]
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            records = [
                {"id": rows[0]["id"], "tool_calls": [b, a], "state_hash": "h1"},
                {"id": rows[1]["id"], "tool_calls": [{**a, "arguments": {"value": "1"}}], "state_hash": "h2"},
            ]
            path = predictions_file(root, records)
            traces = root / "traces"; traces.mkdir()
            for i, record in enumerate(records):
                (traces / f"{i}.json").write_text(json.dumps({
                    "submission": record, "status": "finished", "trace": [],
                }), encoding="utf-8")
            summary, _ = train_eval.evaluate_submission(data, path, count=2, trace_dir=traces, require_finished=True)
            self.assertEqual(summary["finished_samples"], 2)
            self.assertEqual(summary["ordered_exact_accuracy"], 0)
            self.assertEqual(summary["unordered_exact_accuracy"], 0.5)

    def test_missing_status_fails_closed_and_keeps_all_selected_samples_in_denominator(self):
        with workspace() as root:
            data, rows = corpus(root)
            records = [{"id": row["id"], "tool_calls": [], "state_hash": "h"} for row in rows]
            path = predictions_file(root, records)
            traces = root / "traces"; traces.mkdir()
            (traces / "one.json").write_text(json.dumps({
                "submission": records[0], "status": "finished", "trace": [],
            }), encoding="utf-8")
            summary, details = train_eval.evaluate_submission(data, path, count=3, trace_dir=traces, require_finished=True)
            self.assertEqual(summary["ordered_exact_accuracy"], 1 / 3)
            self.assertEqual(summary["call_only_ordered_exact_accuracy"], 1)
            self.assertEqual(summary["unknown_status_samples"], 2)
            self.assertFalse(details[1]["agent_finished"])
            self.assertFalse(details[1]["ordered_exact_match"])
            self.assertTrue(details[1]["call_only_ordered_exact_match"])

    def test_finished_gate_requires_trace_source_and_never_guesses_status_from_calls(self):
        with workspace() as root:
            data, rows = corpus(root)
            path = predictions_file(root, [{"id": rows[0]["id"], "tool_calls": [], "state_hash": "h"}])
            with self.assertRaisesRegex(ValueError, "needs trace_dir"):
                train_eval.evaluate_submission(data, path, count=1, require_finished=True)
            self.assertFalse((root / "accuracy_summary.json").exists())
            summary, details = train_eval.evaluate_submission(data, path, count=1)
            self.assertFalse(summary["require_finished"])
            self.assertEqual(summary["ordered_exact_accuracy"], 1)
            self.assertIsNone(summary["finished_rate"])
            self.assertFalse(details[0]["agent_finished"])

    def test_duplicate_and_outside_subset_prediction_ids_are_errors(self):
        with workspace() as root:
            data, _ = corpus(root)
            record = {"id": "train_0001", "tool_calls": [], "state_hash": "x"}
            for records in ([record, record], [{**record, "id": "train_0002"}]):
                with self.subTest(records=records):
                    path = predictions_file(root, records)
                    with self.assertRaisesRegex(ValueError, "duplicate prediction ID"):
                        train_eval.evaluate_submission(data, path, count=2)

    def test_unordered_accuracy_respects_duplicate_call_counts_and_argument_values(self):
        with workspace() as root:
            data, rows = corpus(root)
            call = {"name": "calendar.a", "arguments": {"count": 1}}
            rows[0]["label"]["tool_calls"] = [call, call]
            rows[1]["label"]["tool_calls"] = [call]
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            path = predictions_file(root, [
                {"id": "train_0001", "tool_calls": [call], "state_hash": "x"},
                {"id": "train_0226", "tool_calls": [{"name": "calendar.a", "arguments": {"count": "1"}}], "state_hash": "x"},
            ])
            summary, _ = train_eval.evaluate_submission(data, path, count=2)
            self.assertEqual(summary["ordered_exact_accuracy"], 0)
            self.assertEqual(summary["unordered_exact_accuracy"], 0)

    def test_call_metrics_separate_wrong_arguments_missing_calls_and_extra_tools(self):
        with workspace() as root:
            data, rows = corpus(root)
            same = {"name": "inventory.lookup", "arguments": {"query": "part"}}
            write = {"name": "inventory.reserve", "arguments": {"amount": 2}}
            rows[0]["label"]["tool_calls"] = [same, same, write]
            rows[1]["label"]["tool_calls"] = []
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            path = predictions_file(root, [
                {"id": rows[0]["id"], "tool_calls": [same, {**write, "arguments": {"amount": "2"}},
                    {"name": "other.send", "arguments": {}}], "state_hash": "a"},
                {"id": rows[1]["id"], "tool_calls": [], "state_hash": "b"},
            ])
            summary, details = train_eval.evaluate_submission(data, path, count=2)
            names, calls = summary["tool_name_metrics"], summary["exact_call_metrics"]
            self.assertEqual((names["matched"], names["gold"], names["predicted"]), (2, 3, 3))
            self.assertAlmostEqual(names["f1"], 2 / 3)
            self.assertEqual((calls["matched"], calls["missing"], calls["extra"]), (1, 2, 2))
            self.assertAlmostEqual(calls["precision"], 1 / 3)
            self.assertEqual(summary["argument_mismatch_calls"], 1)
            self.assertEqual(details[0]["missing_tool_name_calls"], 1)
            self.assertEqual(details[0]["extra_tool_name_calls"], 1)
            self.assertEqual(summary["ordered_exact_accuracy"], 0.5)

    def test_name_order_and_exact_calls_distinguish_order_from_arguments(self):
        with workspace() as root:
            data, rows = corpus(root)
            first = {"name": "reader.find", "arguments": {"text": "A"}}
            second = {"name": "writer.change", "arguments": {"value": 3}}
            rows[0]["label"]["tool_calls"] = [first, second]
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            path = predictions_file(root, [{"id": rows[0]["id"], "tool_calls": [second, first], "state_hash": "h"}])
            summary, _ = train_eval.evaluate_submission(data, path, count=1)
            self.assertEqual(summary["tool_name_metrics"]["f1"], 1)
            self.assertEqual(summary["exact_call_metrics"]["f1"], 1)
            self.assertEqual(summary["unordered_exact_accuracy"], 1)
            self.assertEqual(summary["ordered_tool_name_accuracy"], 0)
            self.assertEqual(summary["ordered_exact_accuracy"], 0)

    def test_explicit_subset_uses_requested_order_across_oracle_prepare_and_evaluation(self):
        with workspace() as root:
            data, rows = corpus(root)
            ids = [rows[2]["id"], rows[0]["id"]]
            speech = train_eval.create_oracle_asr(data, root / "oracle.jsonl", sample_ids=ids)
            prepared = train_eval.prepare_asr_file(data, speech, root / "prepared", sample_ids=ids)
            self.assertEqual(prepared["selected_ids"], ids)
            self.assertEqual(prepared["count"], 2)
            self.assertEqual([row["id"] for row in read_jsonl(prepared["asr_file"])], ids)
            path = predictions_file(root, [{"id": cid, "tool_calls": [], "state_hash": "h"} for cid in reversed(ids)])
            summary, details = train_eval.evaluate_submission(data, path, sample_ids=ids)
            self.assertEqual(summary["expected_samples"], 2)
            self.assertEqual(summary["selected_ids"], ids)
            self.assertEqual([row["id"] for row in details], ids)
            self.assertEqual(summary["ordered_exact_accuracy"], 1)
            self.assertEqual(summary["exact_call_metrics"]["f1"], 1)

    def test_explicit_subset_rejects_unknown_empty_and_duplicate_ids(self):
        with workspace() as root:
            data, rows = corpus(root)
            for ids in ([], [rows[0]["id"]] * 2, ["does-not-exist"], [None], rows[0]["id"]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    train_eval.select_train_items(data, sample_ids=ids)

    def test_trace_diagnostics_measure_runtime_and_format_failures_without_claiming_semantics(self):
        with workspace() as root:
            data, rows = corpus(root)
            call = {"name": "generic.change", "arguments": {}}
            rows[0]["label"]["tool_calls"] = [call, {"name": "generic.notify", "arguments": {}}]
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            record = {"id": rows[0]["id"], "tool_calls": [call], "state_hash": "h"}
            path = predictions_file(root, [record])
            traces = root / "traces"; traces.mkdir()
            (traces / "one.json").write_text(json.dumps({"submission": record, "status": "finished", "trace": [
                {"type": "model_reply", "truncated": True},
                {"type": "rejected", "error": "bad JSON"},
                {"type": "tool_result", "result": {"ok": True, "done": False, "message": "not found"}},
                {"type": "tool_result", "result": {"ok": True, "items": []}},
                {"type": "tool_result", "result": {"ok": False, "error": "schema"}},
            ]}), encoding="utf-8")
            summary, details = train_eval.evaluate_submission(data, path, count=2, trace_dir=traces)
            diag = summary["trace_diagnostics"]
            self.assertEqual(diag["statuses"], {"finished": 1})
            self.assertEqual(diag["missing_traces"], 1)
            for key in ("rejected_events", "truncated_replies", "runtime_noop_results", "runtime_error_results",
                        "empty_list_results", "early_finish_candidates"):
                self.assertEqual(diag[key], 1, key)
            self.assertTrue(details[0]["early_finish_candidate"])
            self.assertIsNone(details[1]["early_finish_candidate"])
            self.assertIn("Heuristic", diag["early_finish_note"])
            self.assertIsNone(summary["state_hash_accuracy"])

    def test_stale_trace_is_rejected_before_evaluation_reports_are_written(self):
        with workspace() as root:
            data, rows = corpus(root)
            record = {"id": rows[0]["id"], "tool_calls": [], "state_hash": "current"}
            path = predictions_file(root, [record])
            traces = root / "traces"; traces.mkdir()
            trace = {"submission": {**record, "state_hash": "old"}, "status": "finished", "trace": []}
            (traces / "one.json").write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_eval.evaluate_submission(data, path, count=1, trace_dir=traces)
            self.assertFalse((root / "accuracy_summary.json").exists())

    def test_real_gold_hash_is_compared_only_when_present(self):
        with workspace() as root:
            data, rows = corpus(root)
            rows[0]["label"]["state_hash"] = "official-comparable-hash"
            (data / "train/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            path = predictions_file(root, [{"id": rows[0]["id"], "tool_calls": [], "state_hash": "official-comparable-hash"}])
            summary, _ = train_eval.evaluate_submission(data, path, count=2)
            self.assertEqual(summary["state_hash_comparable_samples"], 1)
            self.assertEqual(summary["state_hash_accuracy"], 1)

    def test_trace_ids_cannot_be_duplicated_or_silently_joined_to_an_unrelated_run(self):
        with workspace() as root:
            data, rows = corpus(root)
            record = {"id": rows[0]["id"], "tool_calls": [], "state_hash": "h"}
            path = predictions_file(root, [record])
            traces = root / "traces"; traces.mkdir()
            current = {"submission": record, "status": "finished", "trace": []}
            (traces / "one.json").write_text(json.dumps(current), encoding="utf-8")
            (traces / "unrelated.json").write_text(json.dumps({**current,
                "submission": {**record, "id": "not-selected"}}), encoding="utf-8")
            summary, _ = train_eval.evaluate_submission(data, path, count=1, trace_dir=traces)
            self.assertEqual(summary["trace_diagnostics"]["evaluated_traces"], 1)
            self.assertEqual(summary["trace_diagnostics"]["unrelated_trace_files"], 1)
            (traces / "duplicate.json").write_text(json.dumps(current), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate trace ID"):
                train_eval.evaluate_submission(data, path, count=1, trace_dir=traces)

    def test_trace_matching_preserves_json_argument_types(self):
        with workspace() as root:
            data, rows = corpus(root)
            call = {"name": "generic.change", "arguments": {"value": 1}}
            record = {"id": rows[0]["id"], "tool_calls": [call], "state_hash": "h"}
            path = predictions_file(root, [record])
            traces = root / "traces"; traces.mkdir()
            altered = {**record, "tool_calls": [{**call, "arguments": {"value": True}}]}
            (traces / "one.json").write_text(json.dumps({"submission": altered,
                "status": "finished", "trace": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_eval.evaluate_submission(data, path, count=1, trace_dir=traces)

    def test_runner_forwards_agent_config_and_requested_subset_to_existing_pipeline(self):
        with workspace() as root:
            data, rows = corpus(root)
            ids = [rows[2]["id"]]
            speech = train_eval.create_oracle_asr(data, root / "oracle.jsonl", sample_ids=ids)
            records = [{"id": ids[0], "tool_calls": [], "state_hash": "h"}]
            config = object()
            with patch.object(agent, "run_full_pipeline", return_value={"records": records,
                    "submission_path": str(root / "debug_train.jsonl")}) as run:
                result = train_eval.run_train_agent(data, speech, root / "out", sample_ids=ids, agent_config=config)
            self.assertEqual(run.call_args.kwargs["sample_ids"], ids)
            self.assertIs(run.call_args.kwargs["agent_config"], config)
            self.assertTrue(Path(result["test_submition_path"]).is_file())

    def test_notebook_embeds_helpers_and_reuses_the_explicit_qwen_variable(self):
        nb = json.loads((REPO / "notebooks/lifemate/test_lifemate.ipynb").read_text(encoding="utf-8"))
        embedded, inference_found = None, False
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            tree = ast.parse("".join(cell["source"]))
            compile(tree, f"notebook:{index}", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAIN_EVAL_SOURCE" for t in node.targets):
                    embedded = ast.literal_eval(node.value)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run_train_agent":
                    inference_found = True
                    backend = next(kw.value for kw in node.keywords if kw.arg == "backend")
                    self.assertIsInstance(backend, ast.Name)
                    self.assertEqual(backend.id, "qwen_backend")
        self.assertEqual(embedded, (REPO / "src/lifemate/train_eval.py").read_text(encoding="utf-8"))
        self.assertTrue(inference_found)


class KaggleEnvironmentTests(unittest.TestCase):
    def execute_environment_cell(self, root, probe_returncode=0, backend_loaded=False):
        nb = json.loads((REPO / "notebooks/lifemate/test_lifemate.ipynb").read_text(encoding="utf-8"))
        source = "".join(nb["cells"][8]["source"])
        namespace = {
            "os": os, "sys": SimpleNamespace(executable="/usr/bin/python3"),
            "subprocess": SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=probe_returncode))),
            "textwrap": textwrap, "PROJECT_ROOT": REPO, "SCRATCH_ROOT": root,
            "HF_TOKEN_SECRET_NAME": None,
            "checked": Mock(),
        }
        if backend_loaded:
            namespace["qwen_backend"] = object()
        fake_torch = SimpleNamespace(__version__="2.6.0", cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda index: "test-GPU"))
        with patch.dict("sys.modules", {"torch": fake_torch,
                        "transformers": SimpleNamespace(__version__="5.17.0")}), \
             patch.dict(os.environ), redirect_stdout(io.StringIO()):
            exec(compile(source, "kaggle-environment-cell", "exec"), namespace)
        return namespace

    def test_pip_installs_into_current_kernel_without_creating_a_venv(self):
        with workspace() as root, patch("urllib.request.urlretrieve") as download:
            namespace = self.execute_environment_cell(root)
            commands = [call.args[0] for call in namespace["checked"].call_args_list]
            self.assertFalse(any("ensurepip" in command for command in commands))
            self.assertFalse(any("venv" in command for command in commands))
            self.assertEqual(commands[0][:3], ["/usr/bin/python3", "-m", "pip"])
            self.assertNotIn("--python", commands[0])
            download.assert_not_called()

    def test_missing_kernel_pip_uses_official_zipapp(self):
        with workspace() as root:
            def download_zip(url, filename):
                self.assertEqual(url, "https://bootstrap.pypa.io/pip/pip.pyz")
                with zipfile.ZipFile(filename, "w") as archive:
                    archive.writestr("__main__.py", "# offline test fixture")
            with patch("urllib.request.urlretrieve", side_effect=download_zip) as download:
                namespace = self.execute_environment_cell(root, probe_returncode=1)
            download.assert_called_once()
            self.assertTrue(zipfile.is_zipfile(root / "pip.pyz"))
            install = namespace["checked"].call_args_list[0].args[0]
            self.assertEqual(install[:2], ["/usr/bin/python3", str(root / "pip.pyz")])
            self.assertNotIn("--python", install)

    def test_valid_cached_pip_zipapp_does_not_download_again(self):
        with workspace() as root, patch("urllib.request.urlretrieve") as download:
            with zipfile.ZipFile(root / "pip.pyz", "w") as archive:
                archive.writestr("__main__.py", "# offline test fixture")
            self.execute_environment_cell(root, probe_returncode=1)
            download.assert_not_called()

    def test_incomplete_download_stops_before_dependency_install(self):
        with workspace() as root:
            def broken_download(url, filename):
                Path(filename).write_bytes(b"incomplete")
            with patch("urllib.request.urlretrieve", side_effect=broken_download):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    self.execute_environment_cell(root, probe_returncode=1)

    def test_environment_cell_does_not_reinstall_packages_with_a_loaded_model(self):
        with workspace() as root:
            namespace = self.execute_environment_cell(root, backend_loaded=True)
            namespace["checked"].assert_not_called()


class DirectQwenCellTests(unittest.TestCase):
    def notebook(self):
        return json.loads((REPO / "notebooks/lifemate/test_lifemate.ipynb").read_text(encoding="utf-8"))

    def test_default_is_ten_and_load_cell_only_loads_once(self):
        nb = self.notebook()
        namespace = {}
        exec("".join(nb["cells"][2]["source"]), namespace)
        self.assertEqual(namespace["SAMPLE_COUNT"], 10)
        with workspace() as root:
            model = root / "model"
            model.mkdir()
            namespace.update({"MODEL_SOURCE": str(model), "SCRATCH_ROOT": root})
            backend = object()
            with patch.object(agent, "load_qwen_backend", return_value=backend) as load, redirect_stdout(io.StringIO()):
                exec("".join(nb["cells"][10]["source"]), namespace)
                exec("".join(nb["cells"][10]["source"]), namespace)
            load.assert_called_once()
            self.assertIs(namespace["qwen_backend"], backend)

    def test_two_agent_cell_runs_reuse_model_and_create_fresh_environments(self):
        with workspace() as root:
            data, _ = corpus(root)
            speech = train_eval.create_oracle_asr(data, root / "oracle.jsonl", 2)
            created = []

            class Env:
                def __init__(self, split, cid, root):
                    created.append(cid)
                def get_tool_calls(self):
                    return []
                def get_state_hash(self):
                    return "test-hash"

            class Backend:
                def __init__(self):
                    self.generations = 0
                def generate(self, *args, **kwargs):
                    self.generations += 1
                    return agent.ModelReply("<final>Done</final>")

            backend = Backend()
            namespace = {"Path": Path, "datetime": datetime, "timezone": timezone, "uuid": uuid,
                "json": json, "train_eval": train_eval, "qwen_backend": backend,
                "AGENT_OPTIONS": {"workflow_enabled": False},
                "qwen_model_config": agent.ModelLoadConfig(model_source="local-test-model"),
                "loaded_qwen_key": ("local-test-model", "cuda:0"), "MODEL_SOURCE": "local-test-model",
                "DEVICE": "cuda:0", "SAMPLE_COUNT": 2, "DATA_ROOT": data, "RUN_ROOT": root / "runs",
                "prepared": {"asr_file": str(speech)}, "ASR_MODE": "oracle_train_audio_text", "source_commit": "test-commit"}
            source = "".join(self.notebook()["cells"][12]["source"])
            with patch.object(agent, "get_tool_env_factory", return_value=Env), \
                 patch.object(agent, "load_qwen_backend", side_effect=AssertionError("Must not reload Qwen")) as load, \
                 redirect_stdout(io.StringIO()):
                exec(source, namespace)
                first_report = namespace["LAST_AGENT_REPORT_FILE"]
                exec(source, namespace)
                second_report = namespace["LAST_AGENT_REPORT_FILE"]
            load.assert_not_called()
            self.assertIs(namespace["qwen_backend"], backend)
            self.assertEqual(backend.generations, 4)
            self.assertEqual(created, ["train_0001", "train_0226", "train_0001", "train_0226"])
            self.assertNotEqual(first_report, second_report)
            self.assertTrue(first_report.is_file() and second_report.is_file())

    def test_agent_cell_rejects_a_missing_model_without_background_processes(self):
        source = "".join(self.notebook()["cells"][12]["source"])
        self.assertNotIn("subprocess", source)
        self.assertNotIn("load_qwen_backend", source)
        with self.assertRaisesRegex(RuntimeError, "cell 5"):
            exec(source, {})


if __name__ == "__main__":
    unittest.main()
