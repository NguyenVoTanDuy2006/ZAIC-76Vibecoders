"""Agent scoring invariants. Scripted/reference checks are not model accuracy."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import unicodedata
import unittest
from unittest.mock import patch
import uuid
import wave

from lifemate.agent import AgentConfig, ModelReply, SYSTEM_PROMPT
from lifemate.evaluation import data
from lifemate.evaluation.compare import compare
from lifemate.evaluation.evaluate import evaluate
from lifemate.evaluation.io import read_json, read_rows, write_json, write_rows
from lifemate.evaluation.matching import MatchConvention, argument_differences, match_calls, maximum_matching, token_f1
from lifemate.evaluation.runner import run
from lifemate.evaluation.runtime import OfficialRuntime
from lifemate.evaluation.validate import read_predictions, validate_prediction


REPO = Path(__file__).resolve().parents[2]
CATALOG = {"apps": [
    {"id": "demo", "tools": [{"name": "demo.set", "parameters": {"type": "object", "required": ["value"],
        "properties": {"value": {"type": "integer"}, "note": {"type": "string", "x-match": "fuzzy"}}}}]},
    {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "parameters": {"type": "object",
        "properties": {"tool_name": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}}}}}]}]}


@contextmanager
def scratch():
    base = REPO / ".test-scratch"
    base.mkdir(exist_ok=True)
    root = base / uuid.uuid4().hex
    root.mkdir()
    try:
        yield root
    finally:
        assert root.resolve().parent == base.resolve()
        shutil.rmtree(root)
        if not any(base.iterdir()):
            base.rmdir()


def item(sid="a"):
    return {"id": sid, "context": {"installed_apps": {"demo": {"tools": ["demo.set"]}}}, "history": [], "audio": f"audio/{sid}.wav"}


def call(value=1, **extra):
    return {"name": "demo.set", "arguments": {"value": value, **extra}}


class FakeRuntime:
    catalog = CATALOG
    tools = {t["name"]: t for a in CATALOG["apps"] for t in a["tools"]}
    identity = {"physical_split": "train", "catalog_sha256": "fake"}

    @staticmethod
    def schema_error(name, args, tools, allowed):
        if name not in tools or name not in allowed:
            return "tool not available"
        if name == "demo.set" and type(args.get("value")) is not int:
            return "value must be integer"
        return None

    def replay(self, sid, calls):
        value, results = 0, []
        for c in calls:
            error = self.schema_error(c["name"], c["arguments"], self.tools, self.tools)
            if error:
                results.append({"ok": False, "error": error})
            else:
                if c["name"] == "demo.set":
                    value = c["arguments"]["value"]
                results.append({"ok": True})
        state = {"demo": {"value": value}}
        digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        return {"tool_calls": deepcopy(calls), "state_hash": digest, "state": state, "results": results}


def prediction(runtime, sid="a", calls=None):
    calls = calls if calls is not None else [call()]
    replay = runtime.replay(sid, calls)
    return {"id": sid, "tool_calls": calls, "state_hash": replay["state_hash"]}


def gold(runtime, sid="a", calls=None):
    result = prediction(runtime, sid, calls)
    return {**result, "reference_valid": True, "group_id": sid,
            "canonical_state": runtime.replay(sid, result["tool_calls"])["state"]}


def dataset(root):
    rows = []
    for index in range(20):
        row = item(f"s{index:02}")
        row["audio_text"] = f"Set the unique value for sample {chr(65 + index)} please"
        row["label"] = {"tool_calls": [call(index + 1)], "tool_results": [{"ok": True}], "final_state": {"demo": {"value": index + 1}}}
        path = root / "train" / row["audio"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes((index + 1).to_bytes(2, "little") * 80)
        rows.append(row)
    # Shared HISTORY audio is grouped with the sample owning that audio currently.
    rows[1]["history"] = [{"role": "assistant", "text": "Earlier context"},
        {"role": "user", "audio": rows[0]["audio"], "audio_text": rows[0]["audio_text"]}]
    write_rows(root / "train/train.jsonl", rows)
    write_json(root / "train/train_tools_catalog.json", CATALOG)
    return rows


class MatchingTests(unittest.TestCase):
    def test_maximum_matching_uses_augmenting_paths(self):
        pairs = maximum_matching([0, 1], [0, 1], lambda a, b: a == 0 or b == 0)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})

    def test_fuzzy_unicode_and_strict_threshold(self):
        self.assertEqual(token_f1("HẸN cà phê", unicodedata.normalize("NFD", "hẹn cà phê")), 1)
        self.assertEqual(token_f1("a b c d e", "a b c x y"), .6)
        tools = {"demo.set": FakeRuntime.tools["demo.set"]}
        self.assertEqual(match_calls([call(note="a b c d e")], [call(note="a b c x y")], tools), [])
        self.assertEqual(len(match_calls([call(note="a b c d e")], [call(note="a b c d x")], tools)), 1)

    def test_duplicates_can_only_match_once(self):
        self.assertEqual(len(match_calls([call(), call()], [call()], FakeRuntime.tools)), 1)

    def test_exact_types_extra_keys_and_nested_values(self):
        self.assertTrue(argument_differences({"a": True}, {"a": 1}, {}))
        self.assertTrue(argument_differences({"a": 1, "b": ""}, {"a": 1}, {}))
        self.assertTrue(argument_differences({"a": [{"b": 2}]}, {"a": [{"b": 3}]}, {}))


class EvaluationTests(unittest.TestCase):
    def test_malformed_call_keeps_raw_count_but_scores_zero(self):
        runtime = FakeRuntime()
        with scratch() as root:
            pred = prediction(runtime)
            pred["tool_calls"] = [None, {"name": "demo.set", "arguments": "not an object"}]
            write_rows(root / "pred.jsonl", [pred])
            metrics = evaluate([item()], [gold(runtime)], root / "pred.jsonl", runtime, root / "out")
            report = read_rows(root / "out/samples.jsonl")[0]
            self.assertEqual(metrics["score"], 0)
            self.assertEqual(report["K"], 2)

    def test_reference_calls_score_100_but_wrong_hash_is_zero(self):
        runtime = FakeRuntime()
        with scratch() as root:
            path = root / "pred.jsonl"
            write_rows(path, [prediction(runtime)])
            metrics = evaluate([item()], [gold(runtime)], path, runtime, root / "good")
            self.assertEqual(metrics["score"], 100)
            bad = prediction(runtime); bad["state_hash"] = "0" * 64
            path2 = root / "bad.jsonl"; write_rows(path2, [bad])
            metrics = evaluate([item()], [gold(runtime)], path2, runtime, root / "bad")
            self.assertEqual(metrics["score"], 0)
            self.assertEqual(metrics["loss_decomposition"]["L_invalid"], 100)

    def test_final_state_mismatch_cannot_be_saved_by_partial_matching(self):
        runtime = FakeRuntime()
        with scratch() as root:
            path = root / "pred.jsonl"
            write_rows(path, [prediction(runtime, calls=[call(1), call(2)])])
            metrics = evaluate([item()], [gold(runtime)], path, runtime, root / "out")
            self.assertEqual(metrics["score"], 0)
            self.assertEqual(metrics["loss_decomposition"]["L_hash"], 100)
            report = read_rows(root / "out/samples.jsonl")[0]
            self.assertEqual(report["state_differences"][0]["path"], "state.demo.value")

    def test_sample_average_not_micro_and_loss_decomposition(self):
        runtime = FakeRuntime()
        targets = [gold(runtime, "a", [call(), call(), call()]), gold(runtime, "b", [])]
        preds = [prediction(runtime, "a", [call()]), prediction(runtime, "b", [])]
        with scratch() as root:
            write_rows(root / "pred.jsonl", preds)
            metrics = evaluate([item("a"), item("b")], targets, root / "pred.jsonl", runtime, root / "out")
            self.assertEqual(metrics["score"], 75)
            self.assertEqual(metrics["call_match_micro_diagnostic_only"]["f1"], .5)
            self.assertAlmostEqual(sum(metrics["loss_decomposition"].values()), 100 - metrics["score"])

    def test_timeout_submission_is_scored_but_missing_prediction_stays_zero(self):
        runtime = FakeRuntime()
        pred = prediction(runtime, calls=[])
        traces = {"a": {"status": "timeout", "submission": pred}}
        with scratch() as root:
            write_rows(root / "pred.jsonl", [pred])
            metrics = evaluate([item("a"), item("b")], [gold(runtime, "a", []), gold(runtime, "b", [])],
                root / "pred.jsonl", runtime, root / "out", traces=traces)
            self.assertEqual(metrics["score"], 50)
            self.assertEqual(metrics["counts"]["samples"], 2)
            self.assertEqual(metrics["runtime_health"]["unhealthy_samples"], 2)
            self.assertFalse(metrics["strict_validation_passed"])

    def test_step_limit_correct_calls_and_hash_score_independently_of_runtime(self):
        runtime = FakeRuntime()
        pred = prediction(runtime)
        traces = {"a": {"status": "step_limit", "submission": pred}}
        checked = validate_prediction("a", pred, item(), runtime, traces)
        self.assertTrue(checked["submission_valid"])
        self.assertFalse(checked["runtime_valid"])
        self.assertFalse(checked["valid"])  # Strict validation still reports unhealthy runs.
        self.assertEqual(checked["submission_errors"], [])
        self.assertEqual(checked["runtime_errors"][0]["kind"], "runtime_failure")
        with scratch() as root:
            write_rows(root / "pred.jsonl", [pred])
            metrics = evaluate([item()], [gold(runtime)], root / "pred.jsonl", runtime,
                               root / "out", traces=traces)
            self.assertEqual(metrics["score"], 100)
            self.assertTrue(metrics["submission_file_valid"])
            self.assertFalse(metrics["strict_validation_passed"])
            self.assertEqual(metrics["runtime_health"]["statuses"], {"step_limit": 1})
            report = read_rows(root / "out/samples.jsonl")[0]
            self.assertTrue(report["submission_valid"])
            self.assertFalse(report["runtime_valid"])

    def test_missing_trace_is_diagnostic_but_trace_submission_mismatch_is_invalid(self):
        runtime = FakeRuntime()
        pred = prediction(runtime)
        checked = validate_prediction("a", pred, item(), runtime, {})
        self.assertTrue(checked["submission_valid"])
        self.assertFalse(checked["runtime_valid"])
        self.assertFalse(checked["valid"])
        self.assertEqual(checked["runtime_errors"], [{"kind": "missing_trace"}])
        with scratch() as root:
            write_rows(root / "pred.jsonl", [pred])
            metrics = evaluate([item()], [gold(runtime)], root / "pred.jsonl", runtime,
                               root / "missing", traces={})
            self.assertEqual(metrics["score"], 100)
            mismatched = {"a": {"status": "finished", "submission": prediction(runtime, calls=[])}}
            metrics = evaluate([item()], [gold(runtime)], root / "pred.jsonl", runtime,
                               root / "mismatched", traces=mismatched)
            self.assertEqual(metrics["score"], 0)
            details = read_rows(root / "mismatched/samples.jsonl")[0]
            self.assertEqual(details["submission_errors"][0]["kind"], "trace_submission_mismatch")

    def test_finished_with_unreproducible_hash_still_scores_zero(self):
        runtime = FakeRuntime()
        pred = prediction(runtime)
        pred["state_hash"] = "0" * 64
        traces = {"a": {"status": "finished", "submission": pred}}
        with scratch() as root:
            write_rows(root / "pred.jsonl", [pred])
            metrics = evaluate([item()], [gold(runtime)], root / "pred.jsonl", runtime,
                               root / "out", traces=traces)
            self.assertEqual(metrics["score"], 0)
            self.assertEqual(metrics["runtime_health"]["healthy_samples"], 1)
            self.assertFalse(metrics["submission_file_valid"])

    def test_duplicate_foreign_malformed_and_duplicate_json_keys(self):
        runtime = FakeRuntime()
        pred = prediction(runtime)
        with scratch() as root:
            path = root / "pred.jsonl"
            path.write_text(json.dumps(pred) + "\n" + json.dumps(pred) + '\n{"id":"a","id":"b"}\n[]\n' +
                            json.dumps(prediction(runtime, "foreign")) + "\n", encoding="utf-8")
            rows, issues = read_predictions(path, ["a"])
            self.assertEqual(rows["a"]["_invalid"], "duplicate_prediction_id")
            self.assertEqual({i["kind"] for i in issues}, {"duplicate_id", "invalid_json", "foreign_id"})

    def test_rejected_call_is_preserved_in_denominator(self):
        runtime = FakeRuntime()
        invalid = {"name": "demo.set", "arguments": {"value": "wrong"}}
        with scratch() as root:
            write_rows(root / "pred.jsonl", [prediction(runtime, calls=[invalid, call()])])
            metrics = evaluate([item()], [gold(runtime)], root / "pred.jsonl", runtime, root / "out")
            self.assertAlmostEqual(metrics["score"], 200/3)
            self.assertEqual(metrics["counts"]["schema_errors"], 1)
            self.assertEqual(read_rows(root / "out/samples.jsonl")[0]["K"], 2)

    def test_missing_info_requires_correct_tool_name_and_params(self):
        runtime = FakeRuntime()
        a = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "demo.set", "params": ["value"]}}
        b = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "demo.set", "params": ["other"]}}
        with scratch() as root:
            write_rows(root / "pred.jsonl", [prediction(runtime, calls=[b])])
            metrics = evaluate([item()], [gold(runtime, calls=[a])], root / "pred.jsonl", runtime, root / "out")
            self.assertEqual(metrics["score"], 0)
            self.assertEqual(metrics["missing_information_micro"]["f1"], 0)


class DataTests(unittest.TestCase):
    def test_audio_cannot_reference_environment_files(self):
        with scratch() as root:
            rows = dataset(root)
            rows[0]["audio"] = "train_environment/s00/state.json"
            (root / "train/train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "never an environment"):
                data.prepare(root, root / "bundle")

    def test_allowlist_removes_hidden_state_and_gold_fields(self):
        row = item()
        row.update(audio_text="SECRET", label={"final_state": "SECRET"})
        row["context"]["state"] = "SECRET"
        row["context"]["installed_apps"]["demo"]["state"] = "SECRET"
        self.assertNotIn("SECRET", json.dumps(data.agent_item(row)))

    def test_numeric_conversation_templates_are_grouped(self):
        with scratch() as root:
            rows = dataset(root)
            rows[2]["audio_text"] = "Set the alarm at 8 tonight please"
            rows[3]["audio_text"] = "Set the alarm at 9 tonight please"
            source = root / "train/train.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            data.prepare(root, root / "bundle")
            membership = {sid: split for split in ("train", "validation", "local_test")
                for sid in read_json(root / "bundle" / split / "manifest.json")["ids"]}
            self.assertEqual(membership["s02"], membership["s03"])

    def test_split_exact_grouped_reproducible_and_no_gold_in_inputs(self):
        with scratch() as root:
            dataset(root)
            first = data.prepare(root, root / "bundle")
            data.prepare(root, root / "again")
            self.assertEqual([first["splits"][k]["samples"] for k in ("train", "validation", "local_test")], [16, 2, 2])
            membership = {}
            for split in ("train", "validation", "local_test"):
                manifest = read_json(root / "bundle" / split / "manifest.json")
                self.assertEqual(manifest, read_json(root / "again" / split / "manifest.json"))
                membership.update({sid: split for sid in manifest["ids"]})
                for row in read_rows(root / "bundle" / split / "inputs.jsonl"):
                    self.assertNotIn("label", row)
                    self.assertNotIn("audio_text", row)
                    self.assertTrue(all("audio_text" not in t for t in row["history"]))
            self.assertEqual(membership["s00"], membership["s01"])
            self.assertTrue(data.verify(root / "bundle", root)["verified"])
            audio = root / "train/audio/s00.wav"
            audio.write_bytes(audio.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Audio fingerprint"):
                data.verify(root / "bundle", root)

    def test_test_gate_freeze_and_audit_and_changed_artifacts(self):
        with scratch() as root:
            dataset(root); data.prepare(root, root / "bundle")
            with self.assertRaisesRegex(ValueError, "Local test requires"):
                data.access_split(root / "bundle", "local_test")
            config = {"model_source": "local", "model_revision": "fixed", "weights_fingerprint": "sha256:fixed",
                "quantization": "fp16", "generation": {}, "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
            write_json(root / "config.json", config)
            data.freeze(root / "bundle", root / "config.json", root / "frozen.json")
            data.access_split(root / "bundle", "local_test", allow_local_test=True, frozen=root / "frozen.json")
            self.assertEqual(len(read_rows(root / "bundle/audit.jsonl")), 1)
            inputs = root / "bundle/local_test/inputs.jsonl"
            inputs.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                data.access_split(root / "bundle", "local_test", allow_local_test=True, frozen=root / "frozen.json")

    def test_paired_compare_requires_same_ids_and_keeps_groups(self):
        with scratch() as root:
            write_rows(root / "a.jsonl", [{"id": "a", "group_id": "g", "score": 0}, {"id": "b", "group_id": "g", "score": .5}])
            write_rows(root / "b.jsonl", [{"id": "a", "group_id": "g", "score": 1}, {"id": "b", "group_id": "g", "score": .5}])
            report = compare(root / "a.jsonl", root / "b.jsonl", root / "compare.json", iterations=50)
            self.assertEqual(report["delta_points"], 50)
            self.assertEqual(report["ci95_points"], [50, 50])


class OfficialTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Set LIFEMATE_DATA_ROOT for BTC runtime integration")
    def test_reference_replay_and_hash_verification_on_real_train_sample(self):
        root = Path(os.environ["LIFEMATE_DATA_ROOT"])
        runtime = OfficialRuntime(root)
        row = read_rows(root / "train/train.jsonl")[0]
        reference = runtime.reference({"id": row["id"], "label": row["label"]})
        self.assertTrue(reference["reference_valid"])
        pred = {"id": row["id"], "tool_calls": reference["tool_calls"], "state_hash": reference["state_hash"]}
        self.assertTrue(validate_prediction(row["id"], pred, row, runtime)["valid"])
        pred["state_hash"] = "f" * 64
        self.assertFalse(validate_prediction(row["id"], pred, row, runtime)["valid"])


class RunnerTests(unittest.TestCase):
    def test_missing_speech_keeps_failure_and_never_creates_or_retries_env(self):
        with scratch() as root:
            dataset(root); data.prepare(root, root / "bundle")
            manifest = read_json(root / "bundle/validation/manifest.json")
            selected = read_rows(root / "bundle/validation/inputs.jsonl")
            runtime = FakeRuntime()
            runtime.identity = {"catalog_sha256": read_json(root / "bundle/metadata.json")["catalog_sha256"]}
            sid = manifest["ids"][0]
            row = selected[0]
            speech = {"id": sid, "current_text": "ASR_REQUEST", "history_user_texts": [
                {"history_index": i, "text": "ASR_HISTORY"} for i, t in enumerate(row["history"]) if t["role"] == "user"]}
            write_rows(root / "speech.jsonl", [speech])
            created = []
            class Env:
                def get_tool_calls(self): return []
                def get_state_hash(self): return runtime.replay(sid, [])["state_hash"]
            def make_env(sample_id):
                created.append(sample_id)
                return Env()
            class Backend:
                def generate(self, messages, tools, config, remaining_seconds=None):
                    self.messages = messages
                    return ModelReply("<final>done</final>")
            backend = Backend()
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime), \
                 patch("lifemate.evaluation.runner.load_toolenv", return_value=make_env):
                result = run(root / "bundle", "validation", root, root / "speech.jsonl", root / "run", backend=backend)
            self.assertEqual(created, [sid])
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["successful"], 1)
            self.assertEqual(len(read_rows(root / "run/predictions.jsonl")), 1)
            results = read_rows(root / "run/results.jsonl")
            self.assertEqual(results[1]["status"], "input_or_runtime_error")
            self.assertIsNone(results[1]["submission"])
            self.assertNotIn("Set the unique", json.dumps(backend.messages))

    def test_oracle_is_not_available_on_local_test(self):
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            run("unused", "local_test", "unused", None, "unused", oracle=True)
