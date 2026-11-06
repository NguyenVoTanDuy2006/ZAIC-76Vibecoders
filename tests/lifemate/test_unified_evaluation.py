"""Shared Kaggle evaluation flow: scripted AgentSession and isolated SDK replay."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import Mock, patch

from lifemate.agent import ModelReply, SYSTEM_PROMPT, load_toolenv
from lifemate.evaluation import data
from lifemate.evaluation.experiment import prepare_experiment, run_experiment
from lifemate.evaluation.io import artifact_hashes, code_fingerprint, read_json, read_rows, sha256, write_json, write_rows
from lifemate.evaluation.runner import prepare_run
from lifemate.evaluation.runtime import OfficialRuntime
from lifemate.evaluation.selection import create_selection, select_rows

from test_evaluation import FakeRuntime, dataset, gold, scratch


OPTIONS = {"task_reasoning_enabled": False, "terminal_check_enabled": False,
           "grounding_enabled": False}


class ReplayRuntime(FakeRuntime):
    def __init__(self, catalog_hash):
        self.identity = {"physical_split": "train", "catalog_sha256": catalog_hash}
        self.reference_ids = []
        self.before_reference = None

    def reference(self, reference):
        if self.before_reference is not None:
            self.before_reference()
        self.reference_ids.append(reference["id"])
        return gold(self, reference["id"], reference["label"]["tool_calls"])


class ScriptedBackend:
    """Derive synthetic fixture values from speech, never from reference calls."""

    def __init__(self, *, fail_text=None):
        self.fail_text = fail_text
        self.finished = 0
        self.seen = []
        self.before_generate = None

    def generate(self, messages, tools, config, remaining_seconds=None):
        if self.before_generate is not None:
            self.before_generate(messages)
        self.seen.append(deepcopy(messages))
        if messages[-1]["role"] == "tool":
            self.finished += 1
            return ModelReply("<final>Synthetic fixture finished</final>")
        current = next(message["content"] for message in reversed(messages)
                       if message["role"] == "user")
        if current == self.fail_text:
            raise RuntimeError("Synthetic backend failure")
        match = re.search(r"sample ([A-T])", current)
        if match is None:
            raise AssertionError("Expected a synthetic speech request")
        value = ord(match.group(1)) - ord("A") + 1
        return ModelReply(json.dumps({"name": "demo.set", "arguments": {"value": value}}))


def env_loader(runtime, created):
    def load(*_):
        def make(sample_id):
            class Env:
                def __init__(self):
                    self.calls = []

                def get_tool_calls(self):
                    return deepcopy(self.calls)

                def get_state_hash(self):
                    return runtime.replay(sample_id, self.calls)["state_hash"]

                def validate(self, name, arguments):
                    error = runtime.schema_error(name, arguments, runtime.tools, runtime.tools)
                    return {"ok": error is None, "error": error}

                def call(self, name, arguments):
                    self.calls.append({"name": name, "arguments": deepcopy(arguments)})
                    return runtime.replay(sample_id, self.calls)["results"][-1]

            env = Env()
            created.append((sample_id, env))
            return env
        return make
    return load


class SelectionTests(unittest.TestCase):
    def test_subset_preserves_order_and_requires_matching_parent(self):
        with scratch() as root:
            dataset(root)
            data.prepare(root, root / "bundle")
            manifest = root / "bundle/validation/manifest.json"
            rows = read_rows(manifest.parent / "inputs.jsonl")
            selection = root / "selection.json"
            chosen = create_selection(root / "bundle", "validation", selection, limit=1)
            self.assertEqual([r["id"] for r in select_rows(rows, manifest, selection)], chosen["ids"])
            self.assertEqual(len(chosen["ids"]), 1)
            all_rows = create_selection(root / "bundle", "validation", root / "all.json")
            self.assertEqual(len(all_rows["ids"]), 2)
            # Explicit order is authoritative for both inference and evaluation.
            reverse = create_selection(root / "bundle", "validation", root / "reverse.json",
                                       sample_ids=list(reversed(all_rows["ids"])))
            self.assertEqual([r["id"] for r in select_rows(rows, manifest, root / "reverse.json")], reverse["ids"])
            changed = read_json(selection)
            changed["parent_manifest_sha256"] = "0" * 64
            write_json(root / "wrong-parent.json", changed)
            with self.assertRaisesRegex(ValueError, "parent manifest"):
                select_rows(rows, manifest, root / "wrong-parent.json")

    def test_duplicate_foreign_empty_and_invalid_subset_limits_fail(self):
        with scratch() as root:
            dataset(root)
            data.prepare(root, root / "bundle")
            ids = read_json(root / "bundle/validation/manifest.json")["ids"]
            for selection in ([ids[0], ids[0]], ["foreign"], []):
                with self.subTest(ids=selection), self.assertRaises(ValueError):
                    create_selection(root / "bundle", "validation", root / "unused.json", sample_ids=selection)
            for limit in (0, -1, True, 1.5):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    create_selection(root / "bundle", "validation", root / "unused.json", limit=limit)
            with self.assertRaisesRegex(ValueError, "full frozen manifest"):
                create_selection(root / "bundle", "local_test", root / "unused.json", limit=1)


class PreflightTests(unittest.TestCase):
    def test_empty_model_source_is_rejected_before_dataset_preparation_or_model(self):
        with scratch() as root:
            dataset(root)
            with patch("lifemate.evaluation.experiment.data.prepare") as prepare_data, \
                 patch("lifemate.evaluation.runner.load_qwen_backend") as load_model:
                with self.assertRaisesRegex(ValueError, "model_source"):
                    prepare_experiment(root, root / "work", model_source="")
            prepare_data.assert_not_called()
            load_model.assert_not_called()

    def test_reference_tampering_is_rejected_even_in_asr_text_mode(self):
        with scratch() as root:
            dataset(root)
            data.prepare(root, root / "bundle")
            rows = read_rows(root / "bundle/validation/inputs.jsonl")
            write_rows(root / "speech.jsonl", [{"id": row["id"], "current_text": "Synthetic speech",
                "history_user_texts": [{"history_index": i, "text": "Synthetic history"}
                    for i, turn in enumerate(row["history"]) if turn["role"] == "user"]} for row in rows])
            references = root / "bundle/validation/references.jsonl"
            altered = read_rows(references)
            altered[0]["label"]["tool_calls"] = []
            references.write_text("".join(json.dumps(row) + "\n" for row in altered), encoding="utf-8")
            with patch("lifemate.evaluation.runner.load_qwen_backend") as load_model, \
                 patch("lifemate.evaluation.runner.load_toolenv") as load_env:
                with self.assertRaisesRegex(ValueError, "Prepared artifact changed: validation/references.jsonl"):
                    prepare_run(root / "bundle", "validation", root, root / "speech.jsonl",
                                oracle=False, require_complete_speech=True)
            load_model.assert_not_called()
            load_env.assert_not_called()

    def test_missing_history_is_rejected_before_backend_or_inference_env(self):
        with scratch() as root:
            dataset(root)
            data.prepare(root, root / "bundle")
            path = root / "bundle/validation/inputs.jsonl"
            rows = read_rows(path)
            rows[0]["history"] = [{"role": "assistant", "text": "Earlier context"},
                                  {"role": "user", "audio": "audio/s00.wav"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            integrity = read_json(root / "bundle/integrity.json")
            integrity["validation/inputs.jsonl"] = sha256(path)
            (root / "bundle/integrity.json").write_text(json.dumps(integrity), encoding="utf-8")
            write_rows(root / "speech.jsonl", [{"id": row["id"], "current_text": "ASR_REQUEST",
                                               "history_user_texts": []} for row in rows])
            runtime = ReplayRuntime(read_json(root / "bundle/metadata.json")["catalog_sha256"])
            backend, constructor = Mock(), Mock()
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime), \
                 patch("lifemate.evaluation.runner.load_qwen_backend", backend), \
                 patch("lifemate.evaluation.runner.load_toolenv", constructor):
                with self.assertRaisesRegex(ValueError, "Incomplete current/history speech"):
                    prepare_run(root / "bundle", "validation", root, root / "speech.jsonl",
                                require_complete_speech=True)
            backend.assert_not_called()
            constructor.assert_not_called()


class ExperimentTests(unittest.TestCase):
    def run_fixture(self, root, *, count, backend):
        rows = dataset(root)
        runtime = ReplayRuntime(sha256(root / "train/train_tools_catalog.json"))
        created = []
        with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime), \
             patch("lifemate.evaluation.runner.load_toolenv", env_loader(runtime, created)), \
             redirect_stdout(io.StringIO()):
            plan = prepare_experiment(root, root / "work", count=count, oracle=True,
                                      agent_options=OPTIONS)
            runtime.before_reference = lambda: self.assertGreaterEqual(len(backend.seen), len(created))
            backend.before_generate = lambda messages: self.assertEqual(runtime.reference_ids, [])
            report = run_experiment(plan, backend=backend)
        return rows, runtime, created, plan, report

    def test_subset_runs_agent_once_per_id_and_scores_same_denominator(self):
        with scratch() as root:
            backend = ScriptedBackend()
            _, runtime, created, plan, report = self.run_fixture(root, count=1, backend=backend)
            summary = report["summary"]
            self.assertEqual(summary["selected_samples"], 1)
            self.assertEqual(summary["local_estimated_score_100"], 100)
            self.assertEqual([sid for sid, _ in created], plan["selected_ids"])
            self.assertEqual(runtime.reference_ids, plan["selected_ids"])
            self.assertEqual(summary["prediction_rows"], 1)
            self.assertEqual(summary["trace_rows"], 1)
            self.assertTrue(summary["validation"]["trace_coverage_checked"])
            predictions = read_rows(report["predictions"])
            self.assertEqual(predictions[0]["tool_calls"], created[0][1].get_tool_calls())
            self.assertEqual(predictions[0]["state_hash"], created[0][1].get_state_hash())
            encoded = json.dumps(backend.seen)
            self.assertNotIn('"label":', encoded)
            self.assertNotIn('"final_state":', encoded)
            self.assertEqual(read_json(report["summary_path"])["selection_sha256"], sha256(plan["selection"]))

    def test_cli_evaluate_uses_the_same_subset_as_inference(self):
        from lifemate.evaluation.__main__ import main as evaluate_cli

        with scratch() as root:
            _, runtime, _, plan, report = self.run_fixture(root, count=1, backend=ScriptedBackend())
            self.assertEqual(len(read_json(Path(plan["bundle"]) / "validation/manifest.json")["ids"]), 2)
            output = root / "cli-evaluation"
            with patch("lifemate.evaluation.__main__.OfficialRuntime", return_value=runtime), \
                 redirect_stdout(io.StringIO()):
                status = evaluate_cli(["evaluate", "--data-root", str(root), "--bundle", plan["bundle"],
                    "--split", "validation", "--selection", plan["selection"],
                    "--predictions", report["predictions"], "--traces", report["traces"],
                    "--output", str(output)])
            self.assertEqual(status, 0)
            metrics = read_json(output / "metrics.json")
            self.assertEqual(metrics["counts"]["samples"], 1)
            self.assertEqual(metrics["score"], 100)
            self.assertEqual(read_json(output / "selection.json")["selection_sha256"], sha256(plan["selection"]))
            self.assertEqual([r["id"] for r in read_rows(output / "samples.jsonl")], plan["selected_ids"])

    def test_full_selection_preserves_model_failure_and_incremental_outputs(self):
        with scratch() as root:
            rows = dataset(root)
            runtime = ReplayRuntime(sha256(root / "train/train_tools_catalog.json"))
            created = []
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime), \
                 patch("lifemate.evaluation.runner.load_toolenv", env_loader(runtime, created)), \
                 redirect_stdout(io.StringIO()):
                plan = prepare_experiment(root, root / "work", count=None, oracle=True,
                                          agent_options=OPTIONS)
                chosen = {row["id"]: row for row in rows}
                backend = ScriptedBackend(fail_text=chosen[plan["selected_ids"][1]]["audio_text"])
                observed_flushed = []
                def before_generate(messages):
                    self.assertEqual(runtime.reference_ids, [])
                    current = next(m["content"] for m in reversed(messages) if m["role"] == "user")
                    if current == backend.fail_text:
                        paths = list(Path(plan["output_dir"]).glob("run_*/agent/predictions.jsonl"))
                        observed_flushed.extend(read_rows(paths[0]))
                backend.before_generate = before_generate
                report = run_experiment(plan, backend=backend)
            summary = report["summary"]
            self.assertEqual(summary["selected_samples"], 2)
            self.assertEqual(len(created), 2)
            self.assertEqual(len(observed_flushed), 1)
            self.assertEqual(summary["statuses"], {"finished": 1, "model_error": 1})
            self.assertEqual(summary["local_estimated_score_100"], 50)
            self.assertEqual(summary["trace_rows"], 2)
            self.assertEqual(summary["prediction_rows"], 2)  # Actual second env exists; its empty log is real.
            self.assertEqual(summary["runtime_health"]["unhealthy_samples"], 1)
            self.assertEqual(len(read_rows(Path(report["evaluation_dir"]) / "samples.jsonl")), 2)

    def test_changed_config_cannot_reuse_a_preflight_plan(self):
        with scratch() as root:
            dataset(root)
            runtime = ReplayRuntime(sha256(root / "train/train_tools_catalog.json"))
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime):
                plan = prepare_experiment(root, root / "work", count=1, oracle=True, agent_options=OPTIONS)
            path = Path(plan["config"])
            config = read_json(path)
            config["generation"]["max_calls"] += 1
            path.write_text(json.dumps(config), encoding="utf-8")
            backend = ScriptedBackend()
            with self.assertRaisesRegex(ValueError, "inputs changed after preflight"):
                run_experiment(plan, backend=backend)
            self.assertEqual(backend.seen, [])

    def test_oracle_mode_cannot_be_flipped_after_preflight(self):
        with scratch() as root:
            dataset(root)
            runtime = ReplayRuntime(sha256(root / "train/train_tools_catalog.json"))
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime):
                plan = prepare_experiment(root, root / "work", count=1, oracle=True, agent_options=OPTIONS)
            plan["oracle"] = False
            backend = ScriptedBackend()
            with self.assertRaisesRegex(ValueError, "ASR/oracle mode changed after preflight"):
                run_experiment(plan, backend=backend)
            self.assertEqual(backend.seen, [])

    def test_changed_bundle_integrity_cannot_reuse_a_preflight_plan(self):
        with scratch() as root:
            dataset(root)
            runtime = ReplayRuntime(sha256(root / "train/train_tools_catalog.json"))
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime):
                plan = prepare_experiment(root, root / "work", count=1, oracle=True, agent_options=OPTIONS)
            integrity = Path(plan["bundle"]) / "integrity.json"
            integrity.write_text(integrity.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            backend = ScriptedBackend()
            with self.assertRaisesRegex(ValueError, "Prepared bundle changed after preflight"):
                run_experiment(plan, backend=backend)
            self.assertEqual(backend.seen, [])

    def test_changed_runtime_identity_cannot_reuse_a_preflight_plan(self):
        with scratch() as root:
            dataset(root)
            catalog_hash = sha256(root / "train/train_tools_catalog.json")
            runtime = ReplayRuntime(catalog_hash)
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=runtime):
                plan = prepare_experiment(root, root / "work", count=1, oracle=True, agent_options=OPTIONS)
            changed_runtime = ReplayRuntime(catalog_hash)
            changed_runtime.identity["runtime_sha256"] = "modified-official-runtime"
            backend = ScriptedBackend()
            with patch("lifemate.evaluation.runner.OfficialRuntime", return_value=changed_runtime), \
                 patch("lifemate.evaluation.runner.load_toolenv") as load_env:
                with self.assertRaisesRegex(ValueError, "ToolEnv runtime/catalog changed after preflight"):
                    run_experiment(plan, backend=backend)
            self.assertEqual(backend.seen, [])
            load_env.assert_not_called()


class OfficialExperimentTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Set LIFEMATE_DATA_ROOT for real SDK pipeline/evaluation")
    def test_one_real_train_env_exports_canonical_calls_and_runs_evaluation(self):
        # Synthetic speech and one scripted weather call test wiring only. The
        # original label is used exclusively by evaluator replay after inference.
        data_root = Path(os.environ["LIFEMATE_DATA_ROOT"])
        row = read_rows(data_root / "train/train.jsonl")[0]
        sid = row["id"]
        current = {"id": sid, "current_text": "Synthetic weather wiring verification",
                   "history_user_texts": [{"history_index": i, "text": "Synthetic earlier speech"}
                       for i, turn in enumerate(row.get("history", [])) if turn["role"] == "user"]}
        proposed = {"name": "weather.get_forecast",
                    "arguments": {"date": row["context"]["current_time"][:10]}}
        class Backend:
            def __init__(self):
                self.replies = iter(["<tool_call>" + json.dumps(proposed) + "</tool_call>",
                                     "<final>Wiring check complete</final>"])

            def generate(self, *args, **kwargs):
                return ModelReply(next(self.replies))

        with scratch() as root:
            bundle = root / "bundle"
            write_json(bundle / "validation/manifest.json", {"split": "validation", "physical_split": "train",
                       "ids": [sid], "groups": {sid: "real-sdk-wiring"}})
            write_rows(bundle / "validation/inputs.jsonl", [data.agent_item(row)])
            write_rows(bundle / "validation/references.jsonl", [{"id": sid, "group_id": "real-sdk-wiring",
                       "label": row["label"], "oracle_speech": current}])
            write_json(bundle / "metadata.json", {"source_sha256": sha256(data_root / "train/train.jsonl"),
                       "catalog_sha256": sha256(data_root / "train/train_tools_catalog.json")})
            write_json(bundle / "integrity.json", artifact_hashes(bundle))
            config_path, selection_path, speech_path = root / "config.json", root / "selection.json", root / "speech.jsonl"
            write_json(config_path, {"generation": OPTIONS,
                       "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()})
            write_rows(speech_path, [current])
            create_selection(bundle, "validation", selection_path, limit=1)
            prepared = prepare_run(bundle, "validation", data_root, speech_path,
                config_path=config_path, selection_path=selection_path, require_complete_speech=True)
            plan = {"bundle": str(bundle), "data_root": str(data_root), "config": str(config_path),
                    "selection": str(selection_path), "speech_path": str(speech_path), "oracle": False,
                    "output_dir": str(root / "plan"), "selected_ids": [sid],
                    "preflight": {"selection_sha256": sha256(selection_path), "config_sha256": sha256(config_path),
                        "bundle_integrity_sha256": sha256(bundle / "integrity.json"),
                        "speech_sha256": sha256(speech_path), "code": code_fingerprint(),
                        "runtime": prepared["runtime"].identity, "mode": "asr_text"}}
            inference_factory = load_toolenv(data_root, "train")
            created = []
            def counting_loader(*_):
                def make(sample_id):
                    env = inference_factory(sample_id)
                    created.append(env)
                    return env
                return make
            with patch("lifemate.evaluation.runner.load_toolenv", counting_loader), redirect_stdout(io.StringIO()):
                report = run_experiment(plan, backend=Backend())
            self.assertEqual(len(created), 1)
            self.assertEqual(report["summary"]["selected_ids"], [sid])
            self.assertEqual(report["summary"]["selected_samples"], 1)
            self.assertEqual(report["summary"]["trace_rows"], 1)
            self.assertEqual(report["summary"]["prediction_rows"], 1)
            prediction = read_rows(report["predictions"])[0]
            self.assertEqual(prediction["tool_calls"], created[0].get_tool_calls())
            self.assertEqual(prediction["tool_calls"], [proposed])
            self.assertEqual(prediction["state_hash"], created[0].get_state_hash())
            samples = read_rows(Path(report["evaluation_dir"]) / "samples.jsonl")
            self.assertEqual(len(samples), 1)
            self.assertTrue(samples[0]["submission_valid"])
            self.assertEqual(samples[0]["score"], 0)  # Deliberately incomplete original request.
            self.assertEqual(read_json(Path(report["evaluation_dir"]) / "metrics.json")["counts"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
