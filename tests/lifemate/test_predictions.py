"""CPU handoff tests. Invented ASR/model replies are not accuracy benchmarks."""
import ast
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid
import wave

from asr.integration.predictions import (
    FileTranscriber, PredictionContractError, assemble_speech_rows, build_audio_manifest,
    convert_prediction_files, load_items, read_jsonl, validate_speech_rows, write_speech_file,
)
from asr.integration.predict import load_model_settings, predict_lifemate


REPO = Path(__file__).resolve().parents[2]


@contextmanager
def fixture():
    # Normal mkdir inherits Windows ACLs; all cleanup stays within this test root.
    base = REPO / ".test-scratch"
    base.mkdir(exist_ok=True)
    root = base / uuid.uuid4().hex
    root.mkdir()
    try:
        train = root / "train"
        (train / "audio").mkdir(parents=True)
        for filename in ("a.wav", "old.wav", "b.wav"):
            with wave.open(str(train / "audio" / filename), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x00" * 160)
        items = [
            {"id": "a", "audio": "audio/a.wav", "audio_text": "SECRET_GOLD_CURRENT",
             "label": {"tool_calls": ["SECRET_GOLD_LABEL"]},
             "context": {"installed_apps": {"calendar": {"tools": ["calendar.list_events"]}}},
             "history": [{"role": "assistant", "text": "Đã hiểu"},
                         {"role": "user", "audio": "audio/old.wav", "audio_text": "SECRET_GOLD_HISTORY"}]},
            {"id": "b", "audio": "audio/b.wav", "history": [],
             "context": {"installed_apps": {"calendar": {"tools": ["calendar.list_events"]}}}},
        ]
        (train / "train.jsonl").write_text("\n".join(json.dumps(row) for row in items), encoding="utf-8")
        catalog = {"apps": [
            {"id": "calendar", "tools": [{"name": "calendar.list_events", "parameters": {"type": "object"}}]},
            {"id": "sys", "tools": [{"name": "sys.ask_missing_info", "parameters": {"type": "object"}}]},
        ]}
        (train / "train_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        base_model = root / "model"
        base_model.mkdir()
        (base_model / "config.json").write_text("{}", encoding="utf-8")
        config = root / "asr.json"
        config.write_text(json.dumps({"adapter": "qwen_asr", "repo_id": "fake-local-asr", "revision": "r"}), encoding="utf-8")
        yield root, items, config, base_model
    finally:
        assert root.resolve().parent == base.resolve()
        shutil.rmtree(root)
        if not any(base.iterdir()):
            base.rmdir()


def fake_predictions():
    return [{"sample_id": "a", "text": "ASR_CURRENT"},
            {"sample_id": "a_history_001", "text": "ASR_OLD"},
            {"sample_id": "b", "text": "ASR_B"}]


class FakeAdapter:
    def __init__(self, fail_at=None):
        self.loaded = self.unloaded = 0
        self.audio_seen = []
        self.fail_at = fail_at

    def load(self, model_path, settings, runtime):
        self.loaded += 1

    def transcribe(self, audio):
        self.audio_seen.append(audio)
        if self.fail_at == len(self.audio_seen):
            raise RuntimeError("invented ASR failure")
        return SimpleNamespace(text="ASR_" + Path(audio).stem, language="Vietnamese")

    def unload(self):
        self.unloaded += 1


class SpeechContractTests(unittest.TestCase):
    def test_unknown_history_role_uses_prediction_contract_error(self):
        with fixture() as (root, items, _, __):
            items[0]["history"].append({"role": "tool", "text": "result"})
            with self.assertRaisesRegex(PredictionContractError, "History role"):
                build_audio_manifest(items, root, "train")

    def test_manifest_contains_current_and_original_user_history_indices_without_gold(self):
        with fixture() as (root, items, _, __):
            manifest = build_audio_manifest(items, root, "train")
            self.assertEqual([r["sample_id"] for r in manifest], ["a", "a_history_001", "b"])
            self.assertEqual([r["history_index"] for r in manifest], [None, 1, None])
            self.assertNotIn("SECRET_GOLD", json.dumps(manifest))

    def test_assembly_accepts_fine_tuning_prediction_but_never_reference(self):
        with fixture() as (root, items, _, __):
            manifest = build_audio_manifest(items, root, "train")
            predictions = [{"sample_id": r["sample_id"], "prediction": r["text"], "reference": "SECRET_GOLD"}
                           for r in fake_predictions()]
            rows = assemble_speech_rows(items, manifest, predictions)
            self.assertEqual(rows[0], {"id": "a", "current_text": "ASR_CURRENT",
                                      "history_user_texts": [{"history_index": 1, "text": "ASR_OLD"}]})
            self.assertEqual(rows[1]["history_user_texts"], [])
            self.assertNotIn("SECRET_GOLD", json.dumps(rows))

    def test_current_only_benchmark_cannot_silently_skip_history(self):
        with fixture() as (root, items, _, __):
            with self.assertRaisesRegex(PredictionContractError, "history"):
                assemble_speech_rows(items, build_audio_manifest(items, root, "train"), fake_predictions()[::2])

    def test_duplicate_unknown_and_nontext_predictions_are_rejected(self):
        with fixture() as (root, items, _, __):
            manifest = build_audio_manifest(items, root, "train")
            for rows in (fake_predictions() + [fake_predictions()[0]],
                         fake_predictions() + [{"sample_id": "unknown", "text": "x"}],
                         [{"sample_id": "a", "text": 123}]):
                with self.subTest(rows=rows), self.assertRaises(PredictionContractError):
                    assemble_speech_rows(items, manifest, rows)

    def test_wrong_id_duplicate_history_and_missing_history_are_rejected(self):
        with fixture() as (root, items, _, __):
            rows = assemble_speech_rows(items, build_audio_manifest(items, root, "train"), fake_predictions())
            variants = []
            bad = deepcopy(rows); bad[0]["id"] = "wrong"; variants.append(bad)
            bad = deepcopy(rows); bad[0]["history_user_texts"] *= 2; variants.append(bad)
            bad = deepcopy(rows); bad[0]["history_user_texts"] = []; variants.append(bad)
            bad = deepcopy(rows); bad[0]["history_user_texts"][0]["history_index"] = 0; variants.append(bad)
            for bad in variants:
                with self.assertRaises(PredictionContractError):
                    validate_speech_rows(bad, items)

    def test_audio_paths_cannot_escape_or_point_to_hidden_environment(self):
        with fixture() as (root, items, _, __):
            for path in ("../outside.wav", "train_environment/a/state.json"):
                altered = deepcopy(items); altered[0]["audio"] = path
                with self.assertRaises(PredictionContractError):
                    build_audio_manifest(altered, root, "train")

    def test_converter_and_file_transcriber_preserve_unicode_and_do_not_overwrite(self):
        with fixture() as (root, items, _, __):
            raw = root / "raw.jsonl"
            predictions = fake_predictions(); predictions[0]["text"] = "Đổi giờ chạy bộ"
            write_speech_file(raw, predictions)
            output = convert_prediction_files(root, "train", [raw], root / "asr.jsonl")
            transcribe = FileTranscriber(output, items)
            audio_input = {"id": "a", "audio": "unused.wav", "history_user_audio": [{"history_index": 1}]}
            speech = transcribe(audio_input)
            self.assertEqual(speech["current_text"], "Đổi giờ chạy bộ")
            speech["current_text"] = "mutated"
            self.assertEqual(transcribe(audio_input)["current_text"], "Đổi giờ chạy bộ")
            with self.assertRaises(FileExistsError):
                convert_prediction_files(root, "train", [raw], output)


class PredictionWorkflowTests(unittest.TestCase):
    def run_prediction(self, root, config, model, adapter):
        with redirect_stdout(io.StringIO()):
            return predict_lifemate(root, "train", config, root / "predictions", model_path=model,
                device="cpu", adapter_factory=lambda _: adapter, audio_loader=lambda path, _: path)

    def test_full_split_uses_existing_adapter_api_once_then_releases_and_publishes(self):
        with fixture() as (root, items, config, model):
            adapter = FakeAdapter()
            report = self.run_prediction(root, config, model, adapter)
            self.assertEqual((report["conversations"], report["audio_clips"]), (2, 3))
            self.assertEqual((adapter.loaded, adapter.unloaded, len(adapter.audio_seen)), (1, 1, 3))
            rows = read_jsonl(report["prediction_path"])
            self.assertEqual(rows[0]["history_user_texts"], [{"history_index": 1, "text": "ASR_old"}])
            self.assertEqual(set(rows[0]), {"id", "current_text", "history_user_texts"})
            self.assertNotIn("SECRET_GOLD", json.dumps(rows))

    def test_complete_cache_does_not_reload_asr(self):
        with fixture() as (root, _, config, model):
            self.run_prediction(root, config, model, FakeAdapter())
            adapter = FakeAdapter()
            report = self.run_prediction(root, config, model, adapter)
            self.assertEqual(report["new_audio_clips"], 0)
            self.assertEqual(adapter.loaded, 0)

    def test_failed_clip_keeps_cache_and_rerun_only_transcribes_missing_audio(self):
        with fixture() as (root, _, config, model):
            first = FakeAdapter(fail_at=2)
            with self.assertRaisesRegex(RuntimeError, "ASR failure"):
                self.run_prediction(root, config, model, first)
            self.assertEqual(first.unloaded, 1)
            self.assertFalse((root / "predictions/asr_predictions.jsonl").exists())
            self.assertEqual(len(read_jsonl(root / "predictions/audio_predictions.jsonl")), 1)
            second = FakeAdapter()
            report = self.run_prediction(root, config, model, second)
            self.assertEqual(report["new_audio_clips"], 2)
            self.assertEqual(len(second.audio_seen), 2)

    def test_changed_audio_refuses_to_mix_with_old_cache(self):
        with fixture() as (root, _, config, model):
            self.run_prediction(root, config, model, FakeAdapter())
            with (root / "train/audio/a.wav").open("ab") as stream:
                stream.write(b"changed")
            adapter = FakeAdapter()
            with self.assertRaisesRegex(PredictionContractError, "đã đổi"):
                self.run_prediction(root, config, model, adapter)
            self.assertEqual(adapter.loaded, 0)

    def test_single_pissa_config_is_accepted_and_adapter_digest_is_recorded(self):
        with fixture() as (root, _, config, __):
            adapter = root / "adapter"; adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "init_lora_weights": True}), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture-not-real-weights")
            config.write_text(json.dumps({"model": {"backend": "qwen3_asr", "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
                "revision": "a" * 40, "adapter_path": "adapter"}}), encoding="utf-8")
            settings = load_model_settings(config)
            self.assertEqual(settings["adapter"], "qwen_hf_peft")
            self.assertEqual(settings["adapter_path"], str(adapter.resolve()))
            self.assertTrue(settings["adapter_sha256"])

    def test_missing_adapter_does_not_silently_use_untrained_base(self):
        with fixture() as (_, __, config, ___):
            config.write_text(json.dumps({"model": {"backend": "qwen3_asr",
                "model_id": "Qwen/Qwen3-ASR-1.7B-hf", "adapter_path": None}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "không tự chuyển sang base"):
                load_model_settings(config)

    def test_public_split_without_any_gold_fields_is_supported(self):
        with fixture() as (root, items, config, model):
            public = root / "public_test"
            public.mkdir()
            shutil.copytree(root / "train/audio", public / "audio")
            for item in items:
                item.pop("audio_text", None); item.pop("label", None)
                for turn in item["history"]:
                    turn.pop("audio_text", None)
            (public / "public_test.jsonl").write_text("\n".join(json.dumps(row) for row in items), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                report = predict_lifemate(root, "public_test", config, root / "public-predictions",
                    model_path=model, device="cpu", adapter_factory=lambda _: FakeAdapter(),
                    audio_loader=lambda path, _: path)
            self.assertEqual(report["conversations"], 2)
            self.assertEqual(report["audio_clips"], 3)

    def test_stale_final_file_is_not_accepted_as_cache(self):
        with fixture() as (root, _, config, model):
            report = self.run_prediction(root, config, model, FakeAdapter())
            path = Path(report["prediction_path"])
            rows = read_jsonl(path); rows[0]["current_text"] = "tampered"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(PredictionContractError, "không khớp"):
                self.run_prediction(root, config, model, FakeAdapter())


def load_agent_module():
    # Reset mocks between cases; importing the module never loads an LLM.
    return importlib.reload(importlib.import_module("lifemate.agent"))


class AgentHandoffTests(unittest.TestCase):
    def test_agent_consumes_prediction_text_for_current_and_history_and_runs_whole_split(self):
        agent = load_agent_module()
        with fixture() as (root, items, _, __):
            speech_path = root / "speech.jsonl"
            write_speech_file(speech_path, assemble_speech_rows(items, build_audio_manifest(items, root, "train"), fake_predictions()))
            created = []
            class Env:
                def __init__(self): self.calls = []
                def get_tool_calls(self): return deepcopy(self.calls)
                def validate(self, name, arguments): return {"ok": True}
                def call(self, name, arguments):
                    self.calls.append({"name": name, "arguments": arguments})
                    return {"ok": True, "events": []}
                def get_state_hash(self): return hashlib.sha256(json.dumps(self.calls).encode()).hexdigest()
            def env_loader(*_):
                def make_env(cid):
                    env = Env(); created.append((cid, env)); return env
                return make_env
            replies = ['<tool_call>{"name":"calendar.list_events","arguments":{}}</tool_call>',
                       '<final>Demo only</final>', '<final>Demo only</final>']
            seen = []
            class Backend:
                def generate(self, messages, tools, config, remaining_seconds=None):
                    seen.append(deepcopy(messages)); return agent.ModelReply(replies.pop(0))
            with patch.dict(os.environ, {"ZAIC_PROJECT_ROOT": str(REPO)}), redirect_stdout(io.StringIO()):
                report = agent.run_full_pipeline(root, output_dir=root / "agent-output", asr_predictions_path=speech_path,
                                                toolenv_loader=env_loader, backend=Backend())
            self.assertTrue(report["complete_split"])
            self.assertEqual([cid for cid, _ in created], ["a", "b"])
            self.assertEqual(len(created[0][1].calls), 1)
            prompt = json.dumps(seen)
            self.assertIn("ASR_CURRENT", prompt); self.assertIn("ASR_OLD", prompt)
            self.assertNotIn("SECRET_GOLD", prompt)
            for row in read_jsonl(report["submission_path"]):
                self.assertEqual(set(row), {"id", "tool_calls", "state_hash"})

    def test_missing_prediction_file_stops_before_qwen_or_toolenv(self):
        agent = load_agent_module()
        with fixture() as (root, _, __, ___):
            qwen, toolenv = Mock(), Mock()
            agent.load_qwen_backend = qwen
            with patch.dict(os.environ, {"ZAIC_PROJECT_ROOT": str(REPO)}):
                with self.assertRaises(FileNotFoundError):
                    agent.run_full_pipeline(root, asr_predictions_path=root / "missing.jsonl", toolenv_loader=toolenv)
            qwen.assert_not_called(); toolenv.assert_not_called()

    def test_toolenv_loader_returns_factory_without_creating_env(self):
        agent = load_agent_module()
        factory = Mock()
        with patch.object(agent, "get_tool_env_factory", return_value=factory):
            make_env = agent.load_toolenv("data", "train")
            factory.assert_not_called()
            make_env("sample")
        factory.assert_called_once_with("train", "sample", root="data")

    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Set LIFEMATE_DATA_ROOT for optional official-runtime check")
    def test_prediction_file_to_official_toolenv_with_scripted_llm(self):
        # All speech and model replies here are invented. Only ToolEnv is real.
        agent = load_agent_module()
        data_root = Path(os.environ["LIFEMATE_DATA_ROOT"])
        items = load_items(data_root, "train")
        row = items[0]
        speech_rows = [{"id": item["id"], "current_text": "ASR demo weather request",
                        "history_user_texts": [{"history_index": i, "text": "ASR demo history"}
                            for i, turn in enumerate(item.get("history", [])) if turn["role"] == "user"]}
                       for item in items]
        tomorrow = (datetime.fromisoformat(row["context"]["current_time"]) + timedelta(days=1)).date().isoformat()
        replies = ['<tool_call>' + json.dumps({"name": "weather.get_forecast", "arguments": {"date": tomorrow}})
                   + '</tool_call>', '<final>Official ToolEnv wiring check only</final>']
        class Backend:
            def generate(self, *args, **kwargs): return agent.ModelReply(replies.pop(0))
        created = []
        def env_loader(root, split):
            factory = agent.get_tool_env_factory(root, split)
            def make_env(cid):
                env = factory(split, cid, root=root); created.append(env); return env
            return make_env
        with fixture() as (temp, _, __, ___):
            path = write_speech_file(temp / "real-metadata-demo-speech.jsonl", speech_rows)
            with patch.dict(os.environ, {"ZAIC_PROJECT_ROOT": str(REPO)}), redirect_stdout(io.StringIO()):
                report = agent.run_full_pipeline(data_root, sample_ids=[row["id"]], asr_predictions_path=path,
                    output_dir=temp / "official-wiring", toolenv_loader=env_loader, backend=Backend())
            self.assertEqual(len(created), 1)
            self.assertEqual(len(created[0].get_tool_calls()), 1)
            self.assertEqual(report["records"][0]["tool_calls"], created[0].get_tool_calls())
            self.assertEqual(report["records"][0]["state_hash"], created[0].get_state_hash())


if __name__ == "__main__":
    unittest.main()
