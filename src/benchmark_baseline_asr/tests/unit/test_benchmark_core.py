import json
import contextlib
import tempfile
import unittest
import wave
import sys
import types
from unittest import mock
from pathlib import Path

from benchmark_baseline_asr.application.config_service import resolve_experiment
from benchmark_baseline_asr.application.evaluation_service import evaluate_predictions
from benchmark_baseline_asr.application.sharding_service import balance_shards
from benchmark_baseline_asr.domain.entities import DatasetSample
from benchmark_baseline_asr.domain.entities import AudioInput
from benchmark_baseline_asr.infrastructure.dataset.train_jsonl import load_train_manifest
from benchmark_baseline_asr.infrastructure.evaluation.error_metrics import corpus_error_rates
from benchmark_baseline_asr.infrastructure.evaluation.text_normalizer import normalize_text
from benchmark_baseline_asr.infrastructure.storage.artifacts import write_jsonl
from benchmark_baseline_asr.workflow.transcribe import _fingerprint
from benchmark_baseline_asr.infrastructure.models.phowhisper import PhoWhisperAdapter
from benchmark_baseline_asr.application.transcription_service import ensure_resume_allowed


class BenchmarkCoreTests(unittest.TestCase):
    def test_manifest_reads_latest_turn_and_validates_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "audio/id/id.wav"
            audio.parent.mkdir(parents=True)
            with wave.open(str(audio), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b"\0\0" * 32000)
            record = {"id": "id", "audio": "audio/id/id.wav", "audio_text": "current",
                      "history": [{"audio_text": "old"}]}
            jsonl = root / "train.jsonl"
            jsonl.write_text(json.dumps(record), encoding="utf-8")
            rows = load_train_manifest(jsonl, root)
            self.assertEqual((rows[0].reference, rows[0].duration_seconds), ("current", 2.0))

    def test_shards_balance_duration_deterministically(self):
        samples = [DatasetSample(k, Path(k), k, d) for k, d in [("a", 10), ("b", 8), ("c", 7), ("d", 3)]]
        shards = balance_shards(samples, 2)
        self.assertEqual([[row.sample_id for row in shard] for shard in shards], [["a", "d"], ["b", "c"]])
        self.assertEqual(shards, balance_shards(samples, 2))

    def test_normalizer_keeps_diacritics_and_number_spelling(self):
        self.assertEqual(normalize_text(" HẸN   lúc 26 giờ! "), "hẹn lúc 26 giờ")

    def test_corpus_error_rates_count_substitution_deletion_and_insertion(self):
        result = corpus_error_rates([("abc", "adc"), ("x", "xy")], units="character")
        self.assertEqual((result.substitutions, result.deletions, result.insertions, result.reference_units), (1, 0, 1, 4))
        self.assertEqual(result.rate, 0.5)

    def test_evaluation_reports_raw_normalized_and_partial_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / "manifest.jsonl", [
                {"sample_id": "a", "reference": "Hẹn lúc 8 giờ!", "duration_seconds": 2},
                {"sample_id": "b", "reference": "chưa nhận dạng", "duration_seconds": 3},
            ])
            write_jsonl(root / "predictions.jsonl", [{"sample_id": "a", "text": "hẹn lúc 8 giờ", "latency_seconds": 1}])
            summary = evaluate_predictions(root / "manifest.jsonl", root / "predictions.jsonl", root)
            self.assertEqual(summary["coverage"], 0.5)
            self.assertEqual(summary["normalized_wer"], 0.0)
            self.assertFalse(summary["result_complete"])

    def test_kaggle_config_loads_four_models_and_two_gpus(self):
        config = resolve_experiment("src/benchmark_baseline_asr/configs/experiment.json", "kaggle")
        self.assertEqual(len(config["models"]), 4)
        self.assertEqual(config["_loaded"]["execution"]["gpu_ids"], [0, 1])

    def test_phowhisper_generation_budget_includes_decoder_prompt(self):
        config_root = Path("src/benchmark_baseline_asr/configs/models")
        self.assertLess(4 + PhoWhisperAdapter().max_new_tokens, 448)
        for model_name in ("phowhisper_medium", "phowhisper_large"):
            with self.subTest(model=model_name):
                model_config = json.loads((config_root / f"{model_name}.json").read_text(encoding="utf-8"))
                self.assertLess(4 + model_config["max_new_tokens"], 448)

    def test_resume_fingerprint_covers_audio_preprocessing_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            common = (manifest, {"repo": "model"}, {"revision": "rev"}, {"seed": 1}, {"gpu_ids": [0, 1]})
            first = _fingerprint(*common, {"sample_rate": 16000})
            second = _fingerprint(*common, {"sample_rate": 8000})
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_phowhisper_transcribe_uses_settings_saved_during_load(self):
        class Tensor:
            def to(self, device):
                return self

        class Processor:
            def __call__(self, waveform, sampling_rate, return_tensors):
                return {"input_features": Tensor()}

            def batch_decode(self, generated, skip_special_tokens):
                return [" recognized text "]

        class Model:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [1]

        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        adapter = PhoWhisperAdapter()
        adapter.processor = Processor()
        adapter.model = Model()
        adapter.device = "cpu"
        adapter.language = "vi"
        adapter.do_sample = False
        adapter.num_beams = 1
        adapter.max_new_tokens = 77
        with mock.patch.dict(sys.modules, {"torch": torch}):
            result = adapter.transcribe(AudioInput(Path("x.wav"), [0.0], 16000))
        self.assertEqual(result.text, "recognized text")
        self.assertEqual(adapter.model.kwargs["max_new_tokens"], 77)
        self.assertFalse(adapter.model.kwargs["do_sample"])

    def test_phowhisper_load_uses_transformers_dtype_argument(self):
        captured = {}

        class Processor:
            @staticmethod
            def from_pretrained(model_path, local_files_only):
                return object()

        class Model:
            def to(self, device):
                return self

            def eval(self):
                pass

        class ModelLoader:
            @staticmethod
            def from_pretrained(model_path, **kwargs):
                captured.update(kwargs)
                return Model()

        torch = types.ModuleType("torch")
        torch.float16 = object()
        torch.float32 = object()
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForSpeechSeq2Seq = ModelLoader
        transformers.AutoProcessor = Processor

        adapter = PhoWhisperAdapter()
        with mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            adapter.load("model-path", {"language": "vi"}, {"device": "cpu", "dtype": "float16"})

        self.assertIs(captured["dtype"], torch.float16)
        self.assertNotIn("torch_dtype", captured)

    def test_resume_disabled_refuses_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            predictions = Path(temp) / "worker_0/predictions.jsonl"
            predictions.parent.mkdir()
            predictions.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume is disabled"):
                ensure_resume_allowed(Path(temp), resume=False)
            self.assertTrue(predictions.is_file())


if __name__ == "__main__":
    unittest.main()
