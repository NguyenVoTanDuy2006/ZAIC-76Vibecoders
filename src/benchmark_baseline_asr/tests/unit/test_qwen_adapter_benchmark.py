from __future__ import annotations

import json
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


class QwenAdapterBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for method, kind in (("pissa", "LORA"), ("adalora", "ADALORA")):
            run = self.root / "models" / f"qwen_{method}_full"
            for epoch in (1, 2):
                checkpoint = run / f"epoch_{epoch}_adapter"
                checkpoint.mkdir(parents=True)
                config = {"base_model_name_or_path": "Qwen/Qwen3-ASR-1.7B-hf",
                          "revision": "a" * 40 if method == "pissa" else None,
                          "peft_type": kind, "init_lora_weights": True,
                          "rank_pattern": {"model.layer.lora_A": [True, False]} if method == "adalora" else {}}
                (checkpoint / "adapter_config.json").write_text(json.dumps(config))
                (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture-weights")
                if method == "pissa":
                    (checkpoint / "training_method.json").write_text(json.dumps({
                        "method": "pissa", "export": "converted_standard_lora",
                        "load_on": "original_pretrained_base"}))
            (run / "best_epoch.txt").write_text("2")
            (run / "run_config.json").write_text(json.dumps({"model_revision": "a" * 40}))

    def test_selects_best_full_checkpoints_and_pins_training_base(self):
        from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import discover_full_adapters
        selected = discover_full_adapters(self.root / "models")
        self.assertEqual([m["key"] for m in selected], ["qwen_pissa_full", "qwen_adalora_full"])
        for model in selected:
            self.assertEqual(model["settings"]["revision"], "a" * 40)
            self.assertTrue(model["settings"]["adapter_path"].endswith("epoch_2_adapter"))

    def test_changed_weights_change_resume_identity(self):
        from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import discover_full_adapters
        before = discover_full_adapters(self.root / "models")[0]["settings"]["adapter_sha256"]
        (self.root / "models/qwen_pissa_full/epoch_2_adapter/adapter_model.safetensors").write_bytes(b"different")
        after = discover_full_adapters(self.root / "models")[0]["settings"]["adapter_sha256"]
        self.assertNotEqual(before, after)

    def test_missing_best_epoch_does_not_guess_checkpoint(self):
        from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import discover_full_adapters
        (self.root / "models/qwen_adalora_full/best_epoch.txt").unlink()
        with self.assertRaisesRegex(ValueError, "best_epoch"):
            discover_full_adapters(self.root / "models")

    def test_unconverted_pissa_is_rejected(self):
        from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import discover_full_adapters
        path = self.root / "models/qwen_pissa_full/epoch_2_adapter/training_method.json"
        path.write_text('{"export": "raw_pissa"}')
        with self.assertRaisesRegex(ValueError, "converted_standard_lora"):
            discover_full_adapters(self.root / "models")

    def test_conflicting_base_revision_is_rejected(self):
        from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import discover_full_adapters
        (self.root / "models/qwen_adalora_full/run_config.json").write_text('{"model_revision": "other"}')
        with self.assertRaisesRegex(ValueError, "revision"):
            discover_full_adapters(self.root / "models")

    def test_prepare_uses_all_manifest_rows_and_configured_gpu_count(self):
        from benchmark_baseline_asr.workflow.benchmark_qwen_adapters import prepare_adapter_benchmark
        from benchmark_baseline_asr.application.config_service import resolve_experiment
        data = self.root / "train"
        data.mkdir()
        rows = []
        for index in range(3):
            with wave.open(str(data / f"{index}.wav"), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x00" * 160)
            rows.append({"id": str(index), "audio": f"{index}.wav", "audio_text": "xin chào"})
        (data / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        # Extra audio files must never expand the labelled benchmark.
        (data / "unlabelled.wav").write_bytes(b"not a wave")
        config_path, run_dir, samples = prepare_adapter_benchmark(
            self.root / "models", data, self.root / "output", "fixture", gpu_ids=[0, 1])
        config = resolve_experiment(config_path)
        self.assertEqual(len(samples), 3)
        self.assertIsNone(config["_loaded"]["dataset"]["limit"])
        self.assertEqual(len(list((run_dir / "shards").glob("*.jsonl"))), 2)
        self.assertFalse(config["_loaded"]["metrics"]["semantic_score"]["enabled"])
        self.assertEqual(config["models"][0]["settings"]["max_new_tokens"], 768)
        # Successful re-preparation may resume; changing the limit may not reuse it.
        prepare_adapter_benchmark(self.root / "models", data, self.root / "output", "fixture", gpu_ids=[0, 1])
        with self.assertRaisesRegex(ValueError, "new run_id"):
            prepare_adapter_benchmark(self.root / "models", data, self.root / "output", "fixture",
                                      gpu_ids=[0, 1], limit=1)


class QwenHFTranscriptionTests(unittest.TestCase):
    def test_decodes_only_new_tokens_and_preserves_integer_inputs(self):
        import torch
        from benchmark_baseline_asr.domain.entities import AudioInput
        from benchmark_baseline_asr.infrastructure.models.factory import create_adapter

        class Model:
            device, dtype = "cpu", torch.float32

            def to(self, device):
                return self

            def eval(self):
                return self

            def generate(self, input_ids, input_features, attention_mask, **kwargs):
                if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
                    raise AssertionError("token/mask dtype changed")
                if input_features.dtype != torch.float32:
                    raise AssertionError("features must follow model dtype")
                if kwargs != {"max_new_tokens": 768, "do_sample": False, "num_beams": 1}:
                    raise AssertionError("decode settings changed")
                return torch.tensor([[1, 2, 3, 7, 8]])

        class Processor:
            def apply_transcription_request(self, *, audio, language):
                if language != "Vietnamese":
                    raise AssertionError("language must be forced")
                return {"input_ids": torch.tensor([[1, 2, 3]]),
                        "input_features": torch.zeros(1, 2, dtype=torch.float64),
                        "attention_mask": torch.tensor([[1, 1, 1]])}

            def decode(self, tokens, *, return_format):
                if tokens.tolist() == [[7, 8]] and return_format == "transcription_only":
                    return [" xin chào "]
                return ["prompt leaked"]

        # Replace only disk/Hub loading of a 1.7B model; exercise inference logic.
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **kw: Processor())
        transformers.Qwen3ASRForConditionalGeneration = types.SimpleNamespace(from_pretrained=lambda *a, **kw: Model())
        with patch.dict("sys.modules", {"transformers": transformers}):
            adapter = create_adapter("qwen_hf_peft")
            adapter.load("unused", {"max_new_tokens": 768}, {"device": "cpu", "dtype": "float32"})
            result = adapter.transcribe(AudioInput(Path("test.wav"), torch.zeros(160).numpy(), 16000))
        self.assertEqual(result.text, "xin chào")
        self.assertEqual(result.language, "Vietnamese")

    def test_wrong_sample_rate_is_rejected_before_processor(self):
        from benchmark_baseline_asr.domain.entities import AudioInput
        from benchmark_baseline_asr.infrastructure.models.factory import create_adapter
        adapter = create_adapter("qwen_hf_peft")
        with self.assertRaisesRegex(ValueError, "16000"):
            adapter.transcribe(AudioInput(Path("test.wav"), [], 48000))


if __name__ == "__main__":
    unittest.main()
