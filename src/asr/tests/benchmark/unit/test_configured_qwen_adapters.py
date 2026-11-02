import json
import tempfile
import unittest
import wave
from pathlib import Path


class ConfiguredQwenAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for method, peft_type in (("pissa", "LORA"), ("adalora", "ADALORA")):
            checkpoint = self.root / f"input/models/owner/adapters/pytorch/{method}-full/1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "Qwen/Qwen3-ASR-1.7B-hf", "peft_type": peft_type,
                "revision": None, "init_lora_weights": True,
                "rank_pattern": {"layer.lora_E": [False, True]} if method == "adalora" else {},
            }))
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
            path = self.root / f"{method}.json"
            path.write_text(json.dumps({"platform": "kaggle", "model": {
                "key": f"qwen_vimd_{method}_full", "backend": "qwen3_asr",
                "model_id": "Qwen/Qwen3-ASR-1.7B-hf", "revision": "a" * 40,
                "adapter_path": str(checkpoint), "display_name": method,
                "dtype": "float16", "trust_remote_code": False,
            }, "run_id": f"qwen_vimd_{method}_full"}))
            self.paths.append(path)

    def test_flat_kaggle_model_loads_using_config_revision_without_training_run_files(self):
        from asr.benchmark.infrastructure.models.configured_adapters import load_adapter_model_configs
        selected = load_adapter_model_configs(self.paths)
        self.assertEqual([r["key"] for r in selected], ["qwen_vimd_pissa_full", "qwen_vimd_adalora_full"])
        self.assertEqual(selected[0]["settings"]["revision"], "a" * 40)
        self.assertEqual(Path(selected[0]["settings"]["adapter_path"]).parts[-2:], ("pissa-full", "1"))
        self.assertIsNone(selected[0]["settings"]["epoch"])
        self.assertEqual(selected[1]["settings"]["method"], "adalora")

    def test_checkpoint_revision_must_match_config(self):
        from asr.benchmark.infrastructure.models.configured_adapters import load_adapter_model_configs
        cfg = json.loads(self.paths[0].read_text())
        path = Path(cfg["model"]["adapter_path"]) / "adapter_config.json"
        adapter = json.loads(path.read_text()); adapter["revision"] = "b" * 40
        path.write_text(json.dumps(adapter))
        with self.assertRaisesRegex(ValueError, "revision"):
            load_adapter_model_configs(self.paths)

    def test_wrong_base_and_unconverted_pissa_are_rejected(self):
        from asr.benchmark.infrastructure.models.configured_adapters import load_adapter_model_configs
        cfg = json.loads(self.paths[0].read_text())
        path = Path(cfg["model"]["adapter_path"]) / "adapter_config.json"
        adapter = json.loads(path.read_text()); adapter["init_lora_weights"] = "pissa_niter_4"
        path.write_text(json.dumps(adapter))
        with self.assertRaisesRegex(ValueError, "converted"):
            load_adapter_model_configs(self.paths)

    def test_missing_input_reports_explicit_adapter_path(self):
        from asr.benchmark.infrastructure.models.configured_adapters import load_adapter_model_configs
        cfg = json.loads(self.paths[0].read_text())
        (Path(cfg["model"]["adapter_path"]) / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, r"pissa-full[/\\]1"):
            load_adapter_model_configs(self.paths)

    def test_requires_one_adapter_of_each_method(self):
        from asr.benchmark.infrastructure.models.configured_adapters import load_adapter_model_configs
        with self.assertRaisesRegex(ValueError, "PiSSA.*AdaLoRA"):
            load_adapter_model_configs([self.paths[0], self.paths[0]])

    def test_cli_accepts_research_configs_and_preserves_configured_keys(self):
        from asr.benchmark.workflow.benchmark_qwen_adapters import main
        data = self.root / "train"
        data.mkdir()
        with wave.open(str(data / "sample.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 160)
        (data / "train.jsonl").write_text('{"id":"sample","audio":"sample.wav","audio_text":"xin chào"}\n', encoding="utf-8")
        out = self.root / "output"
        self.assertEqual(main(["--model-configs", *map(str, self.paths), "--dataset-root", str(data),
                               "--output-root", str(out), "--run-id", "config-test", "--prepare-only"]), 0)
        resolved = json.loads((out / "runs/config-test/adapter_benchmark.json").read_text())
        self.assertEqual(resolved["models"][0]["key"], "qwen_vimd_pissa_full")
        self.assertEqual(resolved["runtime"]["dtype"], "float16")


if __name__ == "__main__":
    unittest.main()
