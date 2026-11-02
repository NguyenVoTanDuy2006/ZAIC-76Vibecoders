import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


SCRIPT = Path("src/asr/notebooks/benchmark/benchmark-qwen-full-adapters-kaggle.py")


class QwenKaggleNotebookTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("qwen_kaggle_notebook", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()

    def make_run(self, method, dataset="models"):
        run = self.inputs / dataset / f"qwen_{method}_full"
        checkpoint = run / "epoch_2_adapter"
        checkpoint.mkdir(parents=True)
        (run / "best_epoch.txt").write_text("2")
        (checkpoint / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA" if method == "pissa" else "ADALORA"}))
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
        return run

    def test_finds_train_under_new_kaggle_mount(self):
        train = self.inputs / "datasets/owner/zalo/version/train+public_test/train"
        (train / "audio").mkdir(parents=True)
        (train / "train.jsonl").write_text("{}\n")
        self.assertEqual(self.module.discover_train_root(self.inputs), train.resolve())

    def test_ambiguous_train_requires_explicit_override(self):
        for name in ("first", "second"):
            path = self.inputs / name
            (path / "audio").mkdir(parents=True)
            (path / "train.jsonl").touch()
        with self.assertRaisesRegex(ValueError, "DATASET_ROOT_OVERRIDE"):
            self.module.discover_train_root(self.inputs)
        self.assertEqual(self.module.discover_train_root(self.inputs, self.inputs / "second"),
                         (self.inputs / "second").resolve())

    def test_model_configs_point_directly_to_kaggle_exports_without_copying_weights(self):
        pissa = self.make_run("pissa", "pissa-upload")
        adalora = self.make_run("adalora", "adalora-upload")
        paths = self.module.resolve_model_configs(Path.cwd(), self.root / "configs", adapter_overrides={
            "qwen_vimd_pissa_full": pissa / "epoch_2_adapter",
            "qwen_vimd_adalora_full": adalora / "epoch_2_adapter",
        })
        first = json.loads(paths[0].read_text())
        self.assertEqual(first["model"]["adapter_path"], str(pissa / "epoch_2_adapter"))
        self.assertEqual(first["model"]["backend"], "qwen3_asr")
        self.assertFalse((self.root / "configs/adapter_model.safetensors").exists())

    def test_adapter_override_key_typo_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown.*override"):
            self.module.resolve_model_configs(Path.cwd(), self.root / "configs",
                                               adapter_overrides={"typo": self.root})

    def test_sparse_source_bootstrap_is_pinned_and_reusable(self):
        repo = self.root / "repo"
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        git("init")
        git("config", "user.email", "test@example.com")
        git("config", "user.name", "Test")
        module = repo / "src/asr/benchmark/workflow/benchmark_qwen_adapters.py"
        module.parent.mkdir(parents=True)
        module.write_text("# fixture source\n")
        (repo / "requirements").mkdir()
        (repo / "requirements/qwen-adapter-benchmark.txt").touch()
        git("add", ".")
        git("commit", "-m", "fixture")
        revision = git("rev-parse", "HEAD")
        target = self.root / "source"
        self.module.bootstrap_source(target, str(repo), revision)
        self.module.bootstrap_source(target, str(repo), revision)
        self.assertEqual((target / module.relative_to(repo)).read_text(), "# fixture source\n")
        marker = json.loads((target / ".source_manager.json").read_text())
        self.assertEqual(marker["commit"], revision)

    def test_historical_pinned_checkout_keeps_its_original_package_name(self):
        legacy = self.root / "historical-source/src/benchmark_baseline_asr/workflow/benchmark_qwen_adapters.py"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("# historical fixture\n")
        root = self.root / "historical-source"
        self.assertEqual(self.module.benchmark_package(root), "benchmark_baseline_asr")
        current = root / "src/asr/benchmark/workflow/benchmark_qwen_adapters.py"
        current.parent.mkdir(parents=True)
        current.write_text("# current fixture\n")
        self.assertEqual(self.module.benchmark_package(root), "asr.benchmark")

    def test_existing_pipless_venv_is_repaired(self):
        env = self.root / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(env)], check=True)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.module.ensure_pip(python)
        result = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True)
        self.assertEqual(result.returncode, 0)

    def test_prepare_writes_reports_outside_readonly_input_and_weights_outside_output(self):
        runs = [self.make_run(method) for method in ("pissa", "adalora")]
        for run in runs:
            cfg_path = run / "epoch_2_adapter/adapter_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg.update(base_model_name_or_path="Qwen/Qwen3-ASR-1.7B-hf", revision=None,
                       init_lora_weights=True, rank_pattern={"test.lora_E": [True]})
            cfg_path.write_text(json.dumps(cfg))
            if "pissa" in run.name:
                (run / "epoch_2_adapter/training_method.json").write_text(json.dumps({
                    "export": "converted_standard_lora", "load_on": "original_pretrained_base"}))
        data = self.inputs / "zalo/train"
        (data / "audio").mkdir(parents=True)
        with wave.open(str(data / "audio/test.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 160)
        manifest = '{"id":"test","audio":"audio/test.wav","audio_text":"xin chào"}\n'
        (data / "train.jsonl").write_text(manifest, encoding="utf-8")
        m = self.module
        m.MODEL_CONFIG_FILES = m.resolve_model_configs(Path.cwd(), self.root / "scratch/configs", adapter_overrides={
            "qwen_vimd_pissa_full": runs[0] / "epoch_2_adapter",
            "qwen_vimd_adalora_full": runs[1] / "epoch_2_adapter",
        })
        m.DATASET_ROOT, m.OUTPUT_ROOT, m.BASE_CACHE = data, self.root / "working", self.root / "scratch/cache"
        m.SOURCE_ROOT, m.VENV_PYTHON, m.GPU_IDS = Path.cwd(), sys.executable, [0]
        m.child_env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
        config, run_dir = m.prepare_run("fixture")
        self.assertTrue((run_dir / "manifest.jsonl").is_file())
        download = json.loads((config.parent / "download.json").read_text())
        self.assertEqual(download["models_dir"], str(self.root / "scratch/cache"))
        self.assertEqual((data / "train.jsonl").read_text(encoding="utf-8"), manifest)


if __name__ == "__main__":
    unittest.main()
