import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark_baseline_asr.source_manager.cleanup import cleanup_source
from benchmark_baseline_asr.source_manager.download import download_source


class SourceManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-b", "test-ref"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo, check=True)
        (self.repo / "src/pkg").mkdir(parents=True)
        (self.repo / "src/pkg/main.py").write_text("VALUE=42\n", encoding="utf-8")
        (self.repo / "dataset.bin").write_text("exclude", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "fixture"], cwd=self.repo, check=True, capture_output=True)
        self.root = root
        self.managed = root / "working/_source"
        self.target = self.managed / "checkout"
        self.config = root / "source.json"
        self.write_config()

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, **updates):
        config = {"repo_url": str(self.repo), "ref": "test-ref", "source_dir": str(self.target),
                  "managed_root": str(self.managed), "include_paths": ["src"]}
        config.update(updates)
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def test_download_sparse_checkout_and_reuse(self):
        first = download_source(self.config)
        second = download_source(self.config)
        self.assertEqual(first.commit, second.commit)
        self.assertTrue((self.target / "src/pkg/main.py").is_file())
        self.assertFalse((self.target / "dataset.bin").exists())

    def test_cleanup_is_idempotent_and_keeps_other_working_outputs(self):
        download_source(self.config)
        artifact = self.root / "working/asr_benchmark/result.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("keep", encoding="utf-8")
        self.assertTrue(cleanup_source(self.config).removed)
        self.assertFalse(cleanup_source(self.config).removed)
        self.assertEqual(artifact.read_text(encoding="utf-8"), "keep")

    def test_cleanup_refuses_unmarked_or_symlink_targets(self):
        self.target.mkdir(parents=True)
        (self.target / "user-file").touch()
        with self.assertRaisesRegex(ValueError, "unmarked"):
            cleanup_source(self.config)
        self.assertTrue((self.target / "user-file").exists())


if __name__ == "__main__":
    unittest.main()
