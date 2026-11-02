import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SetupTests(unittest.TestCase):
    def test_existing_venv_without_pip_is_repaired_before_installing(self):
        from asr.finetuning.kaggle_setup import ensure_pip

        calls = []

        def run(command, **kwargs):
            calls.append([str(v) for v in command])
            if command[-1] == "--version":
                return subprocess.CompletedProcess(command, 0 if len(calls) > 1 else 1)
            return subprocess.CompletedProcess(command, 0)

        with patch("asr.finetuning.kaggle_setup.subprocess.run", side_effect=run):
            ensure_pip(Path("/venv/bin/python"))
        self.assertTrue(any("ensurepip" in c for c in calls))

    def test_manifest_discovery_refuses_ambiguous_inputs(self):
        from asr.finetuning.kaggle_setup import find_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["a", "b"]:
                (root / name / "audio").mkdir(parents=True)
                (root / name / "train.jsonl").write_text("{}")
            with self.assertRaisesRegex(ValueError, "multiple"):
                find_dataset(root)
            self.assertEqual(find_dataset(root, root / "a"), root / "a")


if __name__ == "__main__":
    unittest.main()
