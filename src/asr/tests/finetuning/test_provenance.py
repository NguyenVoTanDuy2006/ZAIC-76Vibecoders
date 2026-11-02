import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr.finetuning.config import load_config


class ProvenanceTests(unittest.TestCase):
    def test_shared_asr_source_changes_training_identity(self):
        from asr.finetuning.__main__ import provenance

        original = Path.read_text
        before = provenance(load_config())["source_fingerprint"]

        def changed_common(path, *args, **kwargs):
            content = original(path, *args, **kwargs)
            if path.as_posix().endswith("/asr/common/metrics.py"):
                return content + "\n# changed shared metric\n"
            return content

        with patch.object(Path, "read_text", changed_common):
            after = provenance(load_config())["source_fingerprint"]
        self.assertNotEqual(before, after)

    def test_changed_runtime_is_rejected_without_overwriting_original_record(self):
        from asr.finetuning.__main__ import write_provenance

        pins = {
            line.split("==")[0]: line.split("==")[1]
            for line in (Path(__file__).parents[2] / "finetuning/requirements.txt")
            .read_text()
            .splitlines()
            if "==" in line and not line.startswith("#")
        }
        pins["torch"] = "2.6.0+cpu"
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("platform.python_version", return_value="3.12.9"),
        ):
            root = Path(tmp)
            with patch("importlib.metadata.version", side_effect=lambda n: pins[n]):
                write_provenance(root, load_config())
            before = (root / "provenance.json").read_bytes()
            pins["transformers"] = "5.18.0"
            with patch("importlib.metadata.version", side_effect=lambda n: pins[n]):
                with self.assertRaisesRegex(ValueError, "transformers|runtime"):
                    write_provenance(root, load_config())
            self.assertEqual((root / "provenance.json").read_bytes(), before)
            pins["transformers"] = "5.17.0"
            old = json.loads(before)
            old["packages"]["transformers"] = "5.16.0"
            (root / "provenance.json").write_text(json.dumps(old))
            stored = (root / "provenance.json").read_bytes()
            with patch("importlib.metadata.version", side_effect=lambda n: pins[n]):
                with self.assertRaisesRegex(ValueError, "runtime"):
                    write_provenance(root, load_config())
            self.assertEqual((root / "provenance.json").read_bytes(), stored)

    def test_python_312_patch_updates_preserve_initial_provenance(self):
        from asr.finetuning.__main__ import write_provenance

        pins = {
            line.split("==")[0]: line.split("==")[1]
            for line in (Path(__file__).parents[2] / "finetuning/requirements.txt")
            .read_text()
            .splitlines()
            if "==" in line and not line.startswith("#")
        }
        pins["torch"] = "2.6.0+cu124"
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("importlib.metadata.version", side_effect=lambda n: pins[n]),
        ):
            root = Path(tmp)
            with patch("platform.python_version", return_value="3.12.8"):
                write_provenance(root, load_config())
            with patch("platform.python_version", return_value="3.12.9"):
                write_provenance(root, load_config())
            self.assertEqual(
                json.loads((root / "provenance.json").read_text())["python"], "3.12.8"
            )
            with patch("platform.python_version", return_value="3.13.1"):
                with self.assertRaisesRegex(ValueError, "Python|python"):
                    write_provenance(root, load_config())
