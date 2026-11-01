import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


class BundleTests(unittest.TestCase):
    def test_reports_exclude_native_state_and_resume_restores_only_run_artifacts(self):
        from zaic_pissa.bundles import create_bundles, restore_run_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            for name in [
                "dataset.json",
                "adapters/epoch-01/adapter_model.safetensors",
                "pissa_initial/adapter_config.json",
                "validation/base/summary.json",
                ".env",
                "models/weights.bin",
                "orphaned/adapter.bin",
                "venv/python",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            (root / "config.json").write_text(
                json.dumps({"model": {"model_id": "base"}})
            )
            (root / "checkpoints").mkdir()
            torch.save(
                {
                    "epoch": 1,
                    "history": [{"epoch": 1}],
                    "best": {"epoch": 1, "adapter": "adapters/epoch-01"},
                },
                root / "checkpoints/latest.pt",
            )
            paths = create_bundles(root, Path(tmp) / "bundles")
            with zipfile.ZipFile(paths["reports"]) as archive:
                self.assertIn(
                    "adapters/epoch-01/adapter_model.safetensors", archive.namelist()
                )
                self.assertNotIn("checkpoints/latest.pt", archive.namelist())
            destination = Path(tmp) / "restored"
            restore_run_bundle(paths["resume"], destination)
            self.assertTrue((destination / "checkpoints/latest.pt").is_file())
            for name in [".env", "models", "orphaned", "venv"]:
                self.assertFalse((destination / name).exists())
            with self.assertRaises(FileExistsError):
                restore_run_bundle(paths["resume"], destination)

    def test_archives_follow_checkpoint_boundary_and_rebuild_stale_metadata(self):
        from zaic_pissa.bundles import create_bundles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            (root / "checkpoints").mkdir(parents=True)
            best = {"epoch": 1, "adapter": "adapters/epoch-01"}
            for epoch in [1, 2]:
                for name in [
                    f"adapters/epoch-{epoch:02d}/adapter_model.safetensors",
                    f"validation/epoch-{epoch:02d}/summary.json",
                ]:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(epoch))
            (root / "history.json").write_text("[]")
            (root / "best.json").write_text("null")
            (root / "config.json").write_text(
                json.dumps({"model": {"model_id": "base"}})
            )
            (root / "best_adapter").mkdir()
            (root / "best_adapter/adapter_model.safetensors").write_text("stale")
            torch.save(
                {"epoch": 1, "history": [{"epoch": 1}], "best": best},
                root / "checkpoints/latest.pt",
            )
            paths = create_bundles(root, Path(tmp) / "bundles")
            for path in paths.values():
                with zipfile.ZipFile(path) as archive:
                    self.assertFalse(any("epoch-02" in n for n in archive.namelist()))
                    self.assertEqual(
                        json.loads(archive.read("history.json")), [{"epoch": 1}]
                    )
                    self.assertEqual(json.loads(archive.read("best.json")), best)
                    self.assertEqual(
                        archive.read("best_adapter/adapter_model.safetensors"), b"1"
                    )

    def test_restore_rejects_path_traversal_before_creating_outputs(self):
        from zaic_pissa.bundles import restore_run_bundle

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("../escape", "invalid")
                archive.writestr("checkpoints/latest.pt", "invalid")
            destination = Path(tmp) / "restore"
            with self.assertRaises(ValueError):
                restore_run_bundle(bundle, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
