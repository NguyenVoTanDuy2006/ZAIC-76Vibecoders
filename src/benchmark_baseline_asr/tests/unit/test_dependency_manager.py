import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError
from unittest import mock

from benchmark_baseline_asr.infrastructure.runtime.dependencies import ensure_asr_runtime


class ASRDependencyManagerTests(unittest.TestCase):
    def test_keeps_existing_pinned_runtime_without_running_pip(self):
        installed = {"transformers": "4.57.6", "qwen-asr": "0.0.6"}

        with (
            mock.patch("importlib.metadata.version", side_effect=installed.__getitem__),
            mock.patch("benchmark_baseline_asr.infrastructure.runtime.dependencies.subprocess.run") as run,
        ):
            result = ensure_asr_runtime()

        self.assertEqual(result, installed)
        run.assert_not_called()

    def test_installs_qwen_and_pinned_transformers_when_runtime_is_outdated(self):
        installed = {"transformers": "5.0.0"}

        def lookup(package):
            if package not in installed:
                raise PackageNotFoundError(package)
            return installed[package]

        def install(command, check):
            self.assertTrue(check)
            installed.update({"transformers": "4.57.6", "qwen-asr": "0.0.6"})

        with (
            mock.patch("importlib.metadata.version", side_effect=lookup),
            mock.patch(
                "benchmark_baseline_asr.infrastructure.runtime.dependencies.subprocess.run",
                side_effect=install,
            ) as run,
        ):
            result = ensure_asr_runtime()

        command = run.call_args.args[0]
        self.assertIn("transformers==4.57.6", command)
        self.assertIn("qwen-asr==0.0.6", command)
        self.assertEqual(result, {"transformers": "4.57.6", "qwen-asr": "0.0.6"})

    def test_requests_kernel_restart_if_loaded_transformers_must_be_replaced(self):
        installed = {"transformers": "5.0.0", "qwen-asr": "0.0.6"}

        with (
            mock.patch("importlib.metadata.version", side_effect=installed.__getitem__),
            mock.patch("benchmark_baseline_asr.infrastructure.runtime.dependencies.subprocess.run"),
            mock.patch.dict(sys.modules, {"transformers": types.ModuleType("transformers")} ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Restart the Kaggle kernel"):
                ensure_asr_runtime()


if __name__ == "__main__":
    unittest.main()
