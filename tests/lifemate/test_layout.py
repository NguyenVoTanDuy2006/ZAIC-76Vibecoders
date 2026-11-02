"""Exercise launchers from their new folders without loading model weights."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from lifemate import agent, paths
from asr.integration.predictions import assemble_speech_rows, build_audio_manifest, write_speech_file
from test_predictions import REPO, fixture, fake_predictions


@contextmanager
def working_directory(path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def asr_launcher():
    script = REPO / "src/asr/notebooks/integration/predict_asr.py"
    namespace = {"__name__": "asr_launcher_under_test", "__file__": str(script)}
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    return namespace


class LayoutIntegrationTests(TestCase):
    def test_asr_launcher_output_is_the_agent_default_input_even_outside_repo_cwd(self):
        with fixture() as (root, items, _, __):
            with working_directory(root), patch.object(paths, "output_root", return_value=root / "output"):
                launcher = asr_launcher()
                predict = Mock(return_value={"prediction_path": "demo.jsonl", "conversations": 2, "audio_clips": 3})
                launcher["predict_lifemate"] = predict
                with redirect_stdout(io.StringIO()):
                    launcher["main"]()
                generated_file = predict.call_args.args[3] / "asr_predictions.jsonl"
                self.assertEqual(generated_file, paths.asr_prediction_file("train"))
                write_speech_file(generated_file,
                    assemble_speech_rows(items, build_audio_manifest(items, root, "train"), fake_predictions()))
                transcribe = agent.load_asr(root, "train")  # no explicit prediction path
                result = transcribe({"id": "a", "history_user_audio": [{"history_index": 1}]})
                self.assertEqual(result["current_text"], "ASR_CURRENT")
                self.assertEqual(result["history_user_texts"][0]["text"], "ASR_OLD")

    def test_kaggle_launchers_share_working_output_paths(self):
        with patch.object(paths, "_on_kaggle", return_value=True):
            launcher = asr_launcher()
            predict = Mock(return_value={"prediction_path": "demo.jsonl", "conversations": 2, "audio_clips": 3})
            launcher["predict_lifemate"] = predict
            with redirect_stdout(io.StringIO()):
                launcher["main"]()
            expected = Path("/kaggle/working/lifemate/asr/train/asr_predictions.jsonl")
            self.assertEqual(predict.call_args.args[3] / "asr_predictions.jsonl", expected)
            self.assertEqual(paths.asr_prediction_file("train"), expected)
            self.assertEqual(paths.agent_output_dir(), Path("/kaggle/working/lifemate/agent"))

    def test_agent_notebook_runs_from_its_new_folder_using_the_shared_module(self):
        notebook = REPO / "notebooks/lifemate/run_agent.ipynb"
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        for on_kaggle in (False, True):
            with self.subTest(on_kaggle=on_kaggle):
                run = Mock(return_value={"submission_path": "demo.jsonl", "trace_dir": "demo-traces"})
                namespace = {"__name__": "notebook_launcher_under_test"}
                with working_directory(notebook.parent), patch.dict(os.environ), \
                     patch.object(paths, "_on_kaggle", return_value=on_kaggle), \
                     patch.object(agent, "run_full_pipeline", run), redirect_stdout(io.StringIO()):
                    os.environ.pop("ZAIC_PROJECT_ROOT", None)
                    os.environ.pop("LIFEMATE_DATA_ROOT", None)
                    for index, cell in enumerate(nb["cells"]):
                        if cell["cell_type"] == "code":
                            self.assertEqual(cell["outputs"], [])
                            exec(compile("".join(cell["source"]), f"run_agent.ipynb:{index}", "exec"), namespace)
                expected_data = (Path("/kaggle/input/datasets/khangl123/zalo-ai/train+public_test")
                                 if on_kaggle else Path("C:/zalo_ai/train+public_test"))
                expected_output = Path("/kaggle/working/lifemate") if on_kaggle else REPO / "output/lifemate"
                run.assert_called_once()
                self.assertEqual(namespace["PROJECT_ROOT"], REPO)
                self.assertEqual(run.call_args.kwargs["data_root"], expected_data)
                self.assertEqual(run.call_args.kwargs["asr_predictions_path"], expected_output / "asr/train/asr_predictions.jsonl")
                self.assertEqual(run.call_args.kwargs["output_dir"], expected_output / "agent")
                self.assertTrue(run.call_args.kwargs["model_config"].local_files_only)
                self.assertNotIn("sample_ids", run.call_args.kwargs)
                self.assertNotIn("limit", run.call_args.kwargs)

    def test_agent_main_honors_custom_data_and_selects_local_or_kaggle_files(self):
        custom_data = REPO / "custom-data"
        for on_kaggle in (False, True):
            with self.subTest(on_kaggle=on_kaggle), \
                 patch.dict(os.environ, {"LIFEMATE_DATA_ROOT": str(custom_data)}), \
                 patch.object(paths, "_on_kaggle", return_value=on_kaggle), \
                 patch.object(agent, "run_full_pipeline", return_value={"submission_path": "demo.jsonl", "trace_dir": "traces"}) as run, \
                 redirect_stdout(io.StringIO()):
                agent.main()
                expected_output = Path("/kaggle/working/lifemate") if on_kaggle else REPO / "output/lifemate"
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs["data_root"], custom_data)
                self.assertEqual(run.call_args.kwargs["split"], "train")
                self.assertEqual(run.call_args.kwargs["asr_predictions_path"], expected_output / "asr/train/asr_predictions.jsonl")
                self.assertEqual(run.call_args.kwargs["output_dir"], expected_output / "agent")
                self.assertTrue(run.call_args.kwargs["model_config"].local_files_only)
                self.assertNotIn("sample_ids", run.call_args.kwargs)
                self.assertNotIn("limit", run.call_args.kwargs)
