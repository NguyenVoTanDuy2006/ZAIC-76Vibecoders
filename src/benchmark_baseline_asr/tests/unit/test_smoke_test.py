import unittest
from pathlib import Path
from unittest import mock

from benchmark_baseline_asr.domain.entities import AudioInput, DatasetSample, Transcription
from benchmark_baseline_asr.workflow.smoke_test import smoke_test_models


class SmokeTestWorkflowTests(unittest.TestCase):
    def test_runs_each_model_sequentially_on_one_median_duration_sample(self):
        events = []
        adapters = []

        class FakeAdapter:
            def __init__(self, model_key):
                self.model_key = model_key

            def load(self, model_path, model_config, runtime_config):
                events.append(("load", self.model_key, model_path, runtime_config["device"]))

            def transcribe(self, audio):
                self.assert_audio = audio
                events.append(("transcribe", self.model_key))
                return Transcription(text=f"transcript-{self.model_key}")

            def unload(self):
                events.append(("unload", self.model_key))

        models = [
            {"key": key, "settings": {"adapter": key, "revision": "rev"}}
            for key in ("model_a", "model_b", "model_c", "model_d")
        ]
        resolved = {row["key"]: {"path": f"/models/{row['key']}"} for row in models}
        samples = [
            DatasetSample("short", Path("short.wav"), "short", 2),
            DatasetSample("median", Path("median.wav"), "median", 5),
            DatasetSample("long", Path("long.wav"), "long", 9),
        ]
        audio = AudioInput(Path("median.wav"), [0.0], 16000)

        def make_adapter(model_key):
            adapter = FakeAdapter(model_key)
            adapters.append(adapter)
            return adapter

        with (
            mock.patch("benchmark_baseline_asr.workflow.smoke_test._prepare_audio", return_value=audio) as prepare,
            mock.patch("benchmark_baseline_asr.workflow.smoke_test.configure_determinism"),
            mock.patch(
                "benchmark_baseline_asr.workflow.smoke_test.timed_call",
                side_effect=lambda function, device: (function(), 0.01),
            ),
            mock.patch("benchmark_baseline_asr.workflow.smoke_test.create_adapter", side_effect=make_adapter),
        ):
            results = smoke_test_models(samples, models, resolved, {"sample_rate": 16000}, {"seed": 7})

        prepare.assert_called_once_with(Path("median.wav"), {"sample_rate": 16000})
        self.assertEqual([row["model"] for row in results], [row["key"] for row in models])
        self.assertEqual([row["text"] for row in results], [f"transcript-{row['key']}" for row in models])
        self.assertTrue(all(adapter.assert_audio is audio for adapter in adapters))
        self.assertEqual(
            [event[0] for event in events],
            ["load", "transcribe", "unload"] * 4,
        )
        self.assertTrue(all(event[-1] == "cuda:0" for event in events if event[0] == "load"))


if __name__ == "__main__":
    unittest.main()
