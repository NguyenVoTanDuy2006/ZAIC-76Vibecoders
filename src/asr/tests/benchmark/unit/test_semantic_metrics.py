import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from asr.benchmark.application.evaluation_service import evaluate_predictions
from asr.benchmark.application.reporting_service import build_comparison
from asr.benchmark.infrastructure.evaluation.semantic_metrics import PhoBERTBERTScore
from asr.benchmark.infrastructure.models.parameter_count import count_model_parameters
from asr.benchmark.infrastructure.storage.artifacts import read_jsonl, write_json, write_jsonl


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class SemanticMetricTests(unittest.TestCase):
    def test_bertscore_pairs_preserve_order_and_handle_empty_inputs(self):
        scorer = PhoBERTBERTScore.__new__(PhoBERTBERTScore)
        scorer.config = {"batch_size": 2, "word_segment": False}
        scorer.scorer = mock.Mock()
        scorer.scorer.score.return_value = (None, None, _FakeTensor([0.7, 0.8]))
        values = scorer.score_pairs([("a", "b"), ("", ""), ("x", ""), ("c", "d")])
        self.assertEqual(values, [0.7, 1.0, 0.0, 0.8])
        scorer.scorer.score.assert_called_once_with(["b", "d"], ["a", "c"], batch_size=2)

    def test_evaluation_adds_semantic_scores_and_actual_parameter_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / "manifest.jsonl", [
                {"sample_id": "a", "reference": "Bật đèn", "duration_seconds": 1},
                {"sample_id": "b", "reference": "Mở cửa", "duration_seconds": 1},
            ])
            write_jsonl(root / "predictions.jsonl", [
                {"sample_id": "a", "text": "Bật đèn", "latency_seconds": 0.1},
                {"sample_id": "b", "text": "Mở cửa", "latency_seconds": 0.1},
            ])
            write_json(root / "run_timing.json", {"wall_seconds": 1, "workers": [
                {"parameter_count": 123}, {"parameter_count": 123},
            ]})
            semantic = mock.Mock(model_parameter_count=456)
            semantic.score_pairs.side_effect = [[0.9, 0.8], [0.7, 0.6]]
            summary = evaluate_predictions(
                root / "manifest.jsonl", root / "predictions.jsonl", root,
                semantic_scorer=semantic,
                semantic_config={"model_id": "vinai/phobert-base"},
            )
            self.assertEqual(summary["parameter_count"], 123)
            self.assertEqual(summary["semantic_model_parameter_count"], 456)
            self.assertAlmostEqual(summary["raw_bertscore_f1"], 0.85)
            self.assertEqual(read_jsonl(root / "metrics.jsonl")[1]["normalized_bertscore_f1"], 0.6)

    def test_parameter_counter_excludes_inference_wrapper(self):
        parameter = mock.Mock()
        parameter.numel.return_value = 42
        model = mock.Mock()
        model.parameters.return_value = [parameter]
        adapter = SimpleNamespace(model=SimpleNamespace(model=model))
        self.assertEqual(count_model_parameters(adapter), 42)

    def test_comparison_exports_semantic_score_and_parameter_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "qwen" / "summary.json", {
                "parameter_count": 1700000000, "raw_bertscore_f1": 0.91,
                "normalized_bertscore_f1": 0.93,
            })
            build_comparison(root, ["qwen"])
            self.assertIn("parameter_count", (root / "comparison.csv").read_text(encoding="utf-8"))
            self.assertIn("normalized_bertscore_f1", (root / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
