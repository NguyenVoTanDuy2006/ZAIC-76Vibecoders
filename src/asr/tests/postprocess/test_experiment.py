from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from asr.finetuning.artifacts import read_jsonl
from asr.postprocess.checks import hard_check
from asr.postprocess.evaluation import score_variant
from asr.postprocess.experiment import VARIANTS, run_experiment
from asr.postprocess.generation import (
    CORRECTION_TOKENS, JUDGE_TOKENS, MAX_CONTEXT_TOKENS, MAX_PROMPT_TOKENS,
    PromptTooLong, fit_prompt,
)


class FakeLM:
    def __init__(self):
        self.calls = []

    def count_prompt(self, messages):
        return len(json.dumps(messages, ensure_ascii=False)) // 3

    def chat(self, messages, max_new_tokens):
        payload = json.loads(messages[-1]["content"])
        self.calls.append((payload, max_new_tokens))
        serialized = json.dumps(payload, ensure_ascii=False)
        if "VALIDATION_SECRET" in serialized or "reference" in payload:
            raise AssertionError("validation reference leaked into LM input")
        if "proposed_text" in payload:
            return json.dumps({"decision": "accept", "reason": "Sửa một từ có ngữ cảnh rõ"})
        original = payload["asr_text"]
        return json.dumps({"corrected_text": original.replace("Tết", "Tắt")}, ensure_ascii=False)


class FakeASR:
    def load(self, *_args):
        pass

    def transcribe(self, audio):
        texts = {
            "train_1": "Bật đèn phòng khách cho tôi ngay bây giờ nhé",
            "train_2": "Tết đèn tầng hai giúp tôi ngay bây giờ nhé",
            "train_3": "Đặt báo thức sáu giờ sáng ngày nay cho tôi",
        }
        return SimpleNamespace(text=texts[audio.audio_path.stem])

    def unload(self):
        pass


class PostprocessExperimentTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[4] / ".test-scratch"
        self.scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="postprocess-", dir=self.scratch))

    def tearDown(self):
        shutil.rmtree(self.root)
        try:
            self.scratch.rmdir()
        except OSError:
            pass

    def _write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")

    def _fixture(self):
        source = [
            {"sample_id": "val_1_history_000", "conversation_id": "val_1",
             "is_latest": False, "prediction": "Tết đèn tầng hai giúp tôi ngay bây giờ nhé",
             "reference": "Tắt đèn tầng hai giúp tôi ngay bây giờ nhé VALIDATION_SECRET"},
            {"sample_id": "val_1", "conversation_id": "val_1", "is_latest": True,
             "prediction": "Tết hết đèn tầng hai giúp tôi nhé",
             "reference": "Tắt hết đèn tầng hai giúp tôi nhé VALIDATION_SECRET"},
            {"sample_id": "val_2", "conversation_id": "val_2", "is_latest": True,
             "prediction": "Nhắc tôi gọi cho Mai vào tám giờ tối nhé",
             "reference": "Nhắc tôi gọi cho Mai vào tám giờ tối nhé VALIDATION_SECRET"},
        ]
        train = [
            {"sample_id": "train_1", "conversation_id": "train_1", "is_latest": True,
             "audio": "audio/train_1.wav",
             "reference": "Bật đèn phòng khách cho tôi ngay bây giờ nhé"},
            {"sample_id": "train_2", "conversation_id": "train_2", "is_latest": True,
             "audio": "audio/train_2.wav",
             "reference": "Tắt đèn tầng hai giúp tôi ngay bây giờ nhé"},
            {"sample_id": "train_3", "conversation_id": "train_3", "is_latest": True,
             "audio": "audio/train_3.wav",
             "reference": "Đặt báo thức sáu giờ sáng ngày mai cho tôi"},
        ]
        predictions = self.root / "input/predictions.jsonl"
        manifest = self.root / "input/train.jsonl"
        self._write_jsonl(predictions, source)
        self._write_jsonl(manifest, train)
        for row in train:
            audio = self.root / "train_audio" / row["audio"]
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.touch()
        asr_config = self.root / "input/config.json"
        asr_config.write_text(json.dumps({"model": {
            "backend": "qwen3_asr", "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
            "revision": "test", "dtype": "float16"}}), encoding="utf-8")
        asr_model = self.root / "asr_model"
        asr_model.mkdir()
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "init_lora_weights": True, "revision": "test"}), encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"fake")
        lm_model = self.root / "lm_model"
        lm_model.mkdir()
        (lm_model / "config.json").write_text("{}", encoding="utf-8")
        return {"predictions_path": predictions, "train_manifest": manifest,
                "train_audio_root": self.root / "train_audio", "asr_config": asr_config,
                "asr_model_path": asr_model, "adapter_path": adapter,
                "lm_model_path": lm_model, "output_dir": self.root / "result",
                "support_pool_size": 3, "expected_samples": 3, "expected_latest": 2,
                "asr_factory": FakeASR,
                "audio_loader": lambda path, _config: SimpleNamespace(audio_path=path)}

    def test_all_seven_variants_and_resume_without_model_reload(self):
        config = self._fixture()
        lm = FakeLM()
        first = run_experiment(**config, lm_factory=lambda: lm)
        self.assertEqual(set(first), set(VARIANTS))
        self.assertEqual(len(lm.calls), 10)  # 6 generation + 4 changed candidates judged
        for name in VARIANTS:
            rows = read_jsonl(config["output_dir"] / "predictions" / f"{name}.jsonl")
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["sample_id"] for row in rows},
                             {"val_1_history_000", "val_1", "val_2"})
        audit = read_jsonl(config["output_dir"] / "audit.jsonl")
        self.assertEqual(len(audit), 6)
        self.assertTrue((config["output_dir"] / "comparison.csv").is_file())
        selection = json.loads((config["output_dir"] / "selection.json").read_text())
        self.assertEqual(selection["metric"], "all.normalized_wer")
        self.assertIn(selection["best_variant"], VARIANTS)
        second = run_experiment(**config, lm_factory=lambda: self.fail("LM reloaded"))
        self.assertEqual(first, second)

    def test_interrupted_generation_resumes_from_cached_sample(self):
        config = self._fixture()

        class InterruptedLM(FakeLM):
            def chat(self, messages, max_new_tokens):
                if self.calls:
                    raise RuntimeError("simulated Kaggle interruption")
                return super().chat(messages, max_new_tokens)

        with self.assertRaisesRegex(RuntimeError, "simulated Kaggle interruption"):
            run_experiment(**config, lm_factory=InterruptedLM)
        cache = config["output_dir"] / "generation/zero_shot.jsonl"
        self.assertEqual(len(read_jsonl(cache)), 1)
        resumed = run_experiment(**config, lm_factory=FakeLM)
        self.assertEqual(len(resumed), 7)
        self.assertEqual(len(read_jsonl(cache)), 3)

    def test_hard_check_blocks_critical_changes_but_allows_small_correction(self):
        self.assertEqual(hard_check("Ơi, giờ lịch cà phê nhé", "Ơi, dời lịch cà phê nhé", [])[0], True)
        self.assertEqual(hard_check("Tôi đã về. Tết hết đèn tầng hai nhé",
                                    "Tôi đã về. Tắt hết đèn tầng hai nhé", [])[0], True)
        self.assertEqual(hard_check("Chuyển 50 nghìn cho Mai", "Chuyển 500 nghìn cho Mai", [])[1],
                         "number_or_time_changed")
        self.assertEqual(hard_check("Nếu mưa thì không đi", "Nếu mưa thì đi", [])[1],
                         "logic_changed")

    def test_token_budget_trims_oldest_history_and_preserves_current_text(self):
        class CountingLM:
            def count_prompt(self, messages):
                return 16_001 if len(json.loads(messages[-1]["content"])["prior_user_asr"]) > 1 else 16_000

        payload = {"asr_text": "câu hiện tại", "prior_user_asr": [
            {"history_index": 0, "text": "cũ"}, {"history_index": 2, "text": "mới"}]}
        messages = fit_prompt(CountingLM(), "instructions", payload)
        fitted = json.loads(messages[-1]["content"])
        self.assertEqual(fitted["asr_text"], "câu hiện tại")
        self.assertEqual(fitted["prior_user_asr"], [{"history_index": 2, "text": "mới"}])
        self.assertEqual(MAX_PROMPT_TOKENS + CORRECTION_TOKENS <= MAX_CONTEXT_TOKENS, True)
        self.assertEqual(MAX_PROMPT_TOKENS + JUDGE_TOKENS <= MAX_CONTEXT_TOKENS, True)
        with self.assertRaises(PromptTooLong):
            fit_prompt(SimpleNamespace(count_prompt=lambda _messages: 16_001),
                       "instructions", {"asr_text": "giữ nguyên", "prior_user_asr": []})

    def test_baseline_metric_matches_real_validation_summary(self):
        repo = Path(__file__).resolve().parents[4]
        run = repo / "output/zaic-qwen-pissa/zaic-pissa-scst-v1/validation/epoch-03"
        if not (run / "predictions.jsonl").is_file():
            self.skipTest("local validation artifact unavailable")
        source = read_jsonl(run / "predictions.jsonl")
        scores = score_variant(source, [{"sample_id": row["sample_id"],
                                         "prediction": row["prediction"]} for row in source])
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        for group in ("all", "latest"):
            for metric in ("raw_cer", "normalized_cer", "raw_wer", "normalized_wer"):
                self.assertAlmostEqual(scores[group][metric], summary[group][metric])


if __name__ == "__main__":
    unittest.main()
