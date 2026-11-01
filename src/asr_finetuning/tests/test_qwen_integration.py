"""Real native Qwen graph on random tiny weights; optional cached processors."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class NativeQwenTests(unittest.TestCase):
    def test_real_pissa_assistant_loss_and_sample_replay_reach_encoder(self):
        try:
            from transformers import (
                AutoProcessor,
                Qwen3ASRConfig,
                Qwen3ASRForConditionalGeneration,
            )
            import peft  # noqa: F401 - dependency availability probe
        except ImportError:
            self.skipTest("requires isolated Transformers 5 / PEFT environment")
        path = Path(
            os.environ.get(
                "ZAIC_QWEN_PROCESSOR",
                str(
                    Path.home()
                    / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
                ),
            )
        )
        if not (path / "processor_config.json").exists():
            self.skipTest("set ZAIC_QWEN_PROCESSOR to offline processor")
        import numpy as np
        from zaic_pissa.config import load_config
        from zaic_pissa.model import TranscriptPolicy, supervised_batch
        from zaic_pissa.objective import sequence_log_probabilities, scst_loss
        from zaic_pissa.pissa import PiSSAManager

        torch.set_num_threads(2)
        torch.manual_seed(7)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        cfg = Qwen3ASRConfig(
            audio_config={
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 64,
                "d_model": 32,
                "output_dim": 32,
                "downsample_hidden_size": 8,
            },
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 151936,
            },
        )
        model = (
            PiSSAManager(load_config()["pissa"])
            .attach(Qwen3ASRForConditionalGeneration(cfg))
            .train()
        )
        model.config.use_cache = False
        item = {
            "waveform": np.sin(np.arange(16000, dtype=np.float32) * 0.02) * 0.1,
            "record": {"reference": "Xin chào."},
        }
        batch = supervised_batch(processor, item, "Vietnamese")
        ce = model(**batch).loss
        if isinstance(ce, dict):
            ce = ce["loss"]
        self.assertTrue(bool(torch.isfinite(ce)))
        ce.backward()
        model.zero_grad()
        policy = TranscriptPolicy(
            processor, model, {"language": "Vietnamese", "max_new_tokens": 4}
        )
        prompt = policy.prompt(item, "cpu", torch.float32)
        sampled_scores = []
        generate = model.generate

        def capture_distribution(*args, **kwargs):
            result = generate(
                *args, **kwargs, return_dict_in_generate=True, output_scores=True
            )
            sampled_scores.extend(result.scores)
            return result.sequences

        with patch.object(model, "generate", side_effect=capture_distribution):
            tokens, text = policy.generate(model, prompt, sampled=True)
        self.assertTrue(model.training)
        output = model(**policy.replay(prompt, tokens))
        logp, mask = sequence_log_probabilities(
            output.logits,
            prompt["input_ids"].shape[-1],
            tokens,
            policy.eos_ids,
            policy.forbidden_ids,
        )
        rollout_logp = (
            torch.stack(sampled_scores, dim=1)
            .float()
            .log_softmax(-1)
            .gather(-1, tokens.unsqueeze(-1))
            .squeeze(-1)
        )
        rollout_logp = torch.where(mask, rollout_logp, 0.0).sum(-1)
        self.assertTrue(
            torch.allclose(logp.detach(), rollout_logp, atol=1e-5, rtol=1e-5),
            f"rollout {rollout_logp} differs from replay {logp.detach()}",
        )
        loss = scst_loss(logp, torch.tensor([0.8]), torch.tensor([0.6]))
        loss.backward()
        for group in ["audio_tower", "language_model"]:
            grads = [
                p.grad
                for n, p in model.named_parameters()
                if group in n and "lora_" in n and p.grad is not None
            ]
            self.assertTrue(grads, group)
            self.assertTrue(any(bool((g != 0).any()) for g in grads), group)
        self.assertTrue(bool(torch.isfinite(logp).all()))

    def test_real_phobert_scoring_is_frozen_and_handles_long_text(self):
        try:
            import bert_score  # noqa: F401 - dependency availability probe
            import pyvi  # noqa: F401 - dependency availability probe
        except ImportError:
            self.skipTest("requires bert-score and pyvi")
        path = Path(
            os.environ.get(
                "ZAIC_PHOBERT_PATH",
                "artifacts/asr_benchmark/models/semantic_vinai--phobert-base",
            )
        )
        if not (path / "config.json").exists():
            self.skipTest("set ZAIC_PHOBERT_PATH to offline PhoBERT")
        from zaic_pissa.config import load_config
        from zaic_pissa.semantic import PhoBERTReward

        torch.set_num_threads(2)
        scorer = PhoBERTReward(load_config()["semantic"], "cpu", path)
        scores = scorer.score_pairs(
            [
                ("Bật đèn phòng khách.", "bật đèn phòng khách"),
                ("Bật đèn phòng khách.", ""),
                ("Bật đèn.", "Bật đèn. " * 180),
            ]
        )
        self.assertAlmostEqual(scores[0].f1, 1.0, places=5)
        self.assertEqual(scores[1].f1, 0.0)
        self.assertTrue(scores[2].chunked)
        self.assertFalse(
            any(
                p.requires_grad or p.grad is not None for p in scorer.model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
