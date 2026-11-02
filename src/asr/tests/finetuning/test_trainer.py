import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from asr.finetuning.config import load_config
from asr.finetuning.semantic import SemanticScore


class FixedReward:
    """Deterministic frozen reward for testing orchestration without downloading PhoBERT."""

    def score_pairs(self, pairs, normalized=True):
        return [SemanticScore(0.5 if hypothesis else 0.0) for _, hypothesis in pairs]


class TrainerTests(unittest.TestCase):
    def test_native_qwen_epoch_commit_resume_and_heldout_coverage(self):
        import transformers
        if not hasattr(transformers, "Qwen3ASRConfig"):
            self.skipTest("native Qwen3-ASR requires Transformers 5 environment")
        from transformers import (
            AutoProcessor,
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
        )
        from asr.finetuning.trainer import Trainer
        from asr.finetuning.runtime import DistributedRuntime

        processor_path = (
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
        )
        if not processor_path.exists():
            self.skipTest("offline Qwen processor not present")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            data = root / "data"
            data.mkdir()
            torch.manual_seed(18)
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
            Qwen3ASRForConditionalGeneration(cfg).save_pretrained(base)
            AutoProcessor.from_pretrained(
                processor_path, local_files_only=True
            ).save_pretrained(base)
            rows = []
            for i in range(4):
                name = f"{i}.wav"
                with wave.open(str(data / name), "wb") as stream:
                    stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    stream.writeframes(
                        (np.sin(np.arange(8000) * 0.02) * (1000 + i))
                        .astype("<i2")
                        .tobytes()
                    )
                rows.append({"id": f"c{i}", "audio": name, "audio_text": "Bật đèn."})
            (data / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            config = load_config()
            config["model"].update(model_id=str(base), dtype="float32")
            config["dataset"]["root"] = str(data)
            config["runtime"].update(device="cpu", local_files_only=True)
            config["train"].update(
                world_size=1,
                precision="fp32",
                epochs=2,
                num_workers=0,
                gradient_accumulation_steps=3,
            )
            config["decode"]["max_new_tokens"] = 2
            config["output"].update(root=str(root / "runs"), run_id="test")
            runtime = DistributedRuntime("cpu").initialize(config)

            class Interrupted(Trainer):
                def save_epoch(self, *args, **kwargs):
                    super().save_epoch(*args, **kwargs)
                    raise InterruptedError("simulated interruption after durable epoch")

            first = Interrupted(
                config, runtime, reward_factory=lambda *_: FixedReward()
            )
            with self.assertRaisesRegex(InterruptedError, "durable epoch"):
                first.train()
            first.release()
            run = root / "runs/test"
            state = torch.load(
                run / "checkpoints/latest.pt", weights_only=False, map_location="cpu"
            )
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["progress"]["successful_updates"], 1)
            second = Trainer(config, runtime, reward_factory=lambda *_: FixedReward())
            second.train(resume=True)
            second.release()
            state = torch.load(
                run / "checkpoints/latest.pt", weights_only=False, map_location="cpu"
            )
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["progress"]["successful_updates"], 2)
            self.assertEqual(len(state["history"]), 2)
            self.assertTrue(
                (run / "adapters/epoch-01/adapter_model.safetensors").exists()
            )
            self.assertTrue(
                (run / "adapters/epoch-02/adapter_model.safetensors").exists()
            )
            final = Trainer(config, runtime, reward_factory=lambda *_: FixedReward())
            report = final.evaluate_test()
            final.release()
            for name in ["base", "best"]:
                self.assertEqual(report[name]["latest"]["samples"], 1)
                self.assertEqual(report[name]["all"]["coverage"], 1.0)
            self.assertTrue((run / "best_adapter/adapter_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
