import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch


class DistributedTrainerTests(unittest.TestCase):
    def test_real_qwen_two_rank_mixed_loss_with_zero_advantage_rank_and_padding(self):
        import transformers
        if not hasattr(transformers, "Qwen3ASRConfig"):
            self.skipTest("native Qwen3-ASR requires Transformers 5 environment")
        from transformers import (
            AutoProcessor,
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
        )
        from asr.finetuning.config import load_config

        processor_path = (
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
        )
        if not processor_path.exists():
            self.skipTest("offline Qwen processor not present")
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            data = root / "data"
            data.mkdir()
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
            for i in range(5):
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
            c = load_config()
            c["model"].update(model_id=str(base), dtype="float32")
            c["dataset"]["root"] = str(data)
            c["runtime"].update(device="cpu", local_files_only=True)
            c["train"].update(
                world_size=2,
                precision="fp32",
                epochs=1,
                num_workers=0,
                gradient_accumulation_steps=8,
            )
            c["decode"]["max_new_tokens"] = 2
            c["output"].update(root=str(root / "runs"), run_id="ddp")
            path = root / "config.json"
            path.write_text(json.dumps(c))
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(Path(__file__).with_name("trainer_worker.py")),
                    str(path),
                ],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                process.returncode, 0, process.stdout + "\n" + process.stderr
            )
            history = json.loads((root / "runs/ddp/history.json").read_text())
            self.assertEqual(
                history[0]["training"]["examples_including_sampler_padding"], 4
            )
            self.assertEqual(history[0]["training"]["nonzero_advantages"], 0.5)
            self.assertEqual(history[0]["validation"]["latest"]["samples"], 1)
            self.assertTrue(
                all(history[0]["training"]["nonzero_gradient_groups"].values())
            )


if __name__ == "__main__":
    unittest.main()
