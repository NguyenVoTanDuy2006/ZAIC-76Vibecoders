"""Offline CPU API smoke test; set QWEN_TEST_PROCESSOR_PATH to a cached processor.

Uses tiny random Qwen weights, not a substitute for benchmarking the real model.
Run in the dedicated requirements/qwen-adapter-benchmark.txt environment.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
import warnings
from pathlib import Path


@unittest.skipUnless(os.environ.get("QWEN_TEST_PROCESSOR_PATH") and importlib.util.find_spec("peft"),
                     "needs PEFT and QWEN_TEST_PROCESSOR_PATH (offline processor snapshot)")
class QwenHFPEFTIntegrationTests(unittest.TestCase):
    def test_export_reload_and_transcribe_lora_and_compact_adalora(self):
        import numpy as np
        import torch
        from peft import AdaLoraConfig, LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
        from benchmark_baseline_asr.domain.entities import AudioInput
        from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import adapter_digests
        from benchmark_baseline_asr.infrastructure.models.qwen_hf_peft import QwenHFPEFTAdapter

        torch.set_num_threads(2)
        processor = AutoProcessor.from_pretrained(os.environ["QWEN_TEST_PROCESSOR_PATH"], local_files_only=True)
        config = Qwen3ASRConfig(
            audio_config={"encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": 64,
                          "d_model": 32, "output_dim": 32, "downsample_hidden_size": 8},
            text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                         "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
                         "vocab_size": 151936},
        )
        for kind in ("lora", "adalora"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                torch.manual_seed(7)
                base = Qwen3ASRForConditionalGeneration(config).eval()
                base.save_pretrained(root / "base")
                processor.save_pretrained(root / "base")
                if kind == "adalora":
                    peft_config = AdaLoraConfig(init_r=2, target_r=1, total_step=10,
                                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
                else:
                    peft_config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
                adapted = get_peft_model(base, peft_config)
                with torch.no_grad():
                    for name, param in adapted.named_parameters():
                        if "lora_B" in name:
                            param.fill_(0.1)
                        if "lora_E" in name:
                            param[0] = 1.0
                            param[1] = 0.0
                if kind == "adalora":
                    names = [name for name, _ in adapted.base_model.model.named_parameters() if "lora_E" in name]
                    # The supplied full checkpoint also has projections with rank zero.
                    peft_config.rank_pattern = {name: [index != 0, False] for index, name in enumerate(names)}
                with warnings.catch_warnings():
                    # PEFT 0.21.1 mistakes an intentionally pruned AdaLoRA rank zero
                    # for an ungathered distributed tensor. Reload/shape checks below
                    # verify these are valid compact weights.
                    warnings.filterwarnings("ignore", message="Adapter 'default': .*zero-sized.*")
                    adapted.save_pretrained(root / "adapter")
                adapter = QwenHFPEFTAdapter()
                with warnings.catch_warnings():
                    # Generic LoRA pattern validation doesn't recognize AdaLoRA's
                    # tensor-level keys. Its dedicated loader applies the ranks;
                    # checking the loaded rank dimensions below proves this.
                    warnings.filterwarnings("ignore", message="The following rank_pattern keys did not match.*")
                    adapter.load(str(root / "base"), {
                        "adapter_path": str(root / "adapter"), "adapter_sha256": adapter_digests(root / "adapter"),
                        "max_new_tokens": 2,
                    }, {"device": "cpu", "dtype": "float32"})
                # A no-op / omitted adapter would fail this check.
                loaded_b = [p for name, p in adapter.model.named_parameters() if "lora_B" in name]
                self.assertTrue(loaded_b)
                self.assertTrue(all(torch.allclose(p, torch.full_like(p, 0.1)) for p in loaded_b))
                if kind == "adalora":
                    self.assertEqual({p.shape[-1] for p in loaded_b}, {0, 1})
                waveform = np.sin(np.arange(16000, dtype=np.float32) * 0.02) * 0.1
                result = adapter.transcribe(AudioInput(root / "audio.wav", waveform, 16000))
                self.assertIsInstance(result.text, str)
                self.assertEqual(result.language, "Vietnamese")
                adapter.unload()


if __name__ == "__main__":
    unittest.main()
