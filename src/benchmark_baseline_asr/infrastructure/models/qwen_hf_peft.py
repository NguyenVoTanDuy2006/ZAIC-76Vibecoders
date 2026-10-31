"""Native Transformers Qwen3 ASR inference with exported PEFT adapters.

This backend requires Transformers 5 and must run separately from qwen-asr's
Transformers 4 environment. It also accepts a bare base for CPU integration tests.
"""
from __future__ import annotations

from benchmark_baseline_asr.domain.entities import AudioInput, Transcription
from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import adapter_digests


class QwenHFPEFTAdapter:
    def __init__(self):
        self.model = self.processor = None
        self.language = "Vietnamese"
        self.max_new_tokens = 768
        self.device = "cpu"
        self.dtype = None

    def load(self, model_path: str, model_config: dict, runtime_config: dict) -> None:
        import torch
        try:
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen HF adapters require the separate Transformers 5 environment; "
                               "install requirements/qwen-adapter-benchmark.txt") from exc

        self.device = runtime_config.get("device", "cuda:0")
        self.dtype = torch.float16 if runtime_config.get("dtype") == "float16" else torch.float32
        self.language = model_config.get("language", "Vietnamese")
        self.max_new_tokens = model_config.get("max_new_tokens", 768)
        checkpoint = model_config.get("adapter_path")
        if checkpoint and model_config.get("adapter_sha256") != adapter_digests(checkpoint):
            raise ValueError("adapter contents changed after preparation; select a new run_id")
        # Processor and model come from the same pinned local base snapshot.
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_path, dtype=self.dtype, local_files_only=True, attn_implementation="sdpa",
        ).to(self.device)
        if checkpoint:
            import warnings
            from peft import PeftModel

            # PEFT otherwise warns and continues with randomly initialized missing keys.
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*missing adapter keys.*")
                self.model = PeftModel.from_pretrained(self.model, checkpoint, is_trainable=False)
        self.model.eval()

    def transcribe(self, audio: AudioInput) -> Transcription:
        import torch

        if audio.sample_rate != 16000:
            raise ValueError("Qwen HF processor expects mono audio at 16000 Hz")
        inputs = self.processor.apply_transcription_request(audio=audio.waveform, language=self.language)
        inputs = {key: value.to(device=self.device, dtype=self.dtype) if torch.is_tensor(value) and value.is_floating_point()
                  else value.to(self.device) if torch.is_tensor(value) else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                             do_sample=False, num_beams=1)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(generated_ids, return_format="transcription_only")[0]
        return Transcription(text=text.strip(), language=self.language)

    def unload(self) -> None:
        import gc
        import torch

        self.model = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
