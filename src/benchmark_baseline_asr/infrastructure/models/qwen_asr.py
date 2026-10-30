from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.domain.entities import AudioInput, Transcription


class QwenASRAdapter:
    def __init__(self):
        self.model = None
        self.language = "Vietnamese"

    def load(self, model_path: str, model_config: dict, runtime_config: dict) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.language = model_config.get("language", "Vietnamese")
        self.model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float16 if runtime_config.get("dtype") == "float16" else torch.float32,
            device_map=runtime_config.get("device", "cuda:0"),
            max_inference_batch_size=1,
            max_new_tokens=model_config.get("max_new_tokens", 448),
            local_files_only=True,
        )
        self.model.model.generation_config.do_sample = runtime_config.get("do_sample", False)
        self.model.model.generation_config.num_beams = runtime_config.get("num_beams", 1)

    def transcribe(self, audio: AudioInput) -> Transcription:
        result = self.model.transcribe(
            audio=(audio.waveform, audio.sample_rate), language=self.language
        )[0]
        return Transcription(text=result.text.strip(), language=getattr(result, "language", "Vietnamese"))

    def unload(self) -> None:
        import gc
        import torch

        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
