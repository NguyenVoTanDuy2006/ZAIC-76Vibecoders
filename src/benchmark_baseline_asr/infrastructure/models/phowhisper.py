from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.domain.entities import AudioInput, Transcription


class PhoWhisperAdapter:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.language = "vi"
        self.max_new_tokens = 448
        self.do_sample = False
        self.num_beams = 1

    def load(self, model_path: str, model_config: dict, runtime_config: dict) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = runtime_config.get("device", "cuda:0")
        self.language = model_config.get("language", "vi")
        self.max_new_tokens = model_config.get("max_new_tokens", 448)
        self.do_sample = runtime_config.get("do_sample", False)
        self.num_beams = runtime_config.get("num_beams", 1)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if runtime_config.get("dtype") == "float16" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def transcribe(self, audio: AudioInput) -> Transcription:
        import torch
        inputs = self.processor(audio.waveform, sampling_rate=audio.sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                language=self.language,
                task="transcribe",
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return Transcription(text=text, language="Vietnamese")

    def unload(self) -> None:
        import gc
        import torch

        del self.model
        del self.processor
        self.model = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
