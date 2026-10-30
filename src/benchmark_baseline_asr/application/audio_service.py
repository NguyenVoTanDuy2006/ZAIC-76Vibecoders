from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.domain.entities import AudioInput
from benchmark_baseline_asr.infrastructure.audio.waveform import load_waveform


def prepare_audio(audio_path: str | Path, audio_config: dict) -> AudioInput:
    path = Path(audio_path)
    waveform, sample_rate = load_waveform(
        path,
        target_sample_rate=audio_config.get("sample_rate", 16000),
        resample_nonmatching=audio_config.get("resample_nonmatching_audio", True),
    )
    return AudioInput(path, waveform, sample_rate)
