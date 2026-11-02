from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_waveform(
    path: str | Path,
    target_sample_rate: int = 16000,
    *,
    resample_nonmatching: bool = True,
) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_sample_rate:
        if not resample_nonmatching:
            raise ValueError(f"audio sample rate {sample_rate} does not match configured rate {target_sample_rate}")
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("librosa is required to resample audio not matching the configured sample rate") from exc
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    return np.asarray(waveform, dtype=np.float32), sample_rate
