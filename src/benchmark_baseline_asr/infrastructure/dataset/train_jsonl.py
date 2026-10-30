from __future__ import annotations

import json
import wave
from pathlib import Path

from benchmark_baseline_asr.domain.entities import DatasetSample
from benchmark_baseline_asr.domain.errors import DatasetError


def load_train_manifest(jsonl_path: str | Path, audio_root: str | Path) -> list[DatasetSample]:
    jsonl_path, audio_root = Path(jsonl_path), Path(audio_root).resolve()
    samples, seen = [], set()
    try:
        stream = jsonl_path.open(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"cannot open {jsonl_path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"invalid JSON at line {line_number}: {exc}") from exc
            sample_id, audio_rel, reference = row.get("id"), row.get("audio"), row.get("audio_text")
            if not isinstance(sample_id, str) or not sample_id:
                raise DatasetError(f"missing id at line {line_number}")
            if sample_id in seen:
                raise DatasetError(f"duplicate sample id {sample_id!r}")
            if not isinstance(audio_rel, str) or not isinstance(reference, str):
                raise DatasetError(f"sample {sample_id} must have string audio and audio_text")
            audio_path = (audio_root / audio_rel).resolve()
            if audio_root not in audio_path.parents:
                raise DatasetError(f"audio path escapes root for sample {sample_id}")
            if not audio_path.is_file():
                raise DatasetError(f"missing audio for sample {sample_id}: {audio_path}")
            try:
                with wave.open(str(audio_path), "rb") as wav:
                    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                        raise DatasetError(f"sample {sample_id} is not mono 16-bit 16kHz WAV")
                    duration = wav.getnframes() / wav.getframerate()
            except (wave.Error, OSError) as exc:
                raise DatasetError(f"invalid WAV for sample {sample_id}: {exc}") from exc
            samples.append(DatasetSample(sample_id, audio_path, reference, duration))
            seen.add(sample_id)
    return samples
