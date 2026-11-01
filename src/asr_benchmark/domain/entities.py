from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    audio_path: Path
    reference: str
    duration_seconds: float


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str | None = None


@dataclass(frozen=True)
class AudioInput:
    audio_path: Path
    waveform: Any
    sample_rate: int


@dataclass(frozen=True)
class ErrorRates:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        if self.reference_units == 0:
            return 0.0 if not self.errors else None
        return self.errors / self.reference_units
