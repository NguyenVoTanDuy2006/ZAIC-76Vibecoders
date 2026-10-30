from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .entities import AudioInput, Transcription


class ASRAdapter(Protocol):
    def load(self, model_path: str, model_config: dict, runtime_config: dict) -> None: ...
    def transcribe(self, audio: AudioInput) -> Transcription: ...
    def unload(self) -> None: ...
