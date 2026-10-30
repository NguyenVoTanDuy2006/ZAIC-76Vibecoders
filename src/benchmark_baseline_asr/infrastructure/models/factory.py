from __future__ import annotations

from .phowhisper import PhoWhisperAdapter
from .qwen_asr import QwenASRAdapter


def create_adapter(adapter_name: str):
    if adapter_name == "phowhisper":
        return PhoWhisperAdapter()
    if adapter_name == "qwen_asr":
        return QwenASRAdapter()
    raise ValueError(f"unsupported ASR adapter: {adapter_name}")
