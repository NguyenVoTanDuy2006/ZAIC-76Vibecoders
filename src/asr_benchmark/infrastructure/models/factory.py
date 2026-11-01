from __future__ import annotations

from .phowhisper import PhoWhisperAdapter
from .qwen_asr import QwenASRAdapter
from .qwen_hf_peft import QwenHFPEFTAdapter


def create_adapter(adapter_name: str):
    if adapter_name == "phowhisper":
        return PhoWhisperAdapter()
    if adapter_name == "qwen_asr":
        return QwenASRAdapter()
    if adapter_name == "qwen_hf_peft":
        return QwenHFPEFTAdapter()
    raise ValueError(f"unsupported ASR adapter: {adapter_name}")
