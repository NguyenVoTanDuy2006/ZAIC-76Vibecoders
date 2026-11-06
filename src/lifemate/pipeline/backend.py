"""Lazy local Qwen loading and deterministic generation, independent of pipeline stages."""
from __future__ import annotations

import time

from .config import AgentConfig, ContextLimitError, ModelLoadConfig, ModelReply


class HFBackend:
    """Local Transformers generation; uses no external inference API."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, messages, tools, config, remaining_seconds=None):
        import torch
        from transformers import GenerationConfig

        started = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        n_input = inputs["input_ids"].shape[-1]
        if n_input + config.max_new_tokens > config.max_context_tokens:
            raise ContextLimitError(f"{n_input} input + {config.max_new_tokens} output vượt context")
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        eos = self.model.generation_config.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos[0] if isinstance(eos, list) else eos
        gen = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=config.max_new_tokens,
            use_cache=True, cache_implementation="dynamic",
            prefill_chunk_size=config.prefill_chunk_size,
            eos_token_id=eos, pad_token_id=pad,
        )
        if remaining_seconds is not None:
            remaining = remaining_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("Hết ngân sách trước khi generate")
            gen.max_time = remaining  # cooperative, not a hard process deadline
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=gen)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ids = generated[0, n_input:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = len(ids) >= config.max_new_tokens and int(ids[-1]) not in eos_ids
        return ModelReply(text, n_input, len(ids), time.perf_counter() - started, truncated)


def load_qwen_backend(model_config=None, agent_config=None):
    """ĐÃ VIẾT: nạp tokenizer + Qwen FP16, trả backend dùng lại cho mọi mẫu.

    Gọi đúng một lần ở đầu chương trình. Chỉ cho phép tải qua mạng khi chủ động
    đặt local_files_only=False; bản chạy offline dùng thư mục/checkpoint có sẵn.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_config = model_config or ModelLoadConfig()
    agent_config = agent_config or AgentConfig()
    gpu_index = int(model_config.device.split(":")[1])
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"Không có GPU {model_config.device}; cần CUDA để nạp bản Qwen FP16 này")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_source, local_files_only=model_config.local_files_only,
        revision=model_config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_source, local_files_only=model_config.local_files_only,
        revision=model_config.revision,
        dtype=torch.float16, device_map={"": gpu_index}, attn_implementation="sdpa")
    model.eval()
    if not hasattr(model.generation_config, "prefill_chunk_size"):
        raise RuntimeError("Transformers chưa hỗ trợ prefill_chunk_size; kiểm tra phiên bản trước khi chạy")
    model.generation_config.prefill_chunk_size = agent_config.prefill_chunk_size
    model.generation_config.use_cache = True
    if sum(p.numel() for p in model.parameters()) >= 5_000_000_000:
        raise RuntimeError("Model được nạp vượt ngân sách LLM dưới 5B")
    return HFBackend(model, tokenizer)
