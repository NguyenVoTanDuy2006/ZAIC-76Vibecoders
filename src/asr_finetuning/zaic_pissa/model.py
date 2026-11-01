from __future__ import annotations

from pathlib import Path

import torch


def snapshot(spec, config):
    from huggingface_hub import snapshot_download

    local = Path(spec["model_id"]).expanduser()
    if local.is_dir():
        return local.resolve()
    return Path(
        snapshot_download(
            spec["model_id"],
            revision=spec["revision"],
            cache_dir=config["runtime"]["cache_dir"],
            local_files_only=config["runtime"]["local_files_only"],
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.codes",
                "*.jinja",
                "*.safetensors",
                "pytorch_model.bin",
            ],
        )
    )


def load_base(config, runtime):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    path = snapshot(config["model"], config)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    dtype = torch.float16 if config["model"]["dtype"] == "float16" else torch.float32
    model = (
        Qwen3ASRForConditionalGeneration.from_pretrained(
            path, dtype=dtype, local_files_only=True, attn_implementation="sdpa"
        )
        .to(runtime.device)
        .eval()
    )
    return model, processor


def move_inputs(batch, device, dtype):
    return {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        if torch.is_tensor(v)
        else v
        for k, v in batch.items()
    }


def supervised_batch(processor, item, language):
    messages = [
        [
            {"role": "user", "content": [{"type": "audio", "audio": item["waveform"]}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"language {language}<asr_text>{item['record']['reference'].strip()}",
                    }
                ],
            },
        ]
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        processor_kwargs={
            "output_labels": True,
            "sampling_rate": 16000,
            "padding": "longest",
            "truncation": False,
        },
    )
    labels = batch.get("labels")
    if (
        not torch.is_tensor(labels)
        or labels.ndim != 2
        or not (labels != -100).any(dim=-1).all()
    ):
        raise ValueError("each sample must have supervised assistant labels")
    if ((labels != -100) & (batch["attention_mask"] == 0)).any():
        raise ValueError("padding was included in labels")
    return batch


class TranscriptPolicy:
    def __init__(self, processor, model, decode):
        self.processor, self.decode = processor, decode
        raw = model.generation_config.eos_token_id or model.config.eos_token_id
        self.eos_ids = [raw] if isinstance(raw, int) else list(raw)
        tokenizer = processor.tokenizer
        added = tokenizer.get_added_vocab().values()
        specials = set(tokenizer.all_special_ids) | set(added)
        for key in ["audio_token_id", "timestamp_token_id"]:
            value = getattr(model.config, key, None)
            if value is not None:
                specials.add(value)
        self.forbidden_ids = sorted(specials - set(self.eos_ids))
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else self.eos_ids[0]
        )

    def prompt(self, item, device, dtype):
        batch = self.processor.apply_transcription_request(
            audio=item["waveform"], language=self.decode["language"]
        )
        return move_inputs(batch, device, dtype)

    def generate(self, model, prompt, *, sampled=False):
        from transformers import GenerationConfig

        generation = GenerationConfig(
            max_new_tokens=self.decode["max_new_tokens"],
            do_sample=sampled,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
            suppress_tokens=self.forbidden_ids,
            use_cache=True,
        )
        training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                output = model.generate(**prompt, generation_config=generation)
        finally:
            model.train(training)
        # Inference-mode tensors must be cloned outside that context before replay.
        tokens = output[:, prompt["input_ids"].shape[-1] :].clone()
        if tokens.shape[-1] < 1:
            raise ValueError("generation returned no token (including EOS)")
        text = self.processor.decode(tokens, return_format="transcription_only")
        if isinstance(text, (list, tuple)):
            text = text[0]
        return tokens, str(text).strip()

    def replay(self, prompt, tokens):
        batch = dict(prompt)
        batch["input_ids"] = torch.cat([prompt["input_ids"], tokens], dim=-1)
        batch["attention_mask"] = torch.cat(
            [prompt["attention_mask"], torch.ones_like(tokens)], dim=-1
        )
        batch["use_cache"] = False
        return batch
