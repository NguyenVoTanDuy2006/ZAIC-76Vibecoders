"""Read Research-style model configs pointing directly to mounted PEFT exports."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .peft_checkpoints import adapter_digests


def load_adapter_model_configs(paths: list[str | Path]) -> list[dict]:
    models = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        document = json.loads(path.read_text(encoding="utf-8"))
        spec = document.get("model", {})
        key = spec.get("key", "")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ValueError(f"model.key must be a simple unique name: {path}")
        if spec.get("backend") != "qwen3_asr" or spec.get("model_id") != "Qwen/Qwen3-ASR-1.7B-hf":
            raise ValueError(f"unsupported Qwen backend or base model: {path}")
        revision = spec.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"model.revision must pin the pretrained base commit: {path}")
        if spec.get("dtype", "float16") not in {"float16", "float32"} or spec.get("trust_remote_code", False):
            raise ValueError(f"use float16/float32 and trust_remote_code=false: {path}")
        adapter_path = spec.get("adapter_path")
        if not isinstance(adapter_path, str) or not adapter_path.strip():
            raise ValueError(f"model.adapter_path must point directly to the adapter export: {path}")
        checkpoint = Path(adapter_path).expanduser()
        checkpoint = (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        digests = adapter_digests(checkpoint)
        config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("base_model_name_or_path") != spec["model_id"]:
            raise ValueError(f"adapter base model differs from configured model_id: {checkpoint}")
        if config.get("revision") and config["revision"] != revision:
            raise ValueError(f"adapter base revision differs from configured revision: {checkpoint}")
        kind = config.get("peft_type")
        if kind == "ADALORA":
            method = "adalora"
            if not config.get("rank_pattern"):
                raise ValueError(f"missing trained AdaLoRA rank_pattern: {checkpoint}")
        elif kind == "LORA":
            method = "pissa"
            # Converted PiSSA is a standard LoRA delta on the original base.
            # The mounted export need not carry training-run metadata.
            if config.get("init_lora_weights") is not True:
                raise ValueError(f"PiSSA must be converted to standard LoRA on the original base: {checkpoint}")
            metadata_path = checkpoint / "training_method.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (metadata.get("export") != "converted_standard_lora"
                        or metadata.get("load_on") != "original_pretrained_base"):
                    raise ValueError(f"PiSSA must be converted_standard_lora: {checkpoint}")
        else:
            raise ValueError(f"expected LORA/PiSSA or ADALORA export: {checkpoint}")
        models.append({"key": key, "settings": {
            "adapter": "qwen_hf_peft", "repo_id": spec["model_id"], "revision": revision,
            "adapter_path": str(checkpoint), "adapter_sha256": digests, "method": method,
            "display_name": spec.get("display_name", key), "dtype": spec.get("dtype", "float16"),
            "epoch": None, "language": "Vietnamese", "max_new_tokens": 768,
        }})
    if len(models) != 2 or {row["settings"]["method"] for row in models} != {"pissa", "adalora"}:
        raise ValueError("select one PiSSA and one AdaLoRA model config")
    if len({row["key"] for row in models}) != 2:
        raise ValueError("model config keys must be unique")
    identities = {(r["settings"]["repo_id"], r["settings"]["revision"], r["settings"]["dtype"]) for r in models}
    if len(identities) != 1:
        raise ValueError("both adapters must share the same base model, revision and dtype")
    return models
