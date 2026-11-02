"""Resolve exported adapters without guessing epochs or their pretrained base."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_digests(checkpoint: str | Path) -> dict[str, str]:
    checkpoint = Path(checkpoint)
    result = {}
    for name in ("adapter_config.json", "adapter_model.safetensors", "training_method.json"):
        path = checkpoint / name
        if path.is_file():
            with path.open("rb") as stream:
                result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not {"adapter_config.json", "adapter_model.safetensors"} <= result.keys():
        raise ValueError(f"missing adapter_config.json or adapter_model.safetensors: {checkpoint}")
    return result


def discover_full_adapters(models_root: str | Path) -> list[dict]:
    """Select best exported checkpoints from the two full training runs."""
    models = []
    for method, expected_type in (("pissa", "LORA"), ("adalora", "ADALORA")):
        key = f"qwen_{method}_full"
        run = Path(models_root).expanduser().resolve() / key
        try:
            epoch = int((run / "best_epoch.txt").read_text(encoding="utf-8").strip())
            if epoch < 1:
                raise ValueError("epoch must be positive")
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid or missing best_epoch.txt: {run}") from exc
        checkpoint = run / f"epoch_{epoch}_adapter"
        digests = adapter_digests(checkpoint)
        config = _read_json(checkpoint / "adapter_config.json")
        if config.get("peft_type") != expected_type:
            raise ValueError(f"{key} must contain a {expected_type} adapter")
        if method == "pissa":
            metadata_path = checkpoint / "training_method.json"
            metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
            if (metadata.get("export") != "converted_standard_lora"
                    or metadata.get("load_on") != "original_pretrained_base"
                    or config.get("init_lora_weights") is not True):
                raise ValueError("PiSSA must be exported as converted_standard_lora for original_pretrained_base")
        if method == "adalora" and not config.get("rank_pattern"):
            raise ValueError(f"missing trained AdaLoRA rank_pattern: {checkpoint}")
        revisions = {config["revision"]} if config.get("revision") else set()
        for filename in ("run_config.json", "resolved_config.json"):
            path = run / filename
            if path.is_file():
                metadata = _read_json(path)
                revision = metadata.get("model_revision") or metadata.get("model", {}).get("revision")
                if revision:
                    revisions.add(revision)
        if len(revisions) != 1:
            raise ValueError(f"missing or conflicting pretrained base revision for {key}: {sorted(revisions)}")
        repo_id = config.get("base_model_name_or_path")
        if repo_id != "Qwen/Qwen3-ASR-1.7B-hf":
            raise ValueError(f"unsupported pretrained base for {key}: {repo_id}")
        models.append({"key": key, "settings": {
            "adapter": "qwen_hf_peft", "repo_id": repo_id,
            "revision": revisions.pop(), "adapter_path": str(checkpoint),
            "adapter_sha256": digests, "method": method, "epoch": epoch,
            "language": "Vietnamese", "max_new_tokens": 768,
        }})
    identities = {(row["settings"]["repo_id"], row["settings"]["revision"]) for row in models}
    if len(identities) != 1:
        raise ValueError("the two adapters must use the same pretrained base revision")
    return models
