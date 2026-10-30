from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.domain.entities import DatasetSample
from benchmark_baseline_asr.infrastructure.dataset.train_jsonl import load_train_manifest
from benchmark_baseline_asr.infrastructure.storage.artifacts import write_jsonl


def prepare_manifest(dataset_config: dict, output_path: str | Path) -> list[DatasetSample]:
    root = Path(dataset_config["root"])
    jsonl_path = root / dataset_config.get("jsonl", "train.jsonl")
    audio_root = root / dataset_config.get("audio_root", ".")
    samples = load_train_manifest(jsonl_path, audio_root)
    limit = dataset_config.get("limit")
    if limit is not None:
        if limit < 1:
            raise ValueError("dataset limit must be positive or null")
        samples = samples[:limit]
    write_jsonl(output_path, [
        {"sample_id": row.sample_id, "audio_path": str(row.audio_path), "reference": row.reference,
         "duration_seconds": row.duration_seconds}
        for row in samples
    ])
    return samples
