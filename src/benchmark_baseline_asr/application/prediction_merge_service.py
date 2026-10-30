from __future__ import annotations

from benchmark_baseline_asr.domain.entities import DatasetSample
from benchmark_baseline_asr.domain.errors import WorkerError
from benchmark_baseline_asr.infrastructure.storage.artifacts import read_jsonl, write_jsonl


def merge_worker_predictions(shards: list[list[DatasetSample]], worker_paths: list[str], output_path: str) -> list[dict]:
    expected = {sample.sample_id for shard in shards for sample in shard}
    predictions = []
    seen = set()
    for worker_path in worker_paths:
        for row in read_jsonl(worker_path):
            sample_id = row.get("sample_id")
            if sample_id not in expected or sample_id in seen:
                raise WorkerError(f"unexpected or duplicate prediction ID: {sample_id}")
            seen.add(sample_id)
            predictions.append(row)
    merged = sorted(predictions, key=lambda row: row["sample_id"])
    write_jsonl(output_path, merged)
    return merged
