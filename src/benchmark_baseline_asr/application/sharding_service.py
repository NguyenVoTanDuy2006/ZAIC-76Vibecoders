from __future__ import annotations

from benchmark_baseline_asr.domain.entities import DatasetSample


def balance_shards(samples: list[DatasetSample], shard_count: int) -> list[list[DatasetSample]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[DatasetSample]] = [[] for _ in range(shard_count)]
    totals = [0.0] * shard_count
    for sample in sorted(samples, key=lambda row: (-row.duration_seconds, row.sample_id)):
        target = min(range(shard_count), key=lambda index: (totals[index], index))
        shards[target].append(sample)
        totals[target] += sample.duration_seconds
    return shards
