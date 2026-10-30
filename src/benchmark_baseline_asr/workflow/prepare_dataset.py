from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_baseline_asr.application.dataset_service import prepare_manifest
from benchmark_baseline_asr.application.sharding_service import balance_shards
from benchmark_baseline_asr.infrastructure.storage.artifacts import write_jsonl
from benchmark_baseline_asr.workflow.bootstrap import create_run


def prepare(config_path: str, environment: str | None = None, run_id: str | None = None):
    config, run_dir = create_run(config_path, environment, run_id)
    samples = prepare_manifest(config["_loaded"]["dataset"], run_dir / "manifest.jsonl")
    count = config["_loaded"]["execution"]["shard_count"]
    shards = balance_shards(samples, count)
    for index, shard in enumerate(shards):
        write_jsonl(run_dir / "shards" / f"shard_{index}.jsonl", [
            {"sample_id": row.sample_id, "audio_path": str(row.audio_path), "reference": row.reference,
             "duration_seconds": row.duration_seconds} for row in shard
        ])
    return run_dir, samples, shards


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run_dir, samples, shards = prepare(args.config, args.environment, args.run_id)
    print(f"run_dir={run_dir}\nsamples={len(samples)}\nshards={len(shards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
