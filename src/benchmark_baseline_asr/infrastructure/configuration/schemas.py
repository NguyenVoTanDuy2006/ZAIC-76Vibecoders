from __future__ import annotations


def validate_experiment(config: dict) -> None:
    models = config.get("models", [])
    keys = [row.get("key") for row in models]
    if not models or None in keys or len(keys) != len(set(keys)):
        raise ValueError("experiment needs model entries with unique keys")
    loaded = config.get("_loaded", {})
    required = {"dataset", "download", "secrets", "runtime", "execution", "metrics", "normalization", "reporting"}
    missing = required - loaded.keys()
    if missing:
        raise ValueError(f"experiment is missing configs: {', '.join(sorted(missing))}")
    gpu_ids = loaded["execution"].get("gpu_ids")
    if not gpu_ids:
        raise ValueError("execution.gpu_ids must not be empty")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("execution.gpu_ids must be unique")
    shard_count = loaded["execution"].get("shard_count", len(gpu_ids))
    if shard_count != len(gpu_ids):
        raise ValueError("execution.shard_count must match the configured GPU count")
