from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.infrastructure.models.huggingface import download_snapshot


def download_models(models: list[dict], download_config: dict, token: str | None) -> dict[str, dict]:
    cache_root = Path(download_config["models_dir"])
    resolved = {}
    for entry in models:
        settings = entry["settings"]
        target = cache_root / entry["key"]
        path, revision = download_snapshot(
            settings["repo_id"], settings.get("revision"), target, token,
            local_files_only=download_config.get("local_files_only", False),
        )
        resolved[entry["key"]] = {"path": str(path), "revision": revision, "repo_id": settings["repo_id"]}
    return resolved
