from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_baseline_asr.application.config_service import resolve_experiment
from benchmark_baseline_asr.infrastructure.storage.artifacts import write_json


def create_run(config_path: str, environment: str | None = None, run_id: str | None = None):
    config = resolve_experiment(config_path, environment)
    raw_output = Path(config["_loaded"]["reporting"]["output_dir"])
    output_root = raw_output if raw_output.is_absolute() else Path.cwd() / raw_output
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", _safe_config(config))
    write_json(run_dir / "metadata.json", _runtime_metadata(config))
    return config, run_dir


def _safe_config(config: dict) -> dict:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    clean["configs"] = config.get("_loaded", {})
    return clean


def _runtime_metadata(config: dict) -> dict:
    try:
        source_value = os.environ.get("ASR_SOURCE_PATH")
        source_path = Path(source_value) if source_value else None
        marker = json.loads((source_path / ".source_manager.json").read_text()) if source_path else {}
    except (OSError, json.JSONDecodeError):
        marker = {}
    try:
        import torch

        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch_version = torch.__version__
    except ImportError:
        devices, torch_version = [], None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": marker,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_devices": devices,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
