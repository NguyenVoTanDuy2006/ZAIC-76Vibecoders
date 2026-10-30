from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .download import MARKER_NAME


@dataclass(frozen=True)
class CleanupResult:
    path: Path
    removed: bool


def cleanup_source(config_path: str | Path) -> CleanupResult:
    config = load_config(config_path)
    path = config.source_dir
    if not path.exists():
        return CleanupResult(path, False)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"refusing to remove non-directory or symlink: {path}")
    marker_path = path / MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"refusing to remove unmarked source directory: {path}") from exc
    if marker.get("repo_url") != config.repo_url or marker.get("requested_ref") != config.ref:
        raise ValueError(f"source marker does not match config: {path}")
    if marker.get("version") != 1 or not marker.get("commit") or not (path / ".git").is_dir():
        raise ValueError(f"source marker is incomplete or forged: {marker_path}")
    try:
        checkout = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"source marker has no valid Git checkout: {path}") from exc
    if checkout != marker["commit"]:
        raise ValueError(f"source marker commit does not match checkout: {path}")
    shutil.rmtree(path)
    if os.environ.get("ASR_SOURCE_PATH") == str(path):
        os.environ.pop("ASR_SOURCE_PATH", None)
    return CleanupResult(path, True)
