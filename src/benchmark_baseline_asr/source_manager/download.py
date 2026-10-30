from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import load_config

MARKER_NAME = ".source_manager.json"


@dataclass(frozen=True)
class DownloadedSource:
    path: Path
    repo_url: str
    requested_ref: str
    commit: str


def download_source(config_path: str | Path) -> DownloadedSource:
    config = load_config(config_path)
    config.managed_root.mkdir(parents=True, exist_ok=True)
    if config.source_dir.exists():
        return _reuse(config)
    staging = Path(tempfile.mkdtemp(prefix=".source-download-", dir=config.managed_root))
    shutil.rmtree(staging)
    try:
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", "--no-tags", config.repo_url, str(staging)])
        _run(["git", "-C", str(staging), "fetch", "--depth=1", "origin", config.ref])
        _run(["git", "-C", str(staging), "sparse-checkout", "set", "--no-cone", *config.include_paths])
        _run(["git", "-C", str(staging), "checkout", "--detach", "FETCH_HEAD"])
        commit = _run(["git", "-C", str(staging), "rev-parse", "HEAD"], capture=True).strip()
        marker = {"version": 1, "repo_url": config.repo_url, "requested_ref": config.ref, "commit": commit}
        (staging / MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        config.source_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(config.source_dir)
        os.environ["ASR_SOURCE_PATH"] = str(config.source_dir)
        return DownloadedSource(config.source_dir, config.repo_url, config.ref, commit)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reuse(config) -> DownloadedSource:
    if config.source_dir.is_symlink() or not config.source_dir.is_dir():
        raise ValueError(f"source target exists but is not a regular directory: {config.source_dir}")
    marker_path = config.source_dir / MARKER_NAME
    if not marker_path.is_file():
        raise ValueError(f"refusing to overwrite unmanaged source target: {config.source_dir}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        commit = _run(["git", "-C", str(config.source_dir), "rev-parse", "HEAD"], capture=True).strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source marker in {config.source_dir}") from exc
    if marker.get("repo_url") != config.repo_url or marker.get("requested_ref") != config.ref:
        raise ValueError(f"source target belongs to a different repo/ref: {config.source_dir}")
    if marker.get("version") != 1 or marker.get("commit") != commit:
        raise ValueError(f"source marker does not match checkout: {config.source_dir}")
    os.environ["ASR_SOURCE_PATH"] = str(config.source_dir)
    return DownloadedSource(config.source_dir, config.repo_url, config.ref, commit)


def _run(command: list[str], *, capture: bool = False) -> str:
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to download source") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Git command failed: {exc.stderr.strip()}") from exc
    return result.stdout if capture else ""
