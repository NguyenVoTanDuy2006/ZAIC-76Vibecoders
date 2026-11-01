from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    repo_url: str
    ref: str
    source_dir: Path
    managed_root: Path
    include_paths: tuple[str, ...]


def load_config(path: str | Path) -> SourceConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source config {config_path}: {exc}") from exc
    required = ("repo_url", "ref", "source_dir", "include_paths")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"source config is missing: {', '.join(missing)}")
    repo_url, ref, include_paths = raw["repo_url"], raw["ref"], raw["include_paths"]
    if not isinstance(repo_url, str) or not repo_url.strip() or not isinstance(ref, str) or not ref.strip():
        raise ValueError("repo_url and ref must be non-empty strings")
    if not isinstance(include_paths, list) or not include_paths:
        raise ValueError("include_paths must be a non-empty array")
    paths = []
    for item in include_paths:
        if not isinstance(item, str) or not item or Path(item).is_absolute() or ".." in Path(item).parts:
            raise ValueError("include_paths entries must be safe relative paths")
        paths.append(item)
    requested_source = Path(raw["source_dir"]).expanduser()
    if not requested_source.is_absolute():
        requested_source = Path.cwd() / requested_source
    source_dir = requested_source.parent.resolve() / requested_source.name
    managed_root = Path(raw.get("managed_root", source_dir.parent)).expanduser().resolve()
    if managed_root.name != "_source":
        raise ValueError("managed_root must be a dedicated _source directory")
    if source_dir == managed_root or managed_root not in source_dir.parents:
        raise ValueError(f"source_dir must be a descendant of managed root {managed_root}")
    return SourceConfig(repo_url.strip(), ref.strip(), source_dir, managed_root, tuple(paths))
