from __future__ import annotations

import json
from pathlib import Path


def download_snapshot(
    repo_id: str,
    revision: str | None,
    destination: str | Path,
    token: str | None,
    *,
    local_files_only: bool = False,
) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    target = Path(destination)
    marker = target / ".benchmark_model.json"
    if local_files_only:
        if not target.is_dir() or not marker.is_file():
            raise FileNotFoundError(f"no verified local model snapshot found for {repo_id}: {target}")
        local_metadata = json.loads(marker.read_text(encoding="utf-8"))
        if local_metadata.get("repo_id") != repo_id:
            raise ValueError(f"cached model at {target} belongs to a different repository")
        return target, local_metadata["revision"]
    api = HfApi(token=token)
    resolved_revision = revision or api.model_info(repo_id).sha
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, revision=resolved_revision, local_dir=str(target), token=token)
    marker.write_text(json.dumps({"repo_id": repo_id, "revision": resolved_revision}, indent=2) + "\n", encoding="utf-8")
    return target, resolved_revision
