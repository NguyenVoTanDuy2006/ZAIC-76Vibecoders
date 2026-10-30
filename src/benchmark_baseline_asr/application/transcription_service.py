from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.infrastructure.models.factory import create_adapter


def build_adapter(model_config: dict):
    return create_adapter(model_config["adapter"])


def ensure_resume_allowed(model_dir: str | Path, *, resume: bool) -> None:
    if resume:
        return
    if any(Path(model_dir).glob("worker_*/predictions.jsonl")):
        raise ValueError("resume is disabled but worker predictions exist; use a new run_id")
