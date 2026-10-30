from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON configuration {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a JSON object: {target}")
    return value


def load_experiment(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    experiment = load_json(target)
    root = target.parent
    experiment["_loaded"] = {key: load_json(root / relative) for key, relative in experiment.get("configs", {}).items()}
    for model in experiment.get("models", []):
        model["settings"] = load_json(root / model["config"])
    experiment["_config_root"] = str(root)
    return experiment
