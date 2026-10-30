from __future__ import annotations

from pathlib import Path

from benchmark_baseline_asr.infrastructure.configuration.json_loader import load_experiment, load_json
from benchmark_baseline_asr.infrastructure.configuration.schemas import validate_experiment


def resolve_experiment(path: str | Path, environment: str | None = None) -> dict:
    config = load_experiment(path)
    config["_loaded"]["models"] = {model["key"]: model["settings"] for model in config["models"]}
    if environment:
        env_path = Path(config["_config_root"]) / "environments" / f"{environment}.json"
        overrides = load_json(env_path)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config["_loaded"].get(key), dict):
                config["_loaded"][key].update(value)
            else:
                config["_loaded"][key] = value
    validate_experiment(config)
    return config
