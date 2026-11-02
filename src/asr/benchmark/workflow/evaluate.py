from __future__ import annotations

import argparse
import os
from pathlib import Path

from asr.benchmark.application.config_service import resolve_experiment
from asr.benchmark.application.evaluation_service import evaluate_predictions
from asr.benchmark.infrastructure.evaluation.semantic_metrics import PhoBERTBERTScore
from asr.benchmark.infrastructure.models.huggingface import download_snapshot
from asr.benchmark.infrastructure.secrets.kaggle_secrets import get_huggingface_token


def evaluate(config_path: str, run_id: str, environment: str | None = None):
    config = resolve_experiment(config_path, environment)
    output_root = Path(config["_loaded"]["reporting"]["output_dir"])
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    run_dir = output_root / run_id
    if not (run_dir / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"manifest not found in run {run_dir}; prepare the dataset first")
    semantic_config = dict(config["_loaded"]["metrics"].get("semantic_score", {}))
    semantic_scorer = None
    if semantic_config.get("enabled", False):
        secret_config = config["_loaded"]["secrets"]
        token = (get_huggingface_token(secret_config.get("token_label", "HF_TOKEN"))
                 if secret_config.get("provider", "kaggle") == "kaggle"
                 else os.environ.get(secret_config.get("token_env", "HF_TOKEN")))
        model_id = semantic_config["model_id"]
        model_path, revision = download_snapshot(
            model_id, semantic_config.get("revision"),
            Path(config["_loaded"]["download"]["models_dir"]) / f"semantic_{model_id.replace('/', '--')}",
            token, local_files_only=config["_loaded"]["download"].get("local_files_only", False),
            allow_patterns=["*.json", "*.txt", "*.codes", "*.py", "*.safetensors", "pytorch_model.bin"],
        )
        semantic_config["revision"] = revision
        semantic_scorer = PhoBERTBERTScore(str(model_path), semantic_config)
    result = {}
    for model in config["models"]:
        model_dir = run_dir / model["key"]
        result[model["key"]] = evaluate_predictions(
            run_dir / "manifest.jsonl", model_dir / "predictions.jsonl", model_dir,
            config["_loaded"]["normalization"],
            semantic_scorer, semantic_config,
        )
    return run_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    _, result = evaluate(args.config, args.run_id, args.environment)
    print(f"evaluated_models={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
