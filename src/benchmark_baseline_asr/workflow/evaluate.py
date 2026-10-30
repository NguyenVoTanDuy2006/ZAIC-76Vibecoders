from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_baseline_asr.application.config_service import resolve_experiment
from benchmark_baseline_asr.application.evaluation_service import evaluate_predictions


def evaluate(config_path: str, run_id: str, environment: str | None = None):
    config = resolve_experiment(config_path, environment)
    output_root = Path(config["_loaded"]["reporting"]["output_dir"])
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    run_dir = output_root / run_id
    if not (run_dir / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"manifest not found in run {run_dir}; prepare the dataset first")
    result = {}
    for model in config["models"]:
        model_dir = run_dir / model["key"]
        result[model["key"]] = evaluate_predictions(
            run_dir / "manifest.jsonl", model_dir / "predictions.jsonl", model_dir,
            config["_loaded"]["normalization"],
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
