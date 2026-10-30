from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_baseline_asr.application.config_service import resolve_experiment
from benchmark_baseline_asr.application.reporting_service import build_comparison


def compare(config_path: str, run_id: str, environment: str | None = None):
    config = resolve_experiment(config_path, environment)
    output_root = Path(config["_loaded"]["reporting"]["output_dir"])
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    run_dir = output_root / run_id
    return build_comparison(run_dir, [model["key"] for model in config["models"]])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    rows = compare(args.config, args.run_id, args.environment)
    print(f"compared_models={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
