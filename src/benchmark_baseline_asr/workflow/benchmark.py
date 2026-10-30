from __future__ import annotations

import argparse
from datetime import datetime, timezone

from benchmark_baseline_asr.workflow.compare import compare
from benchmark_baseline_asr.workflow.download_models import download
from benchmark_baseline_asr.workflow.evaluate import evaluate
from benchmark_baseline_asr.workflow.prepare_dataset import prepare
from benchmark_baseline_asr.workflow.transcribe import transcribe


def run_benchmark(config_path: str, environment: str | None = None, run_id: str | None = None):
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prepare(config_path, environment, run_id)
    download(config_path, environment, run_id)
    transcribe(config_path, environment, run_id)
    evaluate(config_path, run_id, environment)
    rows = compare(config_path, run_id, environment)
    return run_id, rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run_id, rows = run_benchmark(args.config, args.environment, args.run_id)
    print(f"run_id={run_id}\nmodels={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
