from __future__ import annotations

import argparse

from benchmark_baseline_asr.workflow.benchmark import run_benchmark
from benchmark_baseline_asr.workflow.compare import compare
from benchmark_baseline_asr.workflow.download_models import download
from benchmark_baseline_asr.workflow.evaluate import evaluate
from benchmark_baseline_asr.workflow.prepare_dataset import prepare
from benchmark_baseline_asr.workflow.transcribe import transcribe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asr-benchmark")
    parser.add_argument("command", choices=("prepare-dataset", "download-models", "transcribe", "evaluate", "compare", "benchmark"))
    parser.add_argument("--config", default="src/benchmark_baseline_asr/configs/experiment.json")
    parser.add_argument("--environment", choices=("local", "kaggle"))
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.command == "prepare-dataset":
        run_dir, samples, _ = prepare(args.config, args.environment, args.run_id)
        print(f"run_dir={run_dir}\nsamples={len(samples)}")
    elif args.command == "download-models":
        run_dir, models = download(args.config, args.environment, args.run_id)
        print(f"run_dir={run_dir}\nmodels={len(models)}")
    elif args.command == "transcribe":
        print(f"run_dir={transcribe(args.config, args.environment, args.run_id)}")
    elif args.command in {"evaluate", "compare"}:
        if not args.run_id:
            parser.error(f"--run-id is required for {args.command}")
        if args.command == "evaluate":
            _, results = evaluate(args.config, args.run_id, args.environment)
            print(f"evaluated_models={len(results)}")
        else:
            rows = compare(args.config, args.run_id, args.environment)
            print(f"compared_models={len(rows)}")
    else:
        run_id, rows = run_benchmark(args.config, args.environment, args.run_id)
        print(f"run_id={run_id}\nmodels={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
