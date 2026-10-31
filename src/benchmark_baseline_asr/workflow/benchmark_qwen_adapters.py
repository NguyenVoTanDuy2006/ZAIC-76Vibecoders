"""Full labelled Zalo train benchmark for the exported PiSSA and AdaLoRA runs."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from benchmark_baseline_asr.application.config_service import resolve_experiment
from benchmark_baseline_asr.infrastructure.models.peft_checkpoints import discover_full_adapters
from benchmark_baseline_asr.infrastructure.storage.artifacts import write_json
from benchmark_baseline_asr.workflow.prepare_dataset import prepare


def prepare_adapter_benchmark(
    models_root: str | Path, dataset_root: str | Path, output_root: str | Path,
    run_id: str, *, gpu_ids: list[int] | None = None, limit: int | None = None,
):
    """Write a self-contained experiment config and validate every labelled WAV."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be a simple name, not a path")
    gpu_ids = [0] if gpu_ids is None else gpu_ids
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(g < 0 for g in gpu_ids):
        raise ValueError("gpu_ids must contain unique nonnegative GPU indices")
    models = discover_full_adapters(models_root)
    output_root = Path(output_root).expanduser().resolve()
    config_root = output_root / "configs" / run_id
    defaults_root = Path(__file__).resolve().parents[1] / "configs"
    names = ("dataset", "audio", "download", "secrets", "runtime", "execution",
             "normalization", "metrics", "reporting")
    loaded = {name: json.loads((defaults_root / f"{name}.json").read_text()) for name in names}
    loaded["dataset"].update(root=str(Path(dataset_root).expanduser().resolve()), limit=limit)
    loaded["download"].update(models_dir=str(output_root / "models"), local_files_only=False)
    loaded["secrets"] = {"provider": "environment", "token_env": "HF_TOKEN"}
    loaded["runtime"].update(dtype="float16", do_sample=False, num_beams=1)
    loaded["execution"].update(gpu_ids=gpu_ids, shard_count=len(gpu_ids), resume=True)
    loaded["metrics"]["semantic_score"] = {"enabled": False}
    loaded["reporting"]["output_dir"] = str(output_root / "runs")
    config = {"run_name": "qwen_full_adapters_train", "models": [],
              "configs": {name: f"{name}.json" for name in names}}
    config_path = config_root / "experiment.json"
    run_dir = output_root / "runs" / run_id
    # Check identities before prepare() overwrites the run's manifest/config.
    identity = {"models": models, "dataset": loaded["dataset"], "runtime": loaded["runtime"],
                "execution": loaded["execution"], "audio": loaded["audio"]}
    identity_path = run_dir / "adapter_benchmark.json"
    if identity_path.is_file() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("resume inputs changed; select a new run_id")
    for model in models:
        filename = f'{model["key"]}.json'
        config["models"].append({"key": model["key"], "config": filename})
        write_json(config_root / filename, model["settings"])
    for name, value in loaded.items():
        write_json(config_root / f"{name}.json", value)
    write_json(config_path, config)
    run_dir, samples, _ = prepare(str(config_path), run_id=run_id)
    if not samples:
        raise ValueError("train manifest contains no labelled samples")
    write_json(identity_path, identity)
    return config_path, run_dir, samples


def run_adapter_benchmark(config_path: str | Path, run_id: str) -> tuple[Path, list[dict]]:
    """Use existing GPU sharding, resume, metric and reporting services."""
    import torch

    config = resolve_experiment(config_path)
    gpu_ids = config["_loaded"]["execution"]["gpu_ids"]
    if not torch.cuda.is_available() or max(gpu_ids) >= torch.cuda.device_count():
        raise RuntimeError("Configured CUDA GPUs are unavailable. Preparation is CPU-only; run inference on a GPU host.")
    from benchmark_baseline_asr.application.download_service import download_models
    from benchmark_baseline_asr.workflow.transcribe import transcribe
    from benchmark_baseline_asr.workflow.evaluate import evaluate
    from benchmark_baseline_asr.workflow.compare import compare

    run_dir = Path(config["_loaded"]["reporting"]["output_dir"]) / run_id
    # Both adapters share one immutable base; download it once, not once per adapter.
    first = config["models"][0]
    print("Downloading the pinned pretrained base (shared by both adapters)...", flush=True)
    base = download_models([first], config["_loaded"]["download"], os.environ.get("HF_TOKEN"))[first["key"]]
    expected_revision = first["settings"]["revision"]
    if base["revision"] != expected_revision:
        raise ValueError("downloaded base revision differs from the training revision")
    write_json(run_dir / "models_resolved.json", {model["key"]: base for model in config["models"]})
    print("Transcribing train with both adapters; predictions are saved per sample for resume...", flush=True)
    transcribe(str(config_path), run_id=run_id)
    print("Evaluating WER/CER and writing comparison reports...", flush=True)
    evaluate(str(config_path), run_id)
    rows = compare(str(config_path), run_id)
    if any(not row["result_complete"] for row in rows):
        raise RuntimeError(f"incomplete benchmark; inspect worker errors and resume the same run_id: {run_dir}")
    return run_dir, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--dataset-root", default="data/train+public_test/train")
    parser.add_argument("--output-root", default="output/qwen-adapter-benchmark")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for the entire train manifest")
    parser.add_argument("--prepare-only", action="store_true", help="Validate adapters and train without loading the base")
    parser.add_argument("--env-file", type=Path, help="Read HF_TOKEN from this .env without displaying its value")
    args = parser.parse_args(argv)
    if args.env_file is not None:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_path, run_dir, samples = prepare_adapter_benchmark(
        args.models_root, args.dataset_root, args.output_root, run_id, gpu_ids=args.gpu_ids, limit=args.limit)
    print(f"run_id={run_id}\nsamples={len(samples)}\naudio_hours={sum(s.duration_seconds for s in samples)/3600:.3f}\nconfig={config_path}")
    if not args.prepare_only:
        run_dir, rows = run_adapter_benchmark(config_path, run_id)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
