from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_baseline_asr.application.prediction_merge_service import merge_worker_predictions
from benchmark_baseline_asr.application.transcription_service import ensure_resume_allowed
from benchmark_baseline_asr.domain.entities import DatasetSample
from benchmark_baseline_asr.infrastructure.execution.process_runner import run_gpu_workers
from benchmark_baseline_asr.infrastructure.storage.artifacts import read_json, read_jsonl, write_json
from benchmark_baseline_asr.workflow.bootstrap import create_run, file_sha256


def transcribe(config_path: str, environment: str | None = None, run_id: str | None = None):
    config, run_dir = create_run(config_path, environment, run_id)
    manifest_path = run_dir / "manifest.jsonl"
    shard_dir = run_dir / "shards"
    if not manifest_path.is_file():
        raise FileNotFoundError("run prepare_dataset before transcribe; manifest.jsonl is missing")
    resolved_path = run_dir / "models_resolved.json"
    if not resolved_path.is_file():
        raise FileNotFoundError("run download_models before transcribe; models_resolved.json is missing")
    resolved_models = read_json(resolved_path)
    shards = [read_jsonl(path) for path in sorted(shard_dir.glob("shard_*.jsonl"))]
    if len(shards) != len(config["_loaded"]["execution"]["gpu_ids"]):
        raise ValueError("number of prepared shards must equal the configured GPU count")

    for model_entry in config["models"]:
        key, settings = model_entry["key"], model_entry["settings"]
        model_info = resolved_models[key]
        model_dir = run_dir / key
        model_dir.mkdir(parents=True, exist_ok=True)
        ensure_resume_allowed(model_dir, resume=config["_loaded"]["execution"].get("resume", True))
        fingerprint = _fingerprint(manifest_path, settings, model_info, config["_loaded"]["runtime"],
                                   config["_loaded"]["execution"], config["_loaded"]["audio"])
        fingerprint_path = model_dir / "fingerprint.json"
        if fingerprint_path.exists() and read_json(fingerprint_path) != fingerprint:
            raise ValueError(f"resume configuration changed for {key}; use a new run_id")
        write_json(fingerprint_path, fingerprint)
        specs = []
        for worker_id, (gpu_id, samples) in enumerate(zip(config["_loaded"]["execution"]["gpu_ids"], shards)):
            worker_dir = model_dir / f"worker_{worker_id}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            specs.append({
                "worker_id": worker_id, "gpu_id": gpu_id, "model": settings,
                "model_path": model_info["path"], "runtime": config["_loaded"]["runtime"],
                "audio": config["_loaded"]["audio"],
                "samples": samples, "prediction_path": str(worker_dir / "predictions.jsonl"),
                "errors_path": str(worker_dir / "errors.jsonl"), "summary_path": str(worker_dir / "summary.json"),
                "ready_path": str(worker_dir / "ready.json"), "barrier_path": str(model_dir / ".start_barrier"),
                "barrier_timeout_seconds": config["_loaded"]["execution"]["worker_start_timeout_seconds"],
                "spec_path": str(worker_dir / "worker_spec.json"),
            })
        for spec in specs:
            Path(spec["ready_path"]).unlink(missing_ok=True)
            Path(spec["summary_path"]).unlink(missing_ok=True)
        Path(specs[0]["barrier_path"]).unlink(missing_ok=True)
        timing = run_gpu_workers(
            specs,
            start_timeout=config["_loaded"]["execution"]["worker_start_timeout_seconds"],
            join_timeout=config["_loaded"]["execution"]["worker_join_timeout_seconds"],
            timing_path=model_dir / "run_timing.json",
        )
        worker_predictions = [spec["prediction_path"] for spec in specs]
        domain_shards = [[DatasetSample(row["sample_id"], Path(row["audio_path"]), row["reference"], row["duration_seconds"])
                          for row in shard] for shard in shards]
        merge_worker_predictions(domain_shards, worker_predictions, str(model_dir / "predictions.jsonl"))
        write_json(model_dir / "run_timing.json", timing)
    return run_dir


def _fingerprint(manifest_path, settings, model_info, runtime, execution, audio):
    data = {"manifest_sha256": file_sha256(manifest_path), "model_revision": model_info["revision"],
            "model_config": settings, "runtime": runtime, "execution": execution, "audio": audio}
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    data["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(f"run_dir={transcribe(args.config, args.environment, args.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
