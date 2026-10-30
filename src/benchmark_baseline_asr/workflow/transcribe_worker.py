from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from benchmark_baseline_asr.application.audio_service import prepare_audio
from benchmark_baseline_asr.infrastructure.models.factory import create_adapter
from benchmark_baseline_asr.infrastructure.runtime.reproducibility import configure_determinism
from benchmark_baseline_asr.infrastructure.runtime.timing import synchronize_device, timed_call
from benchmark_baseline_asr.infrastructure.storage.artifacts import append_jsonl, read_jsonl, write_json


def run_worker(spec_path: str | Path) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_id"])
    runtime = dict(spec["runtime"])
    runtime["device"] = "cuda:0"
    configure_determinism(runtime["seed"], runtime.get("cpu_threads_per_worker", 2))
    adapter = create_adapter(spec["model"]["adapter"])
    ready_path = Path(spec["ready_path"])
    output_path = Path(spec["prediction_path"])
    errors_path = Path(spec["errors_path"])
    summary_path = Path(spec["summary_path"])
    try:
        load_start = time.perf_counter()
        adapter.load(spec["model_path"], spec["model"], runtime)
        synchronize_device(runtime["device"])
        load_seconds = time.perf_counter() - load_start
        existing = {row["sample_id"] for row in read_jsonl(output_path)}
        pending = [row for row in spec["samples"] if row["sample_id"] not in existing]
        warmup_seconds = 0.0
        if pending and runtime.get("warmup_samples_per_worker", 1) > 0:
            first = pending[0]
            _, warmup_seconds = timed_call(
                lambda: adapter.transcribe(prepare_audio(first["audio_path"], spec["audio"])), device=runtime["device"]
            )
        write_json(ready_path, {"ok": True, "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
                                "pending": len(pending)})
        barrier = Path(spec["barrier_path"])
        deadline = time.monotonic() + spec["barrier_timeout_seconds"]
        while not barrier.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for both GPU workers to finish warmup")
            time.sleep(0.05)
        wall_start = time.perf_counter()
        succeeded = failed = 0
        for sample in pending:
            try:
                result, latency = timed_call(
                    lambda: adapter.transcribe(prepare_audio(sample["audio_path"], spec["audio"])),
                    device=runtime["device"],
                )
                append_jsonl(output_path, {"sample_id": sample["sample_id"], "text": result.text,
                                           "language": result.language, "latency_seconds": latency})
                succeeded += 1
            except Exception as exc:
                append_jsonl(errors_path, {"sample_id": sample["sample_id"], "error_type": type(exc).__name__,
                                           "error": str(exc)})
                failed += 1
        elapsed = time.perf_counter() - wall_start
        write_json(summary_path, {"worker_id": spec["worker_id"], "gpu_id": spec["gpu_id"],
                                  "expected": len(spec["samples"]), "resumed": len(spec["samples"]) - len(pending),
                                  "succeeded": succeeded, "failed": failed, "model_load_seconds": load_seconds,
                                  "warmup_seconds": warmup_seconds, "elapsed_seconds": elapsed})
        return 0
    except Exception as exc:
        write_json(ready_path, {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 1
    finally:
        adapter.unload()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    return run_worker(args.spec)


if __name__ == "__main__":
    raise SystemExit(main())
