from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark_baseline_asr.domain.errors import WorkerError
from benchmark_baseline_asr.infrastructure.storage.artifacts import write_json


def run_gpu_workers(specs: list[dict], *, start_timeout: int, join_timeout: int, timing_path: str | Path) -> dict:
    ready_paths = [Path(spec["ready_path"]) for spec in specs]
    barrier_path = Path(specs[0]["barrier_path"])
    processes = []
    for spec in specs:
        spec_path = Path(spec["spec_path"])
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_id"])
        env["PYTHONHASHSEED"] = str(spec["runtime"].get("seed", 1234))
        process = subprocess.Popen(
            [sys.executable, "-m", "benchmark_baseline_asr.workflow.transcribe_worker", "--spec", str(spec_path)],
            env=env,
        )
        processes.append(process)

    start_deadline = time.monotonic() + start_timeout
    ready = set()
    while len(ready) < len(specs):
        for index, (process, path) in enumerate(zip(processes, ready_paths)):
            if index in ready or not path.is_file():
                continue
            status = json.loads(path.read_text(encoding="utf-8"))
            if not status.get("ok"):
                _terminate(processes)
                raise WorkerError(f"GPU worker {index} failed to initialize: {status.get('error')}")
            ready.add(index)
        if len(ready) == len(specs):
            break
        if any(process.poll() is not None for process in processes):
            _terminate(processes)
            raise WorkerError("a GPU worker exited before completing model load and warmup")
        if time.monotonic() >= start_deadline:
            _terminate(processes)
            raise WorkerError("timed out while waiting for GPU workers to load and warm up")
        time.sleep(0.1)

    barrier_path.parent.mkdir(parents=True, exist_ok=True)
    barrier_path.write_text(str(time.time()), encoding="utf-8")
    wall_start = time.perf_counter()
    join_deadline = time.monotonic() + join_timeout
    try:
        for process in processes:
            remaining = max(0.0, join_deadline - time.monotonic())
            process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        _terminate(processes)
        raise WorkerError("GPU workers exceeded the configured inference timeout") from exc
    wall_seconds = time.perf_counter() - wall_start
    if any(process.returncode != 0 for process in processes):
        raise WorkerError(f"GPU worker process failed with exit codes {[p.returncode for p in processes]}")
    worker_summaries = [json.loads(Path(spec["summary_path"]).read_text(encoding="utf-8")) for spec in specs]
    timing = {"wall_seconds": wall_seconds, "workers": worker_summaries}
    write_json(timing_path, timing)
    return timing


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
