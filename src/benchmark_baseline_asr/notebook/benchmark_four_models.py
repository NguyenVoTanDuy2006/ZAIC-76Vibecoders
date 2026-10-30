# %% [markdown]
# # Vietnamese ASR benchmark — Kaggle T4 ×2
#
# Python script with `# %%` cells, so it can be opened as an interactive notebook
# in VS Code or run directly with Python. It evaluates all four configured models
# on the train split and writes predictions, metrics, timing, and comparison files
# under `/kaggle/working/asr_benchmark/runs/<RUN_ID>/`.
#
# Before running:
# 1. Attach the train dataset to the Kaggle notebook.
# 2. The Kaggle profile is set to `/kaggle/input/datasets/khangl123/zalo-ai/`
#    `train+public_test/train`; use `DATASET_ROOT_OVERRIDE` below if the mount path changes.
# 3. Select a T4 ×2 accelerator and add a Kaggle Secret named `HF_TOKEN`.
# 4. Install the project ASR extra once from the repository root if needed:
#    `%pip install ".[asr]"`
#
# The source tree must be available in the Kaggle session. This script does not
# delete source files or benchmark outputs when it finishes.

# %%
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MODELS = (
    "phowhisper_medium",
    "phowhisper_large",
    "qwen3_asr_0_6b",
    "qwen3_asr_1_7b",
)
ENVIRONMENT = "kaggle"
RUN_ID = os.environ.get("ASR_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# Optional path to the mounted directory containing train.jsonl and audio/.
# Example: Path("/kaggle/input/zalo-ai-challenge/train")
# Leave as None to use configs/environments/kaggle.json.
DATASET_ROOT_OVERRIDE: Path | None = None


def find_project_root() -> Path:
    explicit_root = os.environ.get("ASR_PROJECT_ROOT")
    candidates = [Path(explicit_root).expanduser()] if explicit_root else []
    candidates.extend([Path.cwd(), *Path(__file__).resolve().parents])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "pyproject.toml").is_file() and (
            root / "src/benchmark_baseline_asr/configs/experiment.json"
        ).is_file():
            return root
    raise FileNotFoundError(
        "Cannot locate the repository root. Set ASR_PROJECT_ROOT to the checkout "
        "containing pyproject.toml and src/benchmark_baseline_asr/."
    )


PROJECT_ROOT = find_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "benchmark_baseline_asr"
CONFIG_ROOT = PACKAGE_ROOT / "configs"
EXPERIMENT_CONFIG = CONFIG_ROOT / "experiment.json"
KAGGLE_PROFILE = CONFIG_ROOT / "environments/kaggle.json"

# Make the source package importable in this process and in the two GPU workers,
# which are launched as child Python processes by the workflow.
sys.path.insert(0, str(SOURCE_ROOT))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_entries = [str(SOURCE_ROOT)]
if existing_pythonpath:
    pythonpath_entries.append(existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
os.chdir(PROJECT_ROOT)


def make_runtime_config() -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Copy small JSON configs only when a dataset path override is supplied."""
    if DATASET_ROOT_OVERRIDE is None:
        return EXPERIMENT_CONFIG, None

    temp_dir = tempfile.TemporaryDirectory(prefix="asr-benchmark-config-")
    runtime_config_root = Path(temp_dir.name) / "configs"
    shutil.copytree(CONFIG_ROOT, runtime_config_root)
    profile_path = runtime_config_root / "environments/kaggle.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.setdefault("dataset", {})["root"] = str(DATASET_ROOT_OVERRIDE.expanduser().resolve())
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return runtime_config_root / "experiment.json", temp_dir


def load_runtime_settings(config_path: Path) -> tuple[dict[str, Any], Path]:
    profile_path = config_path.parent / "environments" / f"{ENVIRONMENT}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    dataset_root = Path(profile["dataset"]["root"]).expanduser()
    if "REPLACE_WITH_DATASET_NAME" in str(dataset_root):
        raise ValueError(
            f"Set dataset.root in {KAGGLE_PROFILE} or set DATASET_ROOT_OVERRIDE "
            "near the top of this script. It must point to the mounted train/ directory."
        )
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    dataset_root = dataset_root.resolve()
    if not (dataset_root / "train.jsonl").is_file():
        raise FileNotFoundError(f"train.jsonl was not found under dataset root: {dataset_root}")
    if not (dataset_root / "audio").is_dir():
        raise FileNotFoundError(f"audio/ was not found under dataset root: {dataset_root}")

    experiment = json.loads(config_path.read_text(encoding="utf-8"))
    model_keys = tuple(row["key"] for row in experiment["models"])
    if model_keys != EXPECTED_MODELS:
        raise ValueError(f"Expected model order {EXPECTED_MODELS}; found {model_keys}")
    return profile, dataset_root


def check_gpu_runtime() -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Install the ASR dependencies in the Kaggle session.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator (T4 ×2).")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"This run expects two GPUs; found {torch.cuda.device_count()}.")
    return [torch.cuda.get_device_name(index) for index in range(2)]


def print_comparison(rows: list[dict[str, Any]]) -> None:
    columns = ("model", "raw_cer", "raw_wer", "normalized_cer", "normalized_wer", "coverage", "rtf_wall")
    print("\nFinal comparison (error rates and coverage are percentages):")
    print(f"{'model':24} {'CER raw':>10} {'WER raw':>10} {'CER norm':>10} {'WER norm':>10} {'coverage':>10} {'RTF wall':>10} status")
    for row in rows:
        values = []
        for key in columns[1:-1]:
            value = row.get(key)
            values.append("n/a" if value is None else f"{100 * value:.2f}%")
        rtf = row.get("rtf_wall")
        rtf_text = "n/a" if rtf is None else f"{rtf:.3f}"
        status = "complete" if row.get("result_complete") else "INCOMPLETE"
        print(f"{row['model']:24} {values[0]:>10} {values[1]:>10} {values[2]:>10} {values[3]:>10} {values[4]:>10} {rtf_text:>10} {status}")


# %%
def run_benchmark_notebook() -> Path:
    from benchmark_baseline_asr.infrastructure.secrets.kaggle_secrets import get_huggingface_token
    from benchmark_baseline_asr.workflow.compare import compare
    from benchmark_baseline_asr.workflow.download_models import download
    from benchmark_baseline_asr.workflow.evaluate import evaluate
    from benchmark_baseline_asr.workflow.prepare_dataset import prepare
    from benchmark_baseline_asr.workflow.transcribe import transcribe

    config_path, config_temp_dir = make_runtime_config()
    try:
        _, dataset_root = load_runtime_settings(config_path)
        devices = check_gpu_runtime()
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Dataset root: {dataset_root}")
        print(f"GPUs: {', '.join(devices)}")
        print(f"Run ID: {RUN_ID}")

        print("\n[1/5] Preparing manifest and two duration-balanced shards...")
        run_dir, samples, shards = prepare(str(config_path), ENVIRONMENT, RUN_ID)
        print(f"Prepared {len(samples)} samples across {len(shards)} shards.")

        print("\n[2/5] Checking Kaggle HF_TOKEN and downloading model snapshots...")
        token = get_huggingface_token("HF_TOKEN")
        try:
            download(str(config_path), ENVIRONMENT, RUN_ID, token=token)
        finally:
            del token

        print("\n[3/5] Transcribing with all four models on the two GPUs...")
        transcribe(str(config_path), ENVIRONMENT, RUN_ID)

        print("\n[4/5] Calculating raw and normalized CER/WER...")
        evaluate(str(config_path), RUN_ID, ENVIRONMENT)

        print("\n[5/5] Comparing model metrics and writing reports...")
        rows = compare(str(config_path), RUN_ID, ENVIRONMENT)
        print_comparison(rows)
        print(f"\nRun artifacts: {run_dir}")
        print(
            "Comparison files: "
            f"{run_dir / 'comparison.json'}, {run_dir / 'comparison.csv'}, {run_dir / 'comparison.md'}"
        )
        return run_dir
    finally:
        if config_temp_dir is not None:
            config_temp_dir.cleanup()


# %%
if __name__ == "__main__":
    run_benchmark_notebook()
