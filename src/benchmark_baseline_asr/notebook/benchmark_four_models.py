# %% [markdown]
# # Vietnamese ASR benchmark — Kaggle T4 ×2
#
# So sánh PhoWhisper-medium/large và Qwen3-ASR-0.6B/1.7B trên train split.
# Notebook tạo transcript, tính CER/WER raw và normalized, đo tốc độ suy luận,
# rồi lưu kết quả vào `/kaggle/working/asr_benchmark/runs/<RUN_ID>/`.
#
# Trước khi chạy, attach dataset `zalo-ai`, chọn GPU T4 ×2 và thêm Kaggle Secret
# `HF_TOKEN`. Kaggle profile đã trỏ tới `/kaggle/input/datasets/khangl123/zalo-ai/`
# `train+public_test/train`; có thể đổi bằng `DATASET_ROOT_OVERRIDE` nếu cần.
# Bật Internet để clone source và tải model. Source không tự bị xóa khi hoàn tất.

# %%
from __future__ import annotations

import json
import os
import shutil
import subprocess
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
SOURCE_REPO_URL = "https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git"
SOURCE_REF = "benchmark_base_asr_model"
SOURCE_INCLUDE_PATHS = (
    "src/benchmark_baseline_asr",
    "src/zalo_ai_challenge",
    "pyproject.toml",
    "uv.lock",
    "README.md",
)
SOURCE_DIR_DEFAULT = Path("/kaggle/working/_source/ZAIC-76Vibecoders")
if not Path("/kaggle/working").is_dir():
    SOURCE_DIR_DEFAULT = Path.cwd() / "_source/ZAIC-76Vibecoders"

# Optional path to the mounted directory containing train.jsonl and audio/.
# Example: Path("/kaggle/input/zalo-ai-challenge/train")
# Leave as None to use configs/environments/kaggle.json.
DATASET_ROOT_OVERRIDE: Path | None = None

# %% [markdown]
# ## 1. Bootstrap source from GitHub
#
# In an IPython notebook, `__file__` is not defined. Project discovery checks the
# working directory and `ASR_PROJECT_ROOT`; when neither contains the source, it
# sparse-clones the configured branch into Kaggle working storage.

# %%

def find_project_root() -> Path:
    explicit_root = os.environ.get("ASR_PROJECT_ROOT")
    candidates = [Path(explicit_root).expanduser()] if explicit_root else []
    candidates.append(Path.cwd())
    script_file = globals().get("__file__")
    if script_file:
        candidates.extend(Path(script_file).resolve().parents)
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


def _run_git(args: list[str], *, capture: bool = False) -> str:
    try:
        result = subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to clone the benchmark source.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{detail}") from exc
    return result.stdout if capture else ""


def refresh_managed_source(source_dir: Path) -> Path:
    """Update an existing source-manager checkout to the configured branch head."""
    marker_path = source_dir / ".source_manager.json"
    if not marker_path.is_file():
        return source_dir

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    current_commit = _run_git(["git", "-C", str(source_dir), "rev-parse", "HEAD"], capture=True).strip()
    if (marker.get("repo_url"), marker.get("requested_ref"), marker.get("commit")) != (
        SOURCE_REPO_URL, SOURCE_REF, current_commit
    ):
        raise ValueError(f"Source marker does not match the existing checkout: {source_dir}")

    changes = _run_git(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"], capture=True
    ).strip()
    if changes:
        raise ValueError(f"Managed source has local tracked changes; refusing to replace them: {source_dir}")

    _run_git(["git", "-C", str(source_dir), "fetch", "--depth=1", "origin", SOURCE_REF])
    remote_commit = _run_git(["git", "-C", str(source_dir), "rev-parse", "FETCH_HEAD"], capture=True).strip()
    if remote_commit != current_commit:
        _run_git(["git", "-C", str(source_dir), "reset", "--hard", "FETCH_HEAD"])
        current_commit = remote_commit

    marker["commit"] = current_commit
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return source_dir


def clone_source() -> Path:
    """Sparse-clone the configured source and leave a source_manager marker."""
    source_dir = Path(os.environ.get("ASR_SOURCE_DIR", str(SOURCE_DIR_DEFAULT))).expanduser().resolve()
    marker_name = ".source_manager.json"
    marker_path = source_dir / marker_name

    if source_dir.exists():
        if source_dir.is_symlink() or not source_dir.is_dir() or not (source_dir / ".git").is_dir():
            raise ValueError(f"Refusing to reuse a non-managed source path: {source_dir}")
        remote = _run_git(["git", "-C", str(source_dir), "remote", "get-url", "origin"], capture=True).strip()
        if remote != SOURCE_REPO_URL:
            raise ValueError(f"Existing source has a different GitHub origin: {source_dir}")
        commit = _run_git(["git", "-C", str(source_dir), "rev-parse", "HEAD"], capture=True).strip()
        if not (source_dir / "src/benchmark_baseline_asr/configs/experiment.json").is_file():
            raise ValueError(f"Benchmark source is missing from the existing checkout: {source_dir}")
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (marker.get("repo_url"), marker.get("requested_ref"), marker.get("commit")) != (
                SOURCE_REPO_URL, SOURCE_REF, commit
            ):
                raise ValueError(f"Source marker does not match the existing checkout: {source_dir}")
        else:
            marker_path.write_text(
                json.dumps(
                    {"version": 1, "repo_url": SOURCE_REPO_URL, "requested_ref": SOURCE_REF, "commit": commit},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        os.environ["ASR_SOURCE_PATH"] = str(source_dir)
        return refresh_managed_source(source_dir)

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".asr-source-download-", dir=source_dir.parent))
    shutil.rmtree(staging)
    try:
        _run_git([
            "git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", "--no-tags",
            SOURCE_REPO_URL, str(staging),
        ])
        _run_git(["git", "-C", str(staging), "fetch", "--depth=1", "origin", SOURCE_REF])
        _run_git(["git", "-C", str(staging), "sparse-checkout", "set", "--no-cone", *SOURCE_INCLUDE_PATHS])
        _run_git(["git", "-C", str(staging), "checkout", "--detach", "FETCH_HEAD"])
        commit = _run_git(["git", "-C", str(staging), "rev-parse", "HEAD"], capture=True).strip()
        (staging / marker_name).write_text(
            json.dumps(
                {"version": 1, "repo_url": SOURCE_REPO_URL, "requested_ref": SOURCE_REF, "commit": commit},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(source_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    os.environ["ASR_SOURCE_PATH"] = str(source_dir)
    return source_dir


# %% [markdown]
# ## 2. Add the checkout to Python's import path

# %%

try:
    PROJECT_ROOT = find_project_root()
except FileNotFoundError:
    PROJECT_ROOT = clone_source()

PROJECT_ROOT = refresh_managed_source(PROJECT_ROOT)

os.environ["ASR_PROJECT_ROOT"] = str(PROJECT_ROOT)
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

# %% [markdown]
# ## 3. Prepare configuration for this run

# %%


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


# %% [markdown]
# ## 4. Validate the mounted dataset and configured models

# %%

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


# %% [markdown]
# ## 5. Check GPU and display helpers

# %%

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


# %% [markdown]
# ## 6. Smoke test all four models
#
# After preparing the manifest and downloading model snapshots, run the same
# median-duration audio sample through every model on GPU 0. Models run one at a
# time and are unloaded after each check, so this verifies loading and inference
# without starting the full 1,600-sample benchmark if a model is unavailable.

# %%
def run_model_smoke_test(config_path: Path, samples: list[Any], resolved_models: dict[str, dict]) -> list[dict]:
    from benchmark_baseline_asr.application.config_service import resolve_experiment
    from benchmark_baseline_asr.application.audio_service import prepare_audio
    from benchmark_baseline_asr.infrastructure.models.factory import create_adapter
    from benchmark_baseline_asr.infrastructure.runtime.reproducibility import configure_determinism
    from benchmark_baseline_asr.infrastructure.runtime.timing import timed_call

    config = resolve_experiment(config_path, ENVIRONMENT)
    if not samples:
        raise ValueError("Cannot smoke-test ASR models: the prepared dataset has no samples.")
    sample = sorted(samples, key=lambda row: (row.duration_seconds, row.sample_id))[len(samples) // 2]
    audio = prepare_audio(sample.audio_path, config["_loaded"]["audio"])
    runtime = dict(config["_loaded"]["runtime"])
    runtime["device"] = "cuda:0"
    configure_determinism(runtime.get("seed", 0), runtime.get("cpu_threads_per_worker", 2))

    results = []
    for model in config["models"]:
        model_key = model["key"]
        model_config = model["settings"]
        adapter = create_adapter(model_config["adapter"])
        try:
            adapter.load(resolved_models[model_key]["path"], model_config, runtime)
            transcription, latency_seconds = timed_call(
                lambda: adapter.transcribe(audio), device=runtime["device"]
            )
            results.append(
                {
                    "model": model_key,
                    "sample_id": sample.sample_id,
                    "text": transcription.text,
                    "latency_seconds": latency_seconds,
                }
            )
        finally:
            adapter.unload()

    print("\nSmoke-test results (one sample per model):")
    for row in results:
        print(f"- {row['model']}: {row['latency_seconds']:.2f}s — {row['text'][:240]}")
    return results


# %% [markdown]
# ## 7. Run the benchmark workflows

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

        print("\n[1/6] Preparing manifest and two duration-balanced shards...")
        run_dir, samples, shards = prepare(str(config_path), ENVIRONMENT, RUN_ID)
        print(f"Prepared {len(samples)} samples across {len(shards)} shards.")

        print("\n[2/6] Checking Kaggle HF_TOKEN and downloading model snapshots...")
        token = get_huggingface_token("HF_TOKEN")
        try:
            _, resolved_models = download(str(config_path), ENVIRONMENT, RUN_ID, token=token)
        finally:
            del token

        print("\n[3/6] Smoke-testing all four models on one representative sample...")
        run_model_smoke_test(config_path, samples, resolved_models)

        print("\n[4/6] Transcribing the full train split with all four models on the two GPUs...")
        transcribe(str(config_path), ENVIRONMENT, RUN_ID)

        print("\n[5/6] Calculating raw and normalized CER/WER...")
        evaluate(str(config_path), RUN_ID, ENVIRONMENT)

        print("\n[6/6] Comparing model metrics and writing reports...")
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
