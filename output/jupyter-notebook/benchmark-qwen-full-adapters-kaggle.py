# %% [markdown]
# # Qwen ASR 1.7B — PiSSA full và AdaLoRA full trên Kaggle
#
# **Chuẩn bị:** Add Input dataset Zalo có `train.jsonl` và `audio/`, cùng hai
# thư mục `qwen_pissa_full`, `qwen_adalora_full`. Mỗi thư mục model cần giữ
# `best_epoch.txt`, metadata run và thư mục `epoch_*_adapter` tương ứng.
# Hai model có thể nằm trong hai Kaggle datasets khác nhau.
# Bật Internet, chọn GPU T4 ×2 (một GPU cũng được), thêm Secret `HF_TOKEN` nếu cần.
#
# Script tự lấy source benchmark đã khóa commit, không cần upload repo hoặc `.env`.
# Chạy lần lượt: chuẩn bị → smoke test hai mẫu → toàn bộ train → xem kết quả.
# Chỉ benchmark các mẫu trong manifest, giữ audio gốc; WER/CER, latency, RTF,
# coverage và resume. Không bật BERTScore/DNSMOS.

# %%
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/qwen-adapter-benchmark")
SCRATCH_ROOT = Path("/kaggle/temp/qwen-adapter-benchmark")
DATASET_ROOT_OVERRIDE = None  # Folder chứa train.jsonl và audio/
PISSA_RUN_OVERRIDE = None     # Folder chứa best_epoch.txt của PiSSA FULL
ADALORA_RUN_OVERRIDE = None   # Folder chứa best_epoch.txt của AdaLoRA FULL
PYTHON312_OVERRIDE = None     # Không cần đặt khi để script tự chuẩn bị Python
GPU_IDS = None               # None = dùng các GPU nhìn thấy; hoặc [0], [0, 1]
RUN_ID = "qwen-full-train-v1"  # Giữ nguyên để resume trong cùng session
INSTALL_DEPENDENCIES = True
HF_SECRET_LABEL = "HF_TOKEN"
SOURCE_REPO = "https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git"
SOURCE_COMMIT = "d564ee3d581e3a73f3f182b56327abcf20865841"
SOURCE_ROOT = SCRATCH_ROOT / "source"
VENV_PYTHON = SCRATCH_ROOT / "venv/bin/python"
BASE_CACHE = SCRATCH_ROOT / "base-models"


def checked(command, *, cwd=None, env=None, capture=False):
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def discover_train_root(input_root, override=None):
    def valid(path):
        return (path / "train.jsonl").is_file() and (path / "audio").is_dir()
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not valid(path):
            raise FileNotFoundError(f"DATASET_ROOT_OVERRIDE phải chứa train.jsonl và audio/: {path}")
        return path
    candidates = sorted({p.parent.resolve() for p in Path(input_root).rglob("train.jsonl") if valid(p.parent)})
    if len(candidates) != 1:
        raise ValueError(f"Đặt DATASET_ROOT_OVERRIDE; tìm thấy {len(candidates)} bộ train: {candidates}")
    return candidates[0]


def discover_adapter_run(input_root, method, override=None):
    label = f"{method.upper()}_RUN_OVERRIDE"
    expected = "LORA" if method == "pissa" else "ADALORA"
    def valid(path):
        try:
            epoch = int((path / "best_epoch.txt").read_text().strip())
            checkpoint = path / f"epoch_{epoch}_adapter"
            config = json.loads((checkpoint / "adapter_config.json").read_text())
            return epoch > 0 and config.get("peft_type") == expected and (checkpoint / "adapter_model.safetensors").is_file()
        except (OSError, ValueError, TypeError):
            return False
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not valid(path):
            raise ValueError(f"{label} cần trỏ tới run có best_epoch.txt và adapter {expected}: {path}")
        return path
    candidates = sorted({p.resolve() for p in Path(input_root).rglob(f"qwen_{method}_full") if valid(p)})
    if len(candidates) != 1:
        raise ValueError(f"Đặt {label}; tìm thấy {len(candidates)} run full phù hợp: {candidates}")
    return candidates[0]


def stage_adapter_links(pissa_run, adalora_run, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for method, source in (("pissa", pissa_run), ("adalora", adalora_run)):
        link = target / f"qwen_{method}_full"
        source = Path(source).resolve()
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source:
                raise ValueError(f"Adapter Input đã đổi; đặt SCRATCH_ROOT mới: {link}")
        else:
            link.symlink_to(source, target_is_directory=True)
    return target.resolve()


def bootstrap_source(target, repo, revision):
    target = Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        checked(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", "--sparse", repo, target])
        checked(["git", "-C", target, "fetch", "--quiet", "--depth=1", "origin", revision])
        checked(["git", "-C", target, "sparse-checkout", "set", "src/benchmark_baseline_asr", "requirements"])
        checked(["git", "-C", target, "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    origin = checked(["git", "-C", target, "remote", "get-url", "origin"], capture=True)
    commit = checked(["git", "-C", target, "rev-parse", "HEAD"], capture=True)
    changes = checked(["git", "-C", target, "status", "--porcelain", "--untracked-files=no"], capture=True)
    if origin != repo or commit != revision or changes:
        raise ValueError("Source có thay đổi hoặc sai revision; chọn SCRATCH_ROOT mới.")
    if not (target / "src/benchmark_baseline_asr/workflow/benchmark_qwen_adapters.py").is_file():
        raise FileNotFoundError("Source thiếu module benchmark Qwen adapters")
    (target / ".source_manager.json").write_text(json.dumps({
        "repo_url": repo, "requested_ref": revision, "commit": commit,
    }, indent=2) + "\n")
    return target


def ensure_pip(python):
    def available():
        return subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True).returncode == 0
    if available():
        return
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], capture_output=True)
    if not available():
        checked([sys.executable, "-m", "pip", "--python", python, "install", "pip"])
    checked([python, "-m", "pip", "--version"])


def find_python312():
    def valid(candidate):
        return candidate and Path(candidate).is_file() and checked(
            [candidate, "-c", "import sys; print(sys.version_info[:2] == (3, 12))"], capture=True) == "True"
    if PYTHON312_OVERRIDE is not None:
        if not valid(PYTHON312_OVERRIDE):
            raise ValueError("PYTHON312_OVERRIDE phải là Python 3.12")
        return Path(PYTHON312_OVERRIDE)
    standalone = SCRATCH_ROOT / "python312/bin/python3"
    for candidate in (sys.executable, shutil.which("python3.12"), standalone):
        if valid(candidate):
            return Path(candidate)
    checked([sys.executable, "-m", "pip", "install", "pbs-installer[all]>=2025.10.7"])
    checked([sys.executable, "-m", "pbs_installer", "3.12", "--destination", standalone.parent.parent])
    if not valid(standalone):
        raise RuntimeError("Không chuẩn bị được Python 3.12; kiểm tra log cài đặt")
    return standalone


def prepare_run(run_id, *, limit=None):
    command = [VENV_PYTHON, "-m", "benchmark_baseline_asr.workflow.benchmark_qwen_adapters",
               "--models-root", MODELS_ROOT, "--dataset-root", DATASET_ROOT,
               "--output-root", OUTPUT_ROOT, "--run-id", run_id, "--prepare-only",
               "--gpu-ids", *GPU_IDS]
    if limit is not None:
        command.extend(["--limit", limit])
    checked(command, cwd=SOURCE_ROOT, env=child_env)
    # Config là artifact của run: đặt base weights ngoài /kaggle/working để Output
    # chỉ chứa báo cáo/prediction. Cả hai adapter và smoke/full dùng cùng cache.
    config_root = OUTPUT_ROOT / "configs" / run_id
    download_path = config_root / "download.json"
    download = json.loads(download_path.read_text())
    download["models_dir"] = str(BASE_CACHE)
    download_path.write_text(json.dumps(download, indent=2) + "\n")
    return config_root / "experiment.json", OUTPUT_ROOT / "runs" / run_id


def infer_run(config_path, run_id):
    checked([VENV_PYTHON, "-c",
             "import sys; from benchmark_baseline_asr.workflow.benchmark_qwen_adapters import run_adapter_benchmark; "
             "run_adapter_benchmark(sys.argv[1], sys.argv[2])", config_path, run_id], cwd=SOURCE_ROOT, env=child_env)

# %% [markdown]
# ## 1. Input, source và token
#
# Các override trỏ tới thư mục run, không trỏ thẳng tới `epoch_2_adapter`.
# Giữ `run_config.json` của AdaLoRA và `training_method.json` của PiSSA để khóa
# đúng base/revision. Các symlink được tạo ở thư mục tạm, không ghi vào Input.

# %%
if __name__ == "__main__":
    if not OUTPUT_ROOT.resolve().is_relative_to(Path("/kaggle/working")):
        raise ValueError("OUTPUT_ROOT phải nằm dưới /kaggle/working")
    if not SCRATCH_ROOT.resolve().is_relative_to(Path("/kaggle/temp")):
        raise ValueError("SCRATCH_ROOT phải nằm dưới /kaggle/temp")
    DATASET_ROOT = discover_train_root(INPUT_ROOT, DATASET_ROOT_OVERRIDE)
    pissa_run = discover_adapter_run(INPUT_ROOT, "pissa", PISSA_RUN_OVERRIDE)
    adalora_run = discover_adapter_run(INPUT_ROOT, "adalora", ADALORA_RUN_OVERRIDE)
    MODELS_ROOT = stage_adapter_links(pissa_run, adalora_run, SCRATCH_ROOT / "adapter-links")
    bootstrap_source(SOURCE_ROOT, SOURCE_REPO, SOURCE_COMMIT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(PYTHONPATH=str(SOURCE_ROOT / "src"), PYTHONUNBUFFERED="1",
                     HF_HOME=str(SCRATCH_ROOT / "huggingface"), ASR_SOURCE_PATH=str(SOURCE_ROOT),
                     PIP_NO_CACHE_DIR="1")
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(HF_SECRET_LABEL)
        if token and token.strip():
            child_env["HF_TOKEN"] = token.strip()
        del token
    except Exception:
        pass  # Base công khai; có thể dùng HF_TOKEN từ environment nếu Secret chưa được gắn.
    print("Train:", DATASET_ROOT, "\nPiSSA:", pissa_run, "\nAdaLoRA:", adalora_run)
    print("Output:", OUTPUT_ROOT, "\nSource commit:", SOURCE_COMMIT)
    print("HF_TOKEN configured:", bool(child_env.get("HF_TOKEN")))

# %% [markdown]
# ## 2. Môi trường riêng bằng pip
#
# Python 3.12 được chuẩn bị riêng nếu kernel Kaggle là Python 3.13.
# Tạo venv không cần pip có sẵn, sau đó tự seed/kiểm tra pip cả khi venv đã tồn tại.
# Dùng Torch 2.6.0 CUDA 12.4, FP16 và SDPA; không cần flash-attn.
# Torch được khóa để pip không tự nâng sang bản CUDA mới hơn khi cài Transformers/PEFT.
# Tham chiếu wheel: https://pytorch.org/get-started/previous-versions/#v260
#
# Nếu đã cài xong trong session, đặt INSTALL_DEPENDENCIES=False.

# %%
if __name__ == "__main__":
    if not VENV_PYTHON.is_file():
        checked([find_python312(), "-m", "venv", "--without-pip", VENV_PYTHON.parent.parent])
    ensure_pip(VENV_PYTHON)
    if INSTALL_DEPENDENCIES:
        constraint = SCRATCH_ROOT / "torch-constraint.txt"
        constraint.write_text("torch==2.6.0\n")
        checked([VENV_PYTHON, "-m", "pip", "install", "torch==2.6.0", "--index-url",
                 "https://download.pytorch.org/whl/cu124"], env=child_env)
        checked([VENV_PYTHON, "-m", "pip", "install", "-r",
                 SOURCE_ROOT / "requirements/qwen-adapter-benchmark.txt", "-c", constraint], env=child_env)
    probe = json.loads(checked([VENV_PYTHON, "-c",
        "import json, torch, transformers, peft; print(json.dumps({"
        "'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "
        "'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}))"],
        capture=True, env=child_env))
    print(probe)
    if not probe["gpus"]:
        raise RuntimeError("Chọn accelerator GPU trong Kaggle Session options rồi chạy lại")
    GPU_IDS = list(range(len(probe["gpus"]))) if GPU_IDS is None else GPU_IDS

# %% [markdown]
# ## 3. Kiểm tra toàn bộ train
#
# Mặc định không giới hạn số mẫu. Dữ liệu hiện tại có 1.545 mẫu (~3,10 giờ).
# Kiểm tra checkpoint, SHA-256, base revision và mọi WAV trước khi inference.

# %%
if __name__ == "__main__":
    CONFIG_PATH, RUN_DIR = prepare_run(RUN_ID)
    manifest = [json.loads(line) for line in (RUN_DIR / "manifest.jsonl").read_text().splitlines() if line]
    identity = json.loads((RUN_DIR / "adapter_benchmark.json").read_text())
    print("Samples:", len(manifest), "Audio hours:", round(sum(r["duration_seconds"] for r in manifest) / 3600, 3))
    for model in identity["models"]:
        print(model["key"], "epoch", model["settings"]["epoch"], "base revision", model["settings"]["revision"])

# %% [markdown]
# ## 4. Smoke test hai mẫu với cả hai adapter
#
# Run ID riêng; nếu lỗi, sửa trước khi chạy full. Lần đầu sẽ tải base model.

# %%
if __name__ == "__main__":
    smoke_id = RUN_ID + "-smoke"
    smoke_config, smoke_dir = prepare_run(smoke_id, limit=2)
    infer_run(smoke_config, smoke_id)
    smoke_results = json.loads((smoke_dir / "comparison.json").read_text())
    assert all(r["result_complete"] and r["successful_samples"] == 2 for r in smoke_results)
    print("Smoke test passed for both adapters")

# %% [markdown]
# ## 5. Full benchmark train và ZIP kết quả
#
# Chạy lại cùng RUN_ID để tiếp tục prediction còn thiếu trong session. Giữ cấu hình
# và trọng số cố định; đổi RUN_ID khi đổi đầu vào. Session mới mất thư mục tạm;
# chạy lại sẽ cài môi trường/tải base lại. ZIP được lưu cả khi inference bị lỗi,
# để tải các prediction/log đã có. ZIP không chứa model, token, source hoặc venv.

# %%
if __name__ == "__main__":
    from IPython.display import FileLink, display
    try:
        infer_run(CONFIG_PATH, RUN_ID)
    finally:
        zip_path = shutil.make_archive(str(OUTPUT_ROOT / RUN_ID), "zip", root_dir=RUN_DIR)
        print("Results ZIP:", zip_path)
        display(FileLink(zip_path))

# %% [markdown]
# ## 6. So sánh WER/CER và nghe theo sample
#
# WER/CER được tính theo tổng edits / tổng đơn vị tham chiếu. Normalization giữ
# dấu và cách viết số, chuyển chữ thường, bỏ dấu câu, gộp khoảng trắng; CER tính
# cả khoảng trắng. Chỉ kết luận full benchmark khi result_complete=True.

# %%
if __name__ == "__main__":
    import pandas as pd
    from IPython.display import Audio
    comparison = pd.DataFrame(json.loads((RUN_DIR / "comparison.json").read_text()))
    columns = ["model", "successful_samples", "coverage", "result_complete", "normalized_wer", "normalized_cer",
               "raw_wer", "raw_cer", "latency_mean_seconds", "latency_p95_seconds", "rtf_sum_latency"]
    display(comparison[columns].sort_values("normalized_wer"))
    assert comparison["result_complete"].all(), "Chưa đủ mẫu; kiểm tra worker_*/errors.jsonl rồi resume."
    SAMPLE_ID = "train_0001"
    sample = next(r for r in manifest if r["sample_id"] == SAMPLE_ID)
    print("Reference:", sample["reference"])
    display(Audio(filename=sample["audio_path"]))
    for model in identity["models"]:
        rows = [json.loads(line) for line in (RUN_DIR / model["key"] / "predictions.jsonl").read_text().splitlines() if line]
        row = next((r for r in rows if r["sample_id"] == SAMPLE_ID), None)
        print(model["key"], ":", row["text"] if row else "Chưa có prediction")
