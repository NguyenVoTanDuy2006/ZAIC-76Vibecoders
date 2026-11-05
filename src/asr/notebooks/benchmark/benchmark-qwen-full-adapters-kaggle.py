# %% [markdown]
# # Qwen ASR 1.7B — PiSSA full và AdaLoRA full trên Kaggle
#
# **Chuẩn bị:** Add Input dataset Zalo có `train.jsonl` và `audio/`.
# Add Input hai Kaggle Models `nvtd1796/qwen3-asr-vimd-adapters`:
# `pytorch/pissa-full/1` và `pytorch/adalora-full/1`.
# Đọc model_id, revision, adapter_path, dtype từ JSON giống module Research.
# adapter_path trỏ thẳng tới folder chứa adapter_config.json và adapter_model.safetensors.
# Bật Internet, chọn GPU T4 ×2 (một GPU cũng được), thêm Secret `HF_TOKEN` nếu cần.
#
# Script tự lấy source benchmark đã khóa commit, không cần upload repo hoặc `.env`.
# Chạy lần lượt: chuẩn bị → smoke test hai mẫu → toàn bộ train → xem kết quả.
# Chỉ benchmark các mẫu trong manifest, giữ audio gốc; WER/CER, latency, RTF,
# coverage và resume. Không bật BERTScore/DNSMOS.

# %%
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/qwen-adapter-benchmark")
SCRATCH_ROOT = Path("/kaggle/temp/qwen-adapter-benchmark-configs")
DATASET_ROOT_OVERRIDE = None  # Folder chứa train.jsonl và audio/
MODEL_CONFIG_PATHS_OVERRIDE = None  # Hoặc [Path(.../pissa.json), Path(.../adalora.json)] trong Input
# Mặc định adapter_path đã đúng theo hai config Research. Chỉ sửa khi mount khác:
ADAPTER_PATH_OVERRIDES = {}  # {"qwen_vimd_pissa_full": Path("/kaggle/input/.../pissa-full/1"), ...}
PYTHON312_OVERRIDE = None     # Không cần đặt khi để script tự chuẩn bị Python
GPU_IDS = None               # None = dùng các GPU nhìn thấy; hoặc [0], [0, 1]
RUN_ID = "qwen-full-train-config-v1"  # Giữ nguyên để resume trong cùng session
INSTALL_DEPENDENCIES = True
HF_SECRET_LABEL = "HF_TOKEN"
SOURCE_REPO = "https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git"
SOURCE_COMMIT = "9a96f3aaf38d428031144678a4fd33605af9ff16"
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


def benchmark_package(source_root):
    """Return the import path for either the current or pinned historical layout."""
    for package, directory in (
        ("asr.benchmark", "asr/benchmark"),
        ("asr_benchmark", "asr_benchmark"),
        ("benchmark_baseline_asr", "benchmark_baseline_asr"),
    ):
        if (Path(source_root) / "src" / directory / "workflow/benchmark_qwen_adapters.py").is_file():
            return package
    raise FileNotFoundError("Source thiếu module benchmark Qwen adapters")


def benchmark_requirements(source_root):
    """Use the package-local file, or the path from the pinned historical SHA."""
    source_root = Path(source_root)
    for relative in ("src/asr/benchmark/requirements.txt",
                     "requirements/qwen-adapter-benchmark.txt"):
        path = source_root / relative
        if path.is_file():
            return path
    raise FileNotFoundError("Source thiếu requirements cho benchmark Qwen adapters")


def resolve_model_configs(source_root, target, *, config_paths=None, adapter_overrides=None):
    config_root = Path(source_root) / "src" / benchmark_package(source_root).replace(".", "/") / "configs/models"
    config_paths = ([config_root / f"qwen_vimd_{method}_full.json" for method in ("pissa", "adalora")]
                    if config_paths is None else [Path(p).expanduser().resolve() for p in config_paths])
    documents = [json.loads(p.read_text(encoding="utf-8")) for p in config_paths]
    overrides = adapter_overrides or {}
    unknown = set(overrides) - {d["model"]["key"] for d in documents}
    if unknown:
        raise ValueError(f"Unknown adapter override keys: {sorted(unknown)}")
    for path, document in zip(config_paths, documents):
        spec = document["model"]
        checkpoint = Path(overrides.get(spec["key"], spec["adapter_path"])).expanduser()
        checkpoint = (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(f"Add Input model {spec['key']}; thiếu {filename} tại {checkpoint}. "
                                        "Kiểm tra ADAPTER_PATH_OVERRIDES nếu đường dẫn mount khác.")
        spec["adapter_path"] = str(checkpoint)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for document in documents:
        key = document["model"]["key"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ValueError("model.key phải là tên file đơn giản")
        path = target / f"{key}.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def bootstrap_source(target, repo, revision):
    target = Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        checked(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", "--sparse", repo, target])
        checked(["git", "-C", target, "fetch", "--quiet", "--depth=1", "origin", revision])
        # SOURCE_COMMIT predates the package rename; fetch either layout without
        # modifying the pinned checkout or claiming a different source revision.
        checked(["git", "-C", target, "sparse-checkout", "set", "--no-cone",
                 "/src/asr/", "/src/asr_benchmark/", "/src/benchmark_baseline_asr/", "/requirements/"])
        checked(["git", "-C", target, "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    origin = checked(["git", "-C", target, "remote", "get-url", "origin"], capture=True)
    commit = checked(["git", "-C", target, "rev-parse", "HEAD"], capture=True)
    changes = checked(["git", "-C", target, "status", "--porcelain", "--untracked-files=no"], capture=True)
    if origin != repo or commit != revision or changes:
        raise ValueError("Source có thay đổi hoặc sai revision; chọn SCRATCH_ROOT mới.")
    benchmark_package(target)
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
    command = [VENV_PYTHON, "-m", f"{benchmark_package(SOURCE_ROOT)}.workflow.benchmark_qwen_adapters",
               "--model-configs", *MODEL_CONFIG_FILES, "--dataset-root", DATASET_ROOT,
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
             f"import sys; from {benchmark_package(SOURCE_ROOT)}.workflow.benchmark_qwen_adapters import run_adapter_benchmark; "
             "run_adapter_benchmark(sys.argv[1], sys.argv[2])", config_path, run_id], cwd=SOURCE_ROOT, env=child_env)

# %% [markdown]
# ## 1. Input, source và token
#
# Hai model JSON mặc định được lấy cùng source, giữ đúng cấu trúc Research.
# adapter_path trỏ trực tiếp tới Kaggle Model version chứa hai file PEFT.
# Không cần best_epoch.txt hoặc run_config.json. Base revision được khóa trong JSON.
# Các config sau override được ghi vào thư mục tạm; Input và source không bị sửa.

# %%
if __name__ == "__main__":
    if not OUTPUT_ROOT.resolve().is_relative_to(Path("/kaggle/working")):
        raise ValueError("OUTPUT_ROOT phải nằm dưới /kaggle/working")
    if not SCRATCH_ROOT.resolve().is_relative_to(Path("/kaggle/temp")):
        raise ValueError("SCRATCH_ROOT phải nằm dưới /kaggle/temp")
    DATASET_ROOT = discover_train_root(INPUT_ROOT, DATASET_ROOT_OVERRIDE)
    bootstrap_source(SOURCE_ROOT, SOURCE_REPO, SOURCE_COMMIT)
    MODEL_CONFIG_FILES = resolve_model_configs(
        SOURCE_ROOT, SCRATCH_ROOT / "model-configs", config_paths=MODEL_CONFIG_PATHS_OVERRIDE,
        adapter_overrides=ADAPTER_PATH_OVERRIDES,
    )
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
    print("Train:", DATASET_ROOT)
    for path in MODEL_CONFIG_FILES:
        model_spec = json.loads(path.read_text())["model"]
        print(model_spec["key"], model_spec["adapter_path"], "base revision:", model_spec["revision"])
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
                 benchmark_requirements(SOURCE_ROOT), "-c", constraint], env=child_env)
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
# Kiểm tra adapter theo config, SHA-256, base revision và mọi WAV trước khi inference.

# %%
if __name__ == "__main__":
    CONFIG_PATH, RUN_DIR = prepare_run(RUN_ID)
    manifest = [json.loads(line) for line in (RUN_DIR / "manifest.jsonl").read_text().splitlines() if line]
    identity = json.loads((RUN_DIR / "adapter_benchmark.json").read_text())
    print("Samples:", len(manifest), "Audio hours:", round(sum(r["duration_seconds"] for r in manifest) / 3600, 3))
    for model in identity["models"]:
        print(model["key"], model["settings"].get("display_name", ""),
              "base revision", model["settings"]["revision"])

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
