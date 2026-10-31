# %% [markdown]
# # Audio Enhancement Survey — Kaggle
#
# Bản clone của `audio-enhancement-survey.ipynb`, chạy độc lập trên Kaggle, đọc audio và transcript
# từ dataset đã gắn trong **Input**. Không cần clone repo dự án hoặc mang theo `.env` local.
#
# **Chuẩn bị trên Kaggle:**
#
# 1. Import notebook này vào Kaggle.
# 2. Add Input dataset `khangl123/zalo-ai` hoặc dataset có `train.jsonl` và thư mục `audio/`.
# 3. Bật **Internet** để cài môi trường và tải model ở lần chạy đầu.
# 4. Chọn GPU nếu muốn thử model; các phép lọc nhẹ vẫn chạy được bằng CPU.
# 5. Nếu cần model có xác thực, thêm secret **HF_TOKEN** và cho phép notebook sử dụng secret đó.
# 6. Run All để khảo sát mặc định 10 dòng đầu manifest; chỉnh phương pháp và số mẫu ở mục 2.
#
# | Nội dung | Đường dẫn mặc định |
# |---|---|
# | Audio và nhãn đọc từ Kaggle Input | `/kaggle/input/.../train/` |
# | Audio xử lý, biểu đồ, điểm và log được lưu | `/kaggle/working/audio-enhancement-survey/runs/<run-id>/` |
# | Model, môi trường và source công cụ tạm thời | `/kaggle/temp/audio-enhancement-survey/` |
#
# Mỗi phương pháp nhận cùng audio gốc. Các bản xử lý được nhóm theo sample để nghe A/B.
# Notebook lưu thời gian xử lý, phiếu nghe CSV, DNSMOS và WER/CER tùy chọn; cuối notebook có ZIP
# chứa riêng kết quả của run để tải về từ Output.
#
# Đây là khảo sát thử nghiệm: model có thể thay đổi chi tiết lời nói. Không dùng bản gốc còn vang
# làm clean reference để tính PESQ/STOI/SI-SDR, và không suy SNR thật từ RMS.

# %% [markdown]
# ## 1. Dependency và dữ liệu Kaggle
#
# Cell đầu chỉ bổ sung các gói bảng/biểu đồ còn thiếu cho kernel. Dependency xử lý audio/model
# được cài vào các môi trường Python riêng ở mục 3.
# Cell tiếp theo tự tìm dataset đã gắn; chỉ cần đặt `DATA_ROOT_OVERRIDE` nếu có nhiều bộ train
# hoặc Kaggle mount dataset bằng một đường dẫn khác.

# %%
import importlib.util
import subprocess
import sys

INSTALL_NOTEBOOK_DEPS = True
required = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing and INSTALL_NOTEBOOK_DEPS:
    command = [sys.executable, "-m", "pip", "install"]
    subprocess.run(command + missing, check=True)
    importlib.invalidate_caches()
elif missing:
    raise RuntimeError(f"Cài các gói còn thiếu vào kernel: {missing}")

# %%
import csv
import hashlib
import html
import json
import math
import os
import platform
import random
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from IPython.display import Audio, HTML, Markdown, display

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
DATA_ROOT_OVERRIDE = None      # Ví dụ Path("/kaggle/input/zalo-ai/train+public_test/train")
ARTIFACT_ROOT_OVERRIDE = None  # Mặc định lưu kết quả trong /kaggle/working
SCRATCH_ROOT_OVERRIDE = None   # Mặc định /kaggle/temp, không được lưu cùng Output của notebook

def discover_train_root(input_root, override=None):
    input_root = Path(input_root).expanduser()
    def valid(path):
        return (path / "train.jsonl").is_file() and (path / "audio").is_dir()
    if override is not None:
        candidate = Path(override).expanduser().resolve()
        if not valid(candidate):
            raise FileNotFoundError(f"DATA_ROOT_OVERRIDE phải chứa train.jsonl và audio/: {candidate}")
        return candidate
    preferred = [
        input_root / "datasets/khangl123/zalo-ai/train+public_test/train",
        input_root / "zalo-ai/train+public_test/train",
        input_root / "train+public_test/train",
    ]
    for candidate in preferred:
        if valid(candidate):
            return candidate.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError("Không thấy Kaggle Input. Import notebook lên Kaggle và Add Input dataset trước.")
    candidates = sorted({p.parent.resolve() for p in input_root.rglob("train.jsonl") if valid(p.parent)})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("Không thấy folder chứa train.jsonl và audio/. Add Input dataset hoặc đặt DATA_ROOT_OVERRIDE.")
    raise ValueError("Tìm thấy nhiều tập train; đặt DATA_ROOT_OVERRIDE thành một trong các đường dẫn:\n" + "\n".join(map(str, candidates)))

DATA_ROOT = discover_train_root(KAGGLE_INPUT_ROOT, DATA_ROOT_OVERRIDE)
MANIFEST_PATH = DATA_ROOT / "train.jsonl"
ARTIFACT_ROOT = (Path(ARTIFACT_ROOT_OVERRIDE).expanduser().resolve() if ARTIFACT_ROOT_OVERRIDE else
                 KAGGLE_WORKING_ROOT / "audio-enhancement-survey")
if not ARTIFACT_ROOT.resolve().is_relative_to(KAGGLE_WORKING_ROOT.resolve()):
    raise ValueError("Đặt ARTIFACT_ROOT_OVERRIDE dưới /kaggle/working để kết quả được lưu vào Output.")
SCRATCH_ROOT = (Path(SCRATCH_ROOT_OVERRIDE).expanduser().resolve() if SCRATCH_ROOT_OVERRIDE else
                Path("/kaggle/temp/audio-enhancement-survey"))
if SCRATCH_ROOT.resolve().is_relative_to(KAGGLE_INPUT_ROOT.resolve()):
    raise ValueError("SCRATCH_ROOT phải là thư mục ghi được, ngoài Kaggle Input.")
TOOL_ROOT = SCRATCH_ROOT / "tools"
MODEL_ROOT = SCRATCH_ROOT / "models"
RUN_DIR = ARTIFACT_ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
for path in [TOOL_ROOT, MODEL_ROOT, RUN_DIR]:
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PIP_CACHE_DIR", str(SCRATCH_ROOT / "pip-cache"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print("Train dataset:", DATA_ROOT)
print("Run output:", RUN_DIR)
print("Model / environment tạm:", SCRATCH_ROOT)
print("Kernel:", sys.executable)

# %% [markdown]
# ### GPU và Kaggle Secrets
#
# `USE_GPU=True` tự chọn GPU nếu có; khi không có GPU, các phép xử lý nhẹ vẫn chạy CPU.
# Với nhiều GPU, chọn một bằng `GPU_INDEX`; mỗi worker dùng `cuda:0` sau khi ánh xạ GPU đó.
# `HF_TOKEN` là tùy chọn cho khảo sát mặc định. SAM-Audio cần tài khoản có quyền checkpoint.
# Secret được đọc trong kernel chính rồi truyền vào worker qua môi trường; không ghi giá trị vào file.

# %%
USE_GPU = True
GPU_INDEX = "0"
HF_SECRET_LABEL = "HF_TOKEN"

gpu_rows = []
try:
    gpu_probe = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=15)
    if gpu_probe.returncode == 0:
        for row in csv.reader(gpu_probe.stdout.splitlines()):
            if len(row) == 3:
                gpu_rows.append({"index": row[0].strip(), "gpu": row[1].strip(), "memory_mb": row[2].strip()})
except (FileNotFoundError, subprocess.TimeoutExpired):
    pass
if USE_GPU and gpu_rows:
    if GPU_INDEX not in {row["index"] for row in gpu_rows}:
        raise ValueError(f"GPU_INDEX={GPU_INDEX} không có trong danh sách GPU.")
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print("Device cho model:", DEVICE)
if gpu_rows:
    display(pd.DataFrame(gpu_rows))
else:
    print("Chưa có GPU; bật accelerator trong Session options nếu muốn chạy model nặng.")

HF_ACCESS_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_TOKEN_SOURCE = "environment" if HF_ACCESS_TOKEN else "not configured"
try:
    from kaggle_secrets import UserSecretsClient
    kaggle_token = UserSecretsClient().get_secret(HF_SECRET_LABEL)
    if kaggle_token and kaggle_token.strip():
        HF_ACCESS_TOKEN = kaggle_token.strip()
        HF_TOKEN_SOURCE = "Kaggle Secrets"
    del kaggle_token
except Exception:
    # Secret không tồn tại/không được cấp quyền không chặn khảo sát DSP và model công khai.
    pass
print("Nguồn HF token:", HF_TOKEN_SOURCE)

# %% [markdown]
# ## 2. Chọn mẫu và phương pháp
#
# Mặc định bật original, EQ nhẹ, RMS normalization, hai mức spectral gating và WPE.
# Đây là các thử nghiệm độc lập; RMS/VAD là đối chứng, không phải phương pháp chữa vang.
# Không có cách xử lý chắc chắn khôi phục âm tiết người nói đã nuốt.
#
# | Key | Mục tiêu | Môi trường |
# |---|---|---|
# | `original` | Bản gốc quy về mono 16 kHz để so sánh | dsp |
# | `eq_mild` | High-pass 70 Hz, giảm nhẹ vùng đục, tăng nhẹ vùng rõ | dsp |
# | `notch_50hz` | Tiếng ù 50 Hz; chỉ bật khi nghi có ù điện | dsp |
# | `rms_control` | Chuẩn hóa mức RMS, giới hạn gain | dsp |
# | `spectral_30`, `spectral_60` | Spectral gating non-stationary mức nhẹ/vừa | dsp |
# | `wpe` | Giảm vang bằng dự đoán tuyến tính, một kênh | dsp |
# | `vad_trim` | Silero VAD: chỉ cắt đầu/cuối, giữ khoảng nghỉ bên trong | silero |
# | `rnnoise` | RNNoise native, đổi qua RAW PCM 48 kHz đúng định dạng demo | dsp + binary |
# | `deepfilternet` | DeepFilterNet3, xử lý 48 kHz | deepfilter |
# | `frcrn`, `mossformergan` | ClearVoice 16 kHz | clearvoice |
# | `resemble_denoise`, `resemble_enhance` | Khử nhiễu / phục hồi giọng 44,1 kHz | resemble |
# | `sgmse_dereverb`, `sgmse_denoise` | Checkpoint giảm vang / khử nhiễu riêng biệt, 16 kHz | sgmse |
# | `sam_audio` | Tách `speech` và residual bằng SAM-Audio | sam |
#
# Các model có thể tự tải weights khi chạy lần đầu. SAM cần tài khoản đã được cấp quyền checkpoint.
# `HF_TOKEN` lấy từ Kaggle Secrets ở mục 1; không được ghi vào cấu hình hay output.
# SGMSE, SAM và phục hồi giọng nên thử trên GPU; mặc định chúng đều đang tắt.
#
# `RUN_DNSMOS`, `RUN_ASR` và `ASR_MODEL` được chọn ở cell cấu hình dưới đây; các mục 8/9 dùng lại lựa chọn này.

# %%
N_SAMPLES = 10
RUN_DNSMOS = False
RUN_ASR = False
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"  # Hoặc đường dẫn model đã gắn trong Kaggle Input
ASR_MAX_NEW_TOKENS = 448
ASR_REVISION = None
SAMPLE_IDS = []           # Ví dụ ["train_0001", "train_0005"]; [] = chọn theo chế độ dưới
SAMPLING = "first"       # "first" hoặc "random"; random có seed cố định
ENABLED = ["original", "eq_mild", "rms_control", "spectral_30", "spectral_60", "wpe"]
WARMUP = True            # Một lượt trên mẫu đầu, không tính vào latency từng clip
PROCESS_TIMEOUT_SECONDS = 3600
if DEVICE not in {"cpu", "cuda:0"}:
    raise ValueError("Dùng DEVICE='cuda:0' và chọn GPU vật lý bằng GPU_INDEX.")

SETTINGS = {
    "eq_mild": {"highpass_hz": 70, "mud_hz": 350, "mud_gain_db": -2, "clarity_hz": 2500, "clarity_gain_db": 2},
    "notch_50hz": {"frequency_hz": 50, "q": 30},
    "rms_control": {"target_dbfs": -23, "max_gain_db": 12},
    "spectral_30": {"prop_decrease": 0.3},
    "spectral_60": {"prop_decrease": 0.6},
    "wpe": {"fft_size": 512, "hop": 128, "taps": 10, "delay": 3, "iterations": 3},
    "vad_trim": {"threshold": 0.5, "speech_pad_ms": 150},
    "rnnoise": {"binary": str(TOOL_ROOT / "rnnoise/examples/rnnoise_demo")},
    "deepfilternet": {"model": "DeepFilterNet3", "atten_lim_db": 12},
    "frcrn": {"model": "FRCRN_SE_16K"},
    "mossformergan": {"model": "MossFormerGAN_SE_16K"},
    "resemble_denoise": {},
    "resemble_enhance": {"nfe": 32, "solver": "midpoint", "lambd": 0.5, "tau": 0.5},
    "sgmse_dereverb": {"repo": str(TOOL_ROOT / "sgmse"), "checkpoint": str(MODEL_ROOT / "sgmse_wsj0_reverb.ckpt"), "steps": 50, "snr": 0.33},
    "sgmse_denoise": {"repo": str(TOOL_ROOT / "sgmse"), "checkpoint": str(MODEL_ROOT / "sgmse_vbdmd.ckpt"), "steps": 30, "snr": 0.5},
    "sam_audio": {"model": "facebook/sam-audio-small", "prompt": "speech", "predict_spans": False, "reranking_candidates": 1, "bf16": False},
}
METHOD_ENV = {**dict.fromkeys(["original", "eq_mild", "notch_50hz", "rms_control", "spectral_30", "spectral_60", "wpe", "rnnoise"], "dsp"),
              "vad_trim": "silero", "deepfilternet": "deepfilter", "frcrn": "clearvoice", "mossformergan": "clearvoice",
              "resemble_denoise": "resemble", "resemble_enhance": "resemble", "sgmse_dereverb": "sgmse", "sgmse_denoise": "sgmse", "sam_audio": "sam"}
if "original" not in ENABLED:
    ENABLED.insert(0, "original")
unknown = set(ENABLED) - METHOD_ENV.keys()
if unknown:
    raise ValueError(f"Phương pháp không có adapter: {unknown}")

rows = [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
by_id = {str(r["id"]): r for r in rows}
if len(by_id) != len(rows):
    raise ValueError("Manifest có ID lặp.")
if SAMPLE_IDS:
    selected = [by_id[s] for s in SAMPLE_IDS]
elif SAMPLING == "random":
    selected = random.Random(SEED).sample(rows, min(N_SAMPLES, len(rows)))
elif SAMPLING == "first":
    selected = rows[:N_SAMPLES]
else:
    raise ValueError("SAMPLING phải là first hoặc random.")
SAMPLES = []
for r in selected:
    sample_id = str(r["id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id) or sample_id in {".", ".."}:
        raise ValueError("ID không phù hợp làm tên file.")
    audio_path = (DATA_ROOT / r["audio"]).resolve()
    if not audio_path.is_relative_to(DATA_ROOT.resolve()) or not audio_path.is_file():
        raise FileNotFoundError(f"Audio không hợp lệ cho {sample_id}")
    SAMPLES.append({"sample_id": sample_id, "audio_path": str(audio_path), "reference": r["audio_text"],
                    "input_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest()})
if not SAMPLES:
    raise ValueError("Chưa chọn được mẫu nào.")
if len({s["sample_id"] for s in SAMPLES}) != len(SAMPLES):
    raise ValueError("SAMPLE_IDS có mẫu lặp.")
(RUN_DIR / "samples.json").write_text(json.dumps(SAMPLES, ensure_ascii=False, indent=2), encoding="utf-8")
display(pd.DataFrame(SAMPLES)[["sample_id", "audio_path", "reference"]])

# %% [markdown]
# ## 3. Cài môi trường riêng trên Kaggle
#
# Để `INSTALL_ENVS=None` để tự cài đúng các nhóm cần cho `ENABLED`, `RUN_DNSMOS` và `RUN_ASR`.
# Đặt `[]` nếu đã cài trong session. Ví dụ chọn `frcrn` và `mossformergan` sẽ cài một môi trường ClearVoice.
# Venv, source và model được đặt ở `/kaggle/temp/audio-enhancement-survey/`, không nằm trong ZIP kết quả.
# Chúng có thể cần cài/tải lại khi Kaggle bắt đầu session mới; Internet cần bật cho bước này.
#
# Tất cả gói được cài bằng `python -m pip install`; môi trường riêng được tạo bằng `python -m venv`.
# Notebook tìm Python 3.11 có sẵn. Nếu máy chưa có, nó cài `pbs-installer[all]` bằng pip để tải
# Python 3.11 vào thư mục tạm. Có thể đặt `PYTHON311_OVERRIDE` hoặc dùng `PYTHON_OVERRIDES`
# cho các môi trường đã chuẩn bị. Python 3.11 giúp các phiên bản model cũ có wheel tương thích.
# Khi chọn RNNoise, notebook build binary chính thức. Khi chọn SGMSE, notebook tải checkpoint đúng nhiệm vụ
# (giảm vang khác khử nhiễu). SAM cần quyền checkpoint trên Hugging Face.
#
# Trước khi cài dependency, notebook kiểm tra pip cả trong venv đã tồn tại và tự bổ sung pip
# bằng ensurepip; nếu cần, dùng pip của kernel với `--python` để seed vào đúng môi trường.

# %%
INSTALL_ENVS = None  # None = tự suy từ cấu hình; [] = bỏ cài; hoặc danh sách nhóm
PYTHON_OVERRIDES = {}    # Ví dụ {"dsp": sys.executable, "asr": "/path/to/venv/bin/python"}
PYTHON311_OVERRIDE = None  # Đường dẫn python3.11 có sẵn, nếu muốn chỉ định
INSTALL_PYTHON311 = True   # Tải Python 3.11 bằng pbs-installer nếu máy chưa có
SGMSE_REVISION = "main"
SAM_REVISION = "main"
BUILD_RNNOISE = "rnnoise" in ENABLED
DOWNLOAD_SGMSE_CHECKPOINTS = any(m.startswith("sgmse_") for m in ENABLED)

def python_for(group):
    if group in PYTHON_OVERRIDES:
        return Path(PYTHON_OVERRIDES[group]).expanduser().absolute()
    return TOOL_ROOT / "venvs" / group / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def checked(command, cwd=None):
    subprocess.run([str(v) for v in command], cwd=cwd, check=True)

def ensure_pip(python):
    python = Path(python)

    def pip_available():
        result = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True, text=True)
        return result.returncode == 0

    if pip_available():
        return
    print(f"Bổ sung pip cho môi trường đã có: {python.parent.parent}")
    # Venv được tạo trước đây có thể chưa được seed pip. ensurepip dùng wheel đi kèm Python.
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], capture_output=True, text=True)
    if not pip_available():
        # pip của kernel có thể quản lý một interpreter khác dù interpreter đó chưa có pip.
        checked([sys.executable, "-m", "pip", "--python", python, "install", "pip"])
    checked([python, "-m", "pip", "--version"])

def find_python311():
    def is_python311(candidate):
        if candidate is None or not Path(candidate).is_file():
            return False
        try:
            version = subprocess.check_output(
                [str(candidate), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
                text=True, timeout=15,
            ).strip()
            return version == "3.11"
        except (OSError, subprocess.SubprocessError):
            return False

    if PYTHON311_OVERRIDE is not None:
        candidate = Path(PYTHON311_OVERRIDE).expanduser().absolute()
        if not is_python311(candidate):
            raise RuntimeError("PYTHON311_OVERRIDE phải trỏ tới Python 3.11 hoạt động được.")
        return candidate
    standalone_root = TOOL_ROOT / "python311"
    standalone_python = standalone_root / "bin/python3"
    candidates = [sys.executable, shutil.which("python3.11"), standalone_python]
    for candidate in candidates:
        if is_python311(candidate):
            return Path(candidate).absolute()
    if not INSTALL_PYTHON311:
        raise RuntimeError("Chưa có Python 3.11: đặt PYTHON311_OVERRIDE hoặc bật INSTALL_PYTHON311.")
    print("Chuẩn bị Python 3.11 cho các môi trường model …")
    checked([sys.executable, "-m", "pip", "install", "pbs-installer[all]>=2025.10.7"])
    checked([sys.executable, "-m", "pbs_installer", "3.11", "--destination", standalone_root])
    if not is_python311(standalone_python):
        raise RuntimeError("Chưa tìm được Python 3.11 sau bước cài đặt; kiểm tra log cài Python.")
    return standalone_python

def checkout(url, destination, revision):
    destination = Path(destination)
    if not destination.exists():
        checked(["git", "clone", url, destination])
    checked(["git", "checkout", revision], cwd=destination)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()

packages = {
    "dsp": ["numpy<2", "scipy", "noisereduce>=3,<4", "nara-wpe"],
    "clearvoice": ["torch==2.5.1", "torchaudio==2.5.1", "clearvoice"],
    "deepfilter": ["numpy<2", "torch==2.1.2", "torchaudio==2.1.2", "deepfilternet==0.5.6"],
    "resemble": ["numpy<2", "torch==2.1.1", "torchaudio==2.1.1", "resemble-enhance==0.0.1"],
    "silero": ["torch==2.5.1", "torchaudio==2.5.1", "silero-vad"],
    "metrics": ["torch==2.5.1", "torchaudio==2.5.1", "torchmetrics[audio]", "onnxruntime", "librosa", "requests", "gdown"],
    "asr": ["torch==2.5.1", "qwen-asr==0.0.6", "transformers==4.57.6", "accelerate==1.12.0", "huggingface-hub"],
}
if INSTALL_ENVS is None:
    groups_needed = {METHOD_ENV[method] for method in ENABLED}
    if RUN_DNSMOS:
        groups_needed.add("metrics")
    if RUN_ASR:
        groups_needed.add("asr")
    INSTALL_ENVS = sorted(groups_needed)
print("Môi trường sẽ cài:", INSTALL_ENVS)
setup_metadata = {"platform": "kaggle", "dataset_root": str(DATA_ROOT), "package_installer": "pip", "environment_builder": "venv"}
for group in INSTALL_ENVS:
    if group not in {*packages, "sgmse", "sam"}:
        raise ValueError(f"Nhóm môi trường không hợp lệ: {group}")
    py = python_for(group)
    if not py.is_file():
        checked([find_python311(), "-m", "venv", py.parent.parent])
    ensure_pip(py)
    checked([py, "-m", "pip", "install", "numpy<2", "scipy", "soundfile"])
    if group == "sgmse":
        repo = TOOL_ROOT / "sgmse"
        setup_metadata["sgmse_commit"] = checkout("https://github.com/sp-uhh/sgmse.git", repo, SGMSE_REVISION)
        checked([py, "-m", "pip", "install", "torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1", "-r", repo / "requirements.txt"])
    elif group == "sam":
        setup_metadata["sam_revision_requested"] = SAM_REVISION
        checked([py, "-m", "pip", "install", f"git+https://github.com/facebookresearch/sam-audio.git@{SAM_REVISION}"])
    else:
        checked([py, "-m", "pip", "install", *packages[group]])

if BUILD_RNNOISE:
    repo = TOOL_ROOT / "rnnoise"
    setup_metadata["rnnoise_commit"] = checkout("https://github.com/xiph/rnnoise.git", repo, "v0.2")
    checked(["bash", "autogen.sh"], cwd=repo)
    checked(["./configure"], cwd=repo)
    checked(["make", "-j2"], cwd=repo)

if DOWNLOAD_SGMSE_CHECKPOINTS:
    py = python_for("sgmse")
    for method, file_id in {"sgmse_dereverb": "1eiOy0VjHh9V9ZUFTxu1Pq2w19izl9ejD", "sgmse_denoise": "1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ"}.items():
        if method in ENABLED:
            destination = Path(SETTINGS[method]["checkpoint"])
            if not destination.is_file():
                checked([py, "-m", "gdown", file_id, "-O", destination])

(RUN_DIR / "setup.json").write_text(json.dumps(setup_metadata, indent=2), encoding="utf-8")
display(pd.DataFrame([{"group": g, "python": str(python_for(g)), "ready": python_for(g).is_file()}
                      for g in sorted({METHOD_ENV[m] for m in ENABLED})]))

# %% [markdown]
# ## 4. Bộ xử lý dùng chung
#
# Cell này ghi worker vào thư mục run. Worker chỉ được chạy ở mục 5; các import model đều tải trễ.
# Kết quả thống nhất mono 16 kHz PCM16 để nghe và dùng ASR. Nếu model tạo đỉnh vượt biên, worker
# giảm gain vừa đủ để lưu an toàn và ghi rõ `save_gain_db`; không tự tăng âm lượng đầu ra.
# WPE bù chiều dài theo inverse STFT; VAD ghi vị trí cắt. SAM lưu thêm residual riêng.

# %%
WORKER_SOURCE = r'''
import argparse, contextlib, hashlib, importlib.metadata, json, math, os, platform, random, re, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import signal

def read_audio(path):
    sr, x = wavfile.read(str(path))
    if np.issubdtype(x.dtype, np.integer):
        if x.dtype == np.uint8:
            x = (x.astype(np.float64) - 128) / 128
        else:
            x = x.astype(np.float64) / max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Audio must be non-empty finite mono/stereo waveform')
    return x, int(sr)

def resample(x, source_sr, target_sr):
    if source_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(int(source_sr), int(target_sr))
    return signal.resample_poly(x, target_sr // g, source_sr // g).astype(np.float32)

def save_audio(path, x, sr):
    x = np.asarray(x).squeeze()
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Model produced empty, multichannel or non-finite output')
    peak = float(np.max(np.abs(x)))
    gain = min(1.0, .999 / max(peak, 1e-12))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, np.round(x * gain * 32767).astype(np.int16))
    return 20 * math.log10(max(gain, 1e-12))

def peaking_eq(x, sr, frequency, gain_db, q=0.8):
    a = 10 ** (gain_db / 40); w = 2 * np.pi * frequency / sr
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha*a, -2*np.cos(w), 1-alpha*a])
    den = np.array([1 + alpha/a, -2*np.cos(w), 1-alpha/a])
    return signal.lfilter(b / den[0], den / den[0], x)

def scrub(message):
    return re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED_TOKEN]', str(message))

def synchronize():
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()

def make_processor(method, cfg, device, work_dir):
    cpu_method = method in {'original', 'eq_mild', 'notch_50hz', 'rms_control', 'wpe', 'rnnoise', 'vad_trim'} or method.startswith('spectral_')
    info = {'method': method, 'device': 'cpu' if cpu_method else device}
    if cpu_method and method not in {'rnnoise', 'vad_trim'}:
        info['parameter_count'] = 0
    if method == 'original':
        return lambda x, sr, p, out: (x, sr, {}), info
    if method == 'eq_mild':
        def process(x, sr, p, out):
            y = signal.sosfilt(signal.butter(2, cfg['highpass_hz'], 'highpass', fs=sr, output='sos'), x)
            y = peaking_eq(y, sr, cfg['mud_hz'], cfg['mud_gain_db'])
            return peaking_eq(y, sr, cfg['clarity_hz'], cfg['clarity_gain_db']), sr, {}
        return process, info
    if method == 'notch_50hz':
        def process(x, sr, p, out):
            b, a = signal.iirnotch(cfg['frequency_hz'], cfg['q'], fs=sr)
            return signal.lfilter(b, a, x), sr, {}
        return process, info
    if method == 'rms_control':
        def process(x, sr, p, out):
            rms = np.sqrt(np.mean(x*x))
            gain_db = min(cfg['max_gain_db'], cfg['target_dbfs'] - 20*np.log10(max(rms, 1e-12)))
            return x * 10**(gain_db/20), sr, {'requested_gain_db': float(gain_db)}
        return process, info
    if method.startswith('spectral_'):
        import noisereduce as nr
        return lambda x, sr, p, out: (nr.reduce_noise(y=x, sr=sr, stationary=False, prop_decrease=cfg['prop_decrease'], n_fft=512, hop_length=128, n_jobs=1), sr, {}), info
    if method == 'wpe':
        from nara_wpe.utils import stft, istft
        from nara_wpe.wpe import wpe
        def process(x, sr, p, out):
            spec = stft(x[None, :], size=cfg['fft_size'], shift=cfg['hop']).transpose(2, 0, 1)
            enhanced = wpe(spec, taps=cfg['taps'], delay=cfg['delay'], iterations=cfg['iterations'], statistics_mode='full')
            y = istft(enhanced.transpose(1, 2, 0), size=cfg['fft_size'], shift=cfg['hop']).squeeze()
            if len(y) < len(x):
                raise ValueError('WPE inverse STFT shorter than input; check padding settings')
            return y[:len(x)], sr, {}
        return process, info
    if method == 'rnnoise':
        binary = Path(cfg['binary']).expanduser().resolve()
        if not binary.is_file():
            raise FileNotFoundError('Build RNNoise or configure its rnnoise_demo binary')
        def process(x, sr, p, out):
            y = resample(x, sr, 48000)
            length = len(y)
            # Demo discards its first output frame: append an extra frame to flush it.
            padded_length = (math.ceil(length/480) + 1) * 480
            y = np.pad(y, (0, padded_length-length))
            raw_in, raw_out = out.with_suffix('.in.raw'), out.with_suffix('.out.raw')
            np.round(np.clip(y, -.999, .999)*32767).astype(np.int16).tofile(raw_in)
            subprocess.run([str(binary), str(raw_in), str(raw_out)], cwd=binary.parent.parent, check=True, capture_output=True)
            result = np.fromfile(raw_out, dtype=np.int16).astype(np.float32)/32768
            if len(result) < length:
                raise ValueError('RNNoise returned too few samples')
            raw_in.unlink(missing_ok=True); raw_out.unlink(missing_ok=True)
            return result[:length], 48000, {}
        return process, info

    import torch
    torch.set_num_threads(2)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this worker environment')
    if method == 'vad_trim':
        from silero_vad import load_silero_vad, get_speech_timestamps
        model = load_silero_vad()
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        def process(x, sr, p, out):
            y = resample(x, sr, 16000)
            spans = get_speech_timestamps(torch.from_numpy(y), model, sampling_rate=16000, threshold=cfg['threshold'], speech_pad_ms=cfg['speech_pad_ms'])
            if not spans:
                raise ValueError('VAD found no speech; review this clip instead of emitting silence')
            a, b = int(spans[0]['start']), int(spans[-1]['end'])
            return y[a:b], 16000, {'trim_start_seconds': a/16000, 'trim_end_seconds': (len(y)-b)/16000}
        return process, info
    if method == 'deepfilternet':
        from df.enhance import init_df, enhance
        model, state, _ = init_df(model_base_dir=cfg['model'], log_file=None)
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        def process(x, sr, p, out):
            y = torch.from_numpy(resample(x, sr, state.sr()))[None, :]
            z = enhance(model, state, y, pad=True, atten_lim_db=cfg.get('atten_lim_db'))
            return z.squeeze().detach().cpu().numpy(), state.sr(), {}
        return process, info
    if method in {'frcrn', 'mossformergan'}:
        from clearvoice import ClearVoice
        model = ClearVoice(task='speech_enhancement', model_names=[cfg['model']])
        backend = getattr(model.models[0], 'model', None)
        if isinstance(backend, torch.nn.Module):
            info['parameter_count'] = sum(p.numel() for p in backend.parameters())
        def process(x, sr, p, out):
            result = model(input_path=str(p), online_write=False)
            native_path = out.with_name(out.stem + '.native.wav')
            native_path.unlink(missing_ok=True)
            model.write(result, output_path=str(native_path))
            y, native_sr = read_audio(native_path)
            return y, native_sr, {}
        return process, info
    if method.startswith('resemble_'):
        from resemble_enhance.enhancer.inference import load_enhancer, denoise, enhance
        model = load_enhancer(None, device)
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        def process(x, sr, p, out):
            waveform = torch.from_numpy(x.astype(np.float32))
            if method == 'resemble_denoise':
                y, native_sr = denoise(waveform, sr, device)
            else:
                y, native_sr = enhance(waveform, sr, device, **cfg)
            return y.detach().cpu().numpy(), native_sr, {}
        return process, info
    if method.startswith('sgmse_'):
        repo, checkpoint = Path(cfg['repo']), Path(cfg['checkpoint'])
        if not (repo / 'sgmse').is_dir() or not checkpoint.is_file():
            raise FileNotFoundError('Prepare official SGMSE source and the task-specific checkpoint in section 3')
        sys.path.insert(0, str(repo))
        from sgmse.model import ScoreModel
        from sgmse.util.other import pad_spec
        model = ScoreModel.load_from_checkpoint(str(checkpoint), map_location=device).to(device).eval()
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        info['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        info['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        def process(x, sr, p, out):
            y = torch.from_numpy(resample(x, sr, 16000))[None, :].to(device)
            scale = y.abs().max().clamp_min(1e-8)
            with torch.inference_mode():
                spec = pad_spec(model._forward_transform(model._stft(y / scale)).unsqueeze(0))
                sampler = model.get_pc_sampler('reverse_diffusion', 'ald', spec, N=cfg['steps'], corrector_steps=1, snr=cfg['snr'])
                sample, _ = sampler()
                result = model.to_audio(sample.squeeze(), y.shape[-1]) * scale
            return result.squeeze().detach().cpu().numpy(), 16000, {}
        return process, info
    if method == 'sam_audio':
        from sam_audio import SAMAudio, SAMAudioProcessor
        model = SAMAudio.from_pretrained(cfg['model']).to(device).eval()
        processor = SAMAudioProcessor.from_pretrained(cfg['model'])
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        if cfg['bf16']:
            if not device.startswith('cuda') or not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 requested but unsupported; use bf16=False')
            model = model.to(torch.bfloat16)
        def process(x, sr, p, out):
            batch = processor(audios=[str(p)], descriptions=[cfg['prompt']]).to(device)
            cast = torch.autocast('cuda', dtype=torch.bfloat16) if cfg['bf16'] else contextlib.nullcontext()
            with torch.inference_mode(), cast:
                result = model.separate(batch, predict_spans=cfg['predict_spans'], reranking_candidates=cfg['reranking_candidates'])
            native_sr = int(processor.audio_sampling_rate)
            y = result.target[0].detach().float().cpu().numpy().squeeze()
            residual_path = out.with_name(out.stem + '.residual.wav')
            residual_gain = save_audio(residual_path, resample(result.residual[0].detach().float().cpu().numpy().squeeze(), native_sr, 16000), 16000)
            return y, native_sr, {'residual_path': str(residual_path), 'residual_save_gain_db': residual_gain}
        return process, info
    raise ValueError('Unknown method: ' + method)

def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    outdir = Path(job['output_dir']); outdir.mkdir(parents=True, exist_ok=True)
    records_path = outdir / 'results.jsonl'
    random.seed(job['seed']); np.random.seed(job['seed'])
    start = time.perf_counter()
    try:
        process, info = make_processor(job['method'], job['settings'], job['device'], outdir)
        synchronize()
        info['initialization_seconds'] = time.perf_counter() - start
        info['warmup_seconds'] = 0.0
        if job['warmup'] and job['method'] not in {'original', 'eq_mild', 'notch_50hz', 'rms_control', 'wpe'}:
            first = job['samples'][0]
            x, sr = read_audio(first['audio_path']); x = resample(x, sr, 16000)
            t = time.perf_counter()
            process(x, 16000, Path(first['audio_path']), outdir / '_warmup.wav')
            synchronize(); info['warmup_seconds'] = time.perf_counter() - t
        info['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
        info['python'] = sys.version; info['platform'] = platform.platform()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
        (outdir / 'method_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    except Exception as e:
        failed = [{**s, 'method': job['method'], 'status': 'setup_error', 'error': scrub(e)} for s in job['samples']]
        records_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in failed), encoding='utf-8')
        print(scrub(traceback.format_exc()), file=sys.stderr)
        return
    with records_path.open('w', encoding='utf-8') as stream:
        for sample in job['samples']:
            record = {**sample, 'method': job['method'], 'effective_device': info['device']}
            try:
                seed = job['seed'] + int(hashlib.sha256(sample['sample_id'].encode()).hexdigest()[:7], 16)
                np.random.seed(seed); random.seed(seed)
                if torch is not None:
                    torch.manual_seed(seed)
                output = outdir / (sample['sample_id'] + '.wav')
                synchronize(); t = time.perf_counter()
                x, sr = read_audio(sample['audio_path'])
                input_seconds = len(x)/sr
                x = resample(x, sr, 16000)
                y, native_sr, extra = process(x, 16000, Path(sample['audio_path']), output)
                y = resample(y, int(native_sr), 16000)
                gain_db = save_audio(output, y, 16000)
                synchronize(); elapsed = time.perf_counter()-t
                record.update(status='ok', output_path=str(output), input_seconds=input_seconds,
                              output_seconds=len(y)/16000, processing_seconds=elapsed, rtf=elapsed/input_seconds,
                              native_sample_rate=int(native_sr), save_gain_db=gain_db, **extra)
            except Exception as e:
                record.update(status='processing_error', error=scrub(e))
                print(scrub(traceback.format_exc()), file=sys.stderr)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
    if torch is not None and torch.cuda.is_available():
        info['peak_cuda_allocated_mb'] = torch.cuda.max_memory_allocated()/1024**2
        (outdir / 'method_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
'''

WORKER_PATH = RUN_DIR / "audio_worker.py"
WORKER_PATH.write_text(WORKER_SOURCE, encoding="utf-8")
compile(WORKER_SOURCE, str(WORKER_PATH), "exec")
print("Worker:", WORKER_PATH)

# %% [markdown]
# ## 5. Chạy từng phương pháp và ghi thời gian
#
# Worker chạy tuần tự, nạp một model tại một thời điểm. `processing_seconds` gồm resample,
# inference, đọc/ghi WAV; không gồm nạp model, tải checkpoint hoặc warm-up.
# RTF = thời gian xử lý / thời lượng audio đầu vào. Lần chạy đầu có thể tải weights trong bước nạp model.
# Các phương pháp lỗi/môi trường thiếu có dòng trạng thái riêng, không được tính như kết quả thành công.
# Đọc log theo đường dẫn được in ra nếu cần sửa dependency hoặc quyền truy cập model.

# %%
def subprocess_environment():
    env = os.environ.copy()
    if HF_ACCESS_TOKEN:
        env["HF_TOKEN"] = HF_ACCESS_TOKEN
        env.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
    else:
        env.pop("HF_TOKEN", None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["TORCH_HOME"] = str(MODEL_ROOT / "torch")
    env["CUDA_VISIBLE_DEVICES"] = GPU_INDEX if DEVICE.startswith("cuda") else ""
    env["HF_HOME"] = str(MODEL_ROOT / "huggingface")
    env["XDG_CACHE_HOME"] = str(MODEL_ROOT / "cache")
    env["WANDB_MODE"] = "disabled"
    return env

def redact(text):
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_TOKEN]", str(text))

run_config = {"seed": SEED, "methods": ENABLED, "settings": {m: SETTINGS.get(m, {}) for m in ENABLED},
              "device": DEVICE, "gpu_index": GPU_INDEX, "warmup": WARMUP,
              "platform": "kaggle", "dataset_root": str(DATA_ROOT), "worker_sha256": hashlib.sha256(WORKER_SOURCE.encode()).hexdigest()}
(RUN_DIR / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
RESULTS = []
for method in ENABLED:
    output_dir = RUN_DIR / "audio" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    py = python_for(METHOD_ENV[method])
    job = {"method": method, "settings": SETTINGS.get(method, {}), "samples": SAMPLES,
           "seed": SEED, "device": DEVICE, "warmup": WARMUP, "output_dir": str(output_dir)}
    job_path = output_dir / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path = output_dir / "results.jsonl"
    result_path.unlink(missing_ok=True)  # Không đọc nhầm kết quả cũ nếu worker lần này không chạy được.
    (output_dir / "method_info.json").unlink(missing_ok=True)
    (output_dir / "worker.log").unlink(missing_ok=True)
    print(f"[{method}] {len(SAMPLES)} clips …", flush=True)
    failure = None
    if not py.is_file():
        failure = f"Chưa có môi trường {METHOD_ENV[method]}; cài ở mục 3 hoặc điền PYTHON_OVERRIDES."
    else:
        try:
            completed = subprocess.run([str(py), str(WORKER_PATH), str(job_path)], env=subprocess_environment(),
                                       cwd=TOOL_ROOT, capture_output=True, text=True, timeout=PROCESS_TIMEOUT_SECONDS)
            (output_dir / "worker.log").write_text(redact(completed.stdout + "\n" + completed.stderr), encoding="utf-8")
            if completed.returncode:
                failure = f"Worker exit={completed.returncode}; xem {output_dir / 'worker.log'}"
        except subprocess.TimeoutExpired:
            failure = "Worker vượt thời gian chờ; các mẫu chưa hoàn tất được ghi worker_error."
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()] if result_path.is_file() else []
    finished_ids = {r["sample_id"] for r in records}
    records.extend({**s, "method": method, "status": "worker_error", "error": failure or "Worker không ghi kết quả cho mẫu này"}
                   for s in SAMPLES if s["sample_id"] not in finished_ids)
    RESULTS.extend(records)
    good = sum(r["status"] == "ok" for r in records)
    print(f"  {good}/{len(SAMPLES)} thành công; logs: {output_dir}")
    if good == 0:
        print(" ", redact(records[0].get("error", "Không có kết quả")))

results_df = pd.DataFrame(RESULTS)
results_df.to_csv(RUN_DIR / "processing_results.csv", index=False)
display(results_df.groupby(["method", "status"], sort=False).size().rename("samples").reset_index())

# %% [markdown]
# ## 6. Chỉ số tín hiệu, thời gian và biểu đồ
#
# RMS/peak, tỷ lệ gần clipping và chiều dài chỉ giúp phát hiện đầu ra bất thường.
# Tỷ lệ frame dưới −50 dBFS là mức năng lượng thấp, **không phải** tỷ lệ im lặng hay chất lượng giọng.
# Không dùng các chỉ số này để tuyên bố giảm vang thành công.
# Bảng so sánh tính trên **giao các sample thành công của toàn bộ phương pháp có đầu ra**;
# bảng coverage vẫn hiện số mẫu thất bại để tránh chọn model chỉ xử lý được các mẫu dễ.

# %%
def read_wav(path):
    sr, x = wavfile.read(str(path))
    if np.issubdtype(x.dtype, np.integer):
        x = ((x.astype(float)-128)/128 if x.dtype == np.uint8 else
             x.astype(float)/max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max))
    else:
        x = x.astype(float)
    return (x.mean(axis=1) if x.ndim == 2 else x), int(sr)

signal_rows = []
ok = results_df[results_df["status"] == "ok"].copy()
for r in ok.to_dict("records"):
    x, sr = read_wav(r["output_path"])
    frame = max(1, round(sr*.02))
    framed = x[:len(x)//frame*frame].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(framed**2, axis=1)) if len(framed) else np.array([0.0])
    signal_rows.append({"method": r["method"], "sample_id": r["sample_id"],
                        "rms_dbfs": 20*np.log10(max(np.sqrt(np.mean(x*x)), 1e-12)),
                        "peak_dbfs": 20*np.log10(max(np.max(np.abs(x)), 1e-12)),
                        "near_clipping_pct": 100*np.mean(np.abs(x) >= .999),
                        "low_energy_frame_pct": 100*np.mean(frame_rms < 10**(-50/20)),
                        "duration_delta_seconds": len(x)/sr-r["input_seconds"]})
signal_df = pd.DataFrame(signal_rows, columns=["method", "sample_id", "rms_dbfs", "peak_dbfs", "near_clipping_pct", "low_energy_frame_pct", "duration_delta_seconds"])
signal_df.to_csv(RUN_DIR / "signal_metrics.csv", index=False)
ok = ok.merge(signal_df, on=["method", "sample_id"], how="left")

successful_sets = [set(g["sample_id"]) for _, g in ok.groupby("method")]
COMMON_IDS = set.intersection(*successful_sets) if successful_sets else set()
paired_ok = ok[ok["sample_id"].isin(COMMON_IDS)]
coverage_df = results_df.assign(success=results_df["status"].eq("ok")).groupby("method", sort=False).agg(
    attempted=("sample_id", "size"), successful=("success", "sum"))
summary_df = paired_ok.groupby("method", sort=False).agg(
    paired_samples=("sample_id", "size"), median_seconds=("processing_seconds", "median"),
    p95_seconds=("processing_seconds", lambda x: x.quantile(.95)),
    total_processing_seconds=("processing_seconds", "sum"), total_audio_seconds=("input_seconds", "sum"),
    mean_rms_dbfs=("rms_dbfs", "mean"), mean_duration_delta=("duration_delta_seconds", "mean")) if len(paired_ok) else pd.DataFrame()
if len(summary_df):
    summary_df["rtf_weighted"] = summary_df["total_processing_seconds"] / summary_df["total_audio_seconds"]
summary_df = coverage_df.join(summary_df)
model_infos = []
for method in ENABLED:
    info_path = RUN_DIR / "audio" / method / "method_info.json"
    if info_path.is_file():
        metadata = json.loads(info_path.read_text(encoding="utf-8"))
        model_infos.append({"method": method, **{k: metadata.get(k) for k in ["device", "parameter_count", "initialization_seconds", "warmup_seconds", "peak_cuda_allocated_mb"]}})
if model_infos:
    summary_df = summary_df.join(pd.DataFrame(model_infos).set_index("method"))
summary_df.to_csv(RUN_DIR / "comparison.csv")
display(summary_df.round(4))
if len(paired_ok):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    paired_ok.boxplot(column="rtf", by="method", ax=axes[0], rot=60)
    paired_ok.boxplot(column="rms_dbfs", by="method", ax=axes[1], rot=60)
    axes[0].set_ylabel("RTF, thấp hơn = xử lý nhanh hơn")
    axes[1].set_ylabel("dBFS; đây không phải điểm chất lượng")
    fig.suptitle(""); fig.tight_layout()
    fig.savefig(RUN_DIR / "runtime-and-levels.png", dpi=160)
    plt.show()

# %% [markdown]
# ## 7. Nghe A/B và xem waveform/spectrogram
#
# `PLAYBACK_MODE="rms_matched"` chỉ đổi âm lượng **khi phát**, không sửa file dùng chấm điểm.
# Các biến thể được cân bằng RMS theo đoạn có năng lượng của bản gốc, rồi dùng chung một hệ số
# headroom để không clipping. Có thể dùng `raw` để nghe mức output thật.
# `normalize=False` ngăn widget tự chuẩn hóa peak riêng từng file.
# Thử `BLIND=True` để tạm ẩn tên phương pháp; bảng giải mã lưu ở cuối mục này.

# %%
LISTEN_SAMPLE_ID = SAMPLES[0]["sample_id"]
PLAYBACK_MODE = "rms_matched"  # "raw" hoặc "rms_matched"
BLIND = False
PLOT_MAX_METHODS = 6
listen_rows = ok[ok["sample_id"] == LISTEN_SAMPLE_ID].to_dict("records")
if not listen_rows:
    print("Mẫu này chưa có đầu ra; kiểm tra mục 5.")
else:
    display(Markdown("**Transcript:** " + html.escape(next(s["reference"] for s in SAMPLES if s["sample_id"] == LISTEN_SAMPLE_ID))))
    if BLIND:
        random.Random(SEED).shuffle(listen_rows)
    waveforms = []
    def active_rms(x, sr):
        frame = max(1, round(.02*sr)); n = len(x)//frame
        f = x[:n*frame].reshape(n, frame) if n else x[None, :]
        energy = np.sqrt(np.mean(f*f, axis=1))
        mask = energy >= max(float(np.max(energy))*.1, 1e-8)
        return float(np.sqrt(np.mean(f[mask]**2))) if mask.any() else 0.0
    original_row = next((r for r in listen_rows if r["method"] == "original"), listen_rows[0])
    original_x, original_sr = read_wav(original_row["output_path"])
    target_level = active_rms(original_x, original_sr)
    for r in listen_rows:
        x, sr = read_wav(r["output_path"])
        gain = target_level / max(active_rms(x, sr), 1e-12) if PLAYBACK_MODE == "rms_matched" else 1.0
        waveforms.append((r, x*min(gain, 100), sr))
    largest_peak = max(np.max(np.abs(x)) for _, x, _ in waveforms)
    common_gain = min(1.0, .98/max(largest_peak, 1e-12))
    decoding = []
    for index, (r, x, sr) in enumerate(waveforms):
        label = f"Variant {index+1}" if BLIND else r["method"]
        decoding.append({"label": label, "method": r["method"], "sample_id": LISTEN_SAMPLE_ID})
        display(Markdown(f"**{label}**"))
        display(Audio(x*common_gain, rate=sr, normalize=False))
        if isinstance(r.get("residual_path"), str) and Path(r["residual_path"]).is_file():
            display(Markdown("Residual do model tách ra — kiểm tra có mất âm tiết chính hay không:"))
            residual, residual_sr = read_wav(r["residual_path"])
            residual /= max(1.0, float(np.max(np.abs(residual)))/.98)
            display(Audio(residual, rate=residual_sr, normalize=False))
    pd.DataFrame(decoding).to_csv(RUN_DIR / f"listening-key-{LISTEN_SAMPLE_ID}.csv", index=False)
    # Cùng thang dB và màu để không làm phương pháp nào trông sạch hơn do tự đổi color scale.
    draw_rows = waveforms[:PLOT_MAX_METHODS]
    fig, axes = plt.subplots(len(draw_rows), 2, figsize=(13, 2.6*len(draw_rows)), squeeze=False)
    for index, (r, x, sr) in enumerate(draw_rows):
        label = f"Variant {index+1}" if BLIND else r["method"]
        axes[index, 0].plot(np.arange(len(x))/sr, x*common_gain, linewidth=.5)
        axes[index, 0].set(title=label, xlabel="Giây", ylim=(-1, 1))
        f, t, power = signal.spectrogram(x*common_gain, fs=sr, nperseg=512, noverlap=384, scaling="spectrum")
        axes[index, 1].pcolormesh(t, f, 10*np.log10(np.maximum(power, 1e-12)), shading="auto", vmin=-100, vmax=-20, cmap="magma")
        axes[index, 1].set(xlabel="Giây", ylabel="Hz", ylim=(0, 8000))
    fig.tight_layout(); fig.savefig(RUN_DIR / f"waveforms-{LISTEN_SAMPLE_ID}.png", dpi=140); plt.show()

# %% [markdown]
# ### Nghe các audio theo từng sample
#
# Mỗi nhóm bên dưới là một `sample_id`, kèm transcript, bản gốc và các phương pháp đã chọn.
# Các player phát file ở mức âm lượng đã lưu. Đổi `DISPLAY_SAMPLE_IDS` để chỉ xem một vài mẫu,
# hoặc để `[]` để hiện toàn bộ mẫu của khảo sát. `DISPLAY_METHODS=[]` hiện các mục trong `ENABLED`.
# Phương pháp chưa chạy, lỗi hoặc thiếu file được ghi rõ trong cùng nhóm. Bấm tên sample để thu gọn/mở rộng.

# %%
# Hiện audio theo từng sample: original và các bản xử lý nằm trong cùng nhóm.
DISPLAY_SAMPLE_IDS = []  # Ví dụ ["train_0001", "train_0005"]; [] = mọi sample đã chọn
DISPLAY_METHODS = []     # Ví dụ ["original", "wpe", "mossformergan"]; [] = ENABLED

sample_lookup = {sample["sample_id"]: sample for sample in SAMPLES}
sample_ids_to_show = DISPLAY_SAMPLE_IDS or list(sample_lookup)
methods_to_show = list(dict.fromkeys(["original", *(DISPLAY_METHODS or ENABLED)]))
unknown_samples = set(sample_ids_to_show) - set(sample_lookup)
if unknown_samples:
    raise ValueError(f"Sample không thuộc khảo sát: {sorted(unknown_samples)}")

# Embed WAV giúp player hoạt động khi mở notebook ở thư mục khác hoặc trên Kaggle.
def audio_player_from_file(file_path):
    player = Audio(filename=str(file_path), embed=True)._repr_html_()
    return player.replace("<audio ", '<audio preload="none" ', 1)

result_lookup = {(str(row["sample_id"]), str(row["method"])): row
                 for row in results_df.to_dict("records")}
audio_groups = []
for sample_id in sample_ids_to_show:
    sample = sample_lookup[sample_id]
    cards = []
    for method in methods_to_show:
        record = result_lookup.get((sample_id, method), {})
        output_file = record.get("output_path")
        # Original vẫn có thể nghe từ dataset nếu lượt xử lý baseline chưa thành công.
        if method == "original" and (record.get("status") != "ok" or not isinstance(output_file, str)):
            output_file = sample["audio_path"]
        available = isinstance(output_file, str) and Path(output_file).is_file()
        if available and (method == "original" or record.get("status") == "ok"):
            content = audio_player_from_file(output_file)
            content += f'<div style="font-size:12px;overflow-wrap:anywhere">{html.escape(str(output_file))}</div>'
        else:
            status = record.get("status", "chưa chạy")
            reason = record.get("error")
            message = str(reason) if isinstance(reason, str) and reason else "Chưa có file audio đầu ra."
            content = f'<p><strong>{html.escape(str(status))}</strong>: {html.escape(message)}</p>'
        cards.append(
            '<div style="flex:1 1 320px;min-width:280px;max-width:100%;border:1px solid #aaa;'
            'border-radius:6px;padding:10px;box-sizing:border-box">'
            f'<div style="font-weight:600;margin-bottom:6px">{html.escape(method)}</div>{content}</div>'
        )
    audio_groups.append(
        '<details open style="margin:14px 0;padding:12px;border:1px solid #aaa;border-radius:8px">'
        f'<summary style="cursor:pointer;font-weight:700">{html.escape(sample_id)}</summary>'
        f'<p>{html.escape(str(sample["reference"]))}</p>'
        '<div style="display:flex;flex-wrap:wrap;gap:10px">' + "".join(cards) + '</div></details>'
    )
if audio_groups:
    display(HTML("".join(audio_groups)))
else:
    print("Chưa chọn sample nào để hiển thị.")

# %% [markdown]
# ### Phiếu nghe thủ công
#
# Điền CSV dưới đây sau khi nghe: `clarity`, `naturalness`, `reverb_suppression`, `noise_suppression`
# theo thang 1–5 (cao hơn = tốt hơn); `lost_or_changed_speech` là 0/1; `notes` ghi phụ âm,
# thanh điệu, tên riêng hay con số bị thay đổi. Dùng cùng người nghe, tai nghe và âm lượng.
# Không tự điền điểm để tránh nhầm giá trị mặc định với đánh giá thật.

# %%
rating_path = RUN_DIR / "listening_ratings.csv"
rating_columns = ["clarity", "naturalness", "reverb_suppression", "noise_suppression", "lost_or_changed_speech", "notes"]
ratings = ok[["sample_id", "method"]].copy()
if rating_path.exists():
    previous_ratings = pd.read_csv(rating_path).drop_duplicates(["sample_id", "method"], keep="last")
    ratings = ratings.merge(previous_ratings, on=["sample_id", "method"], how="left")
for column in rating_columns:
    if column not in ratings:
        ratings[column] = ""
ratings.to_csv(rating_path, index=False)
print("Điền điểm tại:", rating_path)

# %% [markdown]
# ## 8. DNSMOS tùy chọn: điểm giọng, nền và tổng thể
#
# Đặt `RUN_DNSMOS=True` ở mục 2; mục 3 tự cài môi trường `metrics`, rồi chạy cell này.
# DNSMOS dự đoán SIG/BAK/OVRL; không cần clean reference. Đây là model dự đoán cảm nhận,
# không phải phép đo chính xác độ vang hay độ đúng âm tiết. Không chỉ tối ưu BAK vì có thể
# được nền yên nhưng giọng kém. Model DNSMOS có thể tải thêm weights lần đầu.

# %%
METRICS_SOURCE = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
job = json.loads(Path(sys.argv[1]).read_text())
with Path(job['output']).open('w') as stream:
    for row in job['rows']:
        record = {k: row[k] for k in ('sample_id', 'method')}
        try:
            sr, x = wavfile.read(row['output_path'])
            x = x.astype(np.float32)/32768
            scores = deep_noise_suppression_mean_opinion_score(torch.from_numpy(x), sr, personalized=False, device=torch.device('cpu')).detach().cpu().numpy().reshape(-1)
            record.update(zip(['dnsmos_p808', 'dnsmos_sig', 'dnsmos_bak', 'dnsmos_ovrl'], map(float, scores)))
            record['status'] = 'ok'
        except Exception as e:
            record.update(status='error', error=str(e))
        stream.write(json.dumps(record)+'\n'); stream.flush()
'''

# RUN_DNSMOS được cấu hình ở mục 2.
dnsmos_df = pd.DataFrame()
if RUN_DNSMOS:
    metric_script = RUN_DIR / "dnsmos_worker.py"
    metric_script.write_text(METRICS_SOURCE, encoding="utf-8")
    metric_job = RUN_DIR / "dnsmos_job.json"
    metric_job.write_text(json.dumps({"rows": ok[["sample_id", "method", "output_path"]].to_dict("records"), "output": str(RUN_DIR / "dnsmos.jsonl")}), encoding="utf-8")
    py = python_for("metrics")
    if not py.is_file():
        raise FileNotFoundError("Cài môi trường metrics ở mục 3.")
    completed = subprocess.run([str(py), str(metric_script), str(metric_job)], env=subprocess_environment(), capture_output=True, text=True, timeout=PROCESS_TIMEOUT_SECONDS)
    (RUN_DIR / "dnsmos.log").write_text(redact(completed.stdout + "\n" + completed.stderr), encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("DNSMOS worker lỗi; xem dnsmos.log")
    dnsmos_df = pd.DataFrame([json.loads(line) for line in (RUN_DIR / "dnsmos.jsonl").read_text().splitlines()])
    dnsmos_df.to_csv(RUN_DIR / "dnsmos.csv", index=False)
    display(dnsmos_df)
else:
    print("DNSMOS chưa bật.")

# %% [markdown]
# ## 9. Chạy lại ASR và đo WER/CER tùy chọn
#
# Đặt `RUN_ASR=True` ở mục 2; mục 3 tự cài môi trường `asr`. Chọn **một model cố định** cho mọi phương pháp.
# Có thể dùng đường dẫn snapshot local hoặc model ID Hugging Face. Worker nạp Qwen3-ASR 1.7B một lần,
# chạy cả original và mọi bản xử lý, không dùng lại transcript của lần benchmark audio gốc.
#
# Mặc định dùng `Qwen/Qwen3-ASR-1.7B` qua `qwen-asr==0.0.6`, backend Transformers,
# ngôn ngữ `Vietnamese`, batch size 1, beam=1, không sampling, `max_new_tokens=448` như cấu hình
# Qwen 1.7B trong benchmark của repo. GPU dùng FP16, CPU dùng FP32. Snapshot được cố định theo
# revision trước khi nạp cả model và processor. Khi khảo sát clip dài, tăng `ASR_MAX_NEW_TOKENS`
# để tránh giới hạn đầu ra quá thấp. So sánh run cũ cần cùng checkpoint và decoding.
#
# API chính thức: https://github.com/QwenLM/Qwen3-ASR
#
# Text normalization: Unicode NFC, lowercase, dấu câu thành khoảng trắng, gộp khoảng trắng;
# giữ dấu tiếng Việt và cách viết số như bộ benchmark hiện có. WER dùng token tách bằng khoảng trắng,
# CER có tính khoảng trắng. Bảng tính corpus WER/CER bằng tổng edit / tổng reference units.

# %%
ASR_SOURCE = r'''
import json, os, sys, time, unicodedata
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel

def normalize(text):
    text = unicodedata.normalize('NFC', text).lower()
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.split())

def distance(a, b):
    old = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1]+1, old[j]+1, old[j-1]+(x != y)))
        old = new
    return old[-1]

job = json.loads(Path(sys.argv[1]).read_text())
device = job['device']
if device.startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('ASR CUDA unavailable')
dtype = torch.float16 if device.startswith('cuda') else torch.float32
start = time.perf_counter()
local_model = Path(job['model']).expanduser()
if local_model.is_dir():
    model_path = str(local_model)
    resolved_revision = None
else:
    model_path = snapshot_download(repo_id=job['model'], revision=job.get('revision'))
    resolved_revision = Path(model_path).name
max_new_tokens = int(job.get('max_new_tokens', 448))
model = Qwen3ASRModel.from_pretrained(
    model_path, dtype=dtype, device_map=device, max_inference_batch_size=1,
    max_new_tokens=max_new_tokens, local_files_only=True,
)
model.model.generation_config.do_sample = False
model.model.generation_config.num_beams = 1
model.model.eval()
if device.startswith('cuda'):
    torch.cuda.synchronize()
metadata = {'model': job['model'], 'model_path': model_path, 'requested_revision': job.get('revision'),
            'resolved_revision': resolved_revision or getattr(model.model.config, '_commit_hash', None),
            'backend': 'qwen-asr/transformers', 'device': device, 'dtype': str(dtype),
            'load_seconds': time.perf_counter()-start, 'max_inference_batch_size': 1,
            'num_beams': 1, 'max_new_tokens': max_new_tokens, 'do_sample': False, 'language': 'Vietnamese'}
Path(job['metadata']).write_text(json.dumps(metadata, indent=2))
with Path(job['output']).open('w', encoding='utf-8') as stream:
    for row in job['rows']:
        record = {k: row[k] for k in ('sample_id', 'method', 'reference')}
        try:
            sr, x = wavfile.read(row['output_path'])
            if sr != 16000 or x.ndim != 1 or x.size == 0:
                raise ValueError('Requires non-empty mono 16 kHz audio')
            with torch.inference_mode():
                prediction = model.transcribe(audio=(x.astype(np.float32)/32768, sr), language='Vietnamese')[0]
            hypothesis = prediction.text.strip()
            reference, predicted = normalize(row['reference']), normalize(hypothesis)
            words, chars = reference.split(), list(reference)
            word_edits, char_edits = distance(words, predicted.split()), distance(chars, list(predicted))
            record.update(status='ok', prediction=hypothesis, reference_normalized=reference, prediction_normalized=predicted,
                          word_edits=word_edits, reference_words=len(words), char_edits=char_edits, reference_chars=len(chars),
                          wer=word_edits/len(words) if words else None, cer=char_edits/len(chars) if chars else None)
        except Exception as e:
            record.update(status='error', error=str(e))
        stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
'''

# RUN_ASR, ASR_MODEL và ASR_REVISION được cấu hình ở mục 2.
asr_df = pd.DataFrame()
if RUN_ASR:
    asr_script = RUN_DIR / "asr_worker.py"
    asr_script.write_text(ASR_SOURCE, encoding="utf-8")
    asr_job = RUN_DIR / "asr_job.json"
    asr_job.write_text(json.dumps({"model": str(ASR_MODEL), "revision": ASR_REVISION, "device": DEVICE, "max_new_tokens": ASR_MAX_NEW_TOKENS,
                                   "rows": ok[["sample_id", "method", "output_path", "reference"]].to_dict("records"),
                                   "output": str(RUN_DIR / "asr_predictions.jsonl"), "metadata": str(RUN_DIR / "asr_metadata.json")}, ensure_ascii=False), encoding="utf-8")
    py = python_for("asr")
    if not py.is_file():
        raise FileNotFoundError("Cài môi trường asr ở mục 3.")
    completed = subprocess.run([str(py), str(asr_script), str(asr_job)], env=subprocess_environment(), capture_output=True, text=True, timeout=PROCESS_TIMEOUT_SECONDS)
    (RUN_DIR / "asr.log").write_text(redact(completed.stdout + "\n" + completed.stderr), encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("ASR worker lỗi; xem asr.log")
    asr_df = pd.DataFrame([json.loads(line) for line in (RUN_DIR / "asr_predictions.jsonl").read_text(encoding="utf-8").splitlines()])
    asr_df.to_csv(RUN_DIR / "asr_predictions.csv", index=False)
    display(asr_df[["sample_id", "method", "status", *[c for c in ["prediction", "wer", "cer"] if c in asr_df]]])
else:
    print("ASR chưa bật; chưa có bằng chứng phương pháp nào cải thiện WER/CER.")

# %% [markdown]
# ## 10. Gộp kết quả và kết luận khảo sát
#
# Chỉ so sánh điểm trên cùng tập mẫu thành công của mọi phương pháp được chọn.
# Nếu một phương pháp thất bại toàn bộ, nó chỉ xuất hiện ở coverage chứ không có điểm chất lượng.
# Ghi lại mẫu âm tiết bị đổi, đặc biệt phụ âm cuối, thanh điệu, số, phủ định và tên riêng.
# DNSMOS/WER là hai góc nhìn khác nhau; không tự cộng thành một “điểm chất lượng tổng”.
# Mẫu 10 clip chỉ giúp chọn ứng viên; khi thấy cách tốt nên khảo sát tiếp trên mẫu random lớn hơn.

# %%
comparison = summary_df.copy()
def common_success_rows(frame):
    if frame.empty or "status" not in frame:
        return frame.iloc[:0]
    valid = frame[frame["status"].eq("ok") & frame["sample_id"].isin(COMMON_IDS)]
    methods_expected = set(ok["method"])
    sets = [set(valid.loc[valid["method"].eq(m), "sample_id"]) for m in methods_expected]
    common = set.intersection(*sets) if sets else set()
    return valid[valid["sample_id"].isin(common)]

dns_valid = common_success_rows(dnsmos_df)
if len(dns_valid):
    dns_summary = dns_valid.groupby("method").agg(dnsmos_samples=("sample_id", "size"), dnsmos_sig=("dnsmos_sig", "mean"), dnsmos_bak=("dnsmos_bak", "mean"), dnsmos_ovrl=("dnsmos_ovrl", "mean"))
    comparison = comparison.join(dns_summary)
asr_valid = common_success_rows(asr_df)
if len(asr_valid):
    asr_summary = asr_valid.groupby("method").agg(asr_samples=("sample_id", "size"), word_edits=("word_edits", "sum"), reference_words=("reference_words", "sum"), char_edits=("char_edits", "sum"), reference_chars=("reference_chars", "sum"))
    asr_summary["wer"] = asr_summary["word_edits"]/asr_summary["reference_words"].replace(0, np.nan)
    asr_summary["cer"] = asr_summary["char_edits"]/asr_summary["reference_chars"].replace(0, np.nan)
    asr_summary["wer_delta_vs_original"] = np.nan
    if "original" in asr_summary.index:
        asr_summary["wer_delta_vs_original"] = asr_summary["wer"] - asr_summary.loc["original", "wer"]
    comparison = comparison.join(asr_summary[["asr_samples", "wer", "cer", "wer_delta_vs_original"]])
if rating_path.exists():
    rated = pd.read_csv(rating_path)
    score_columns = ["clarity", "naturalness", "reverb_suppression", "noise_suppression"]
    for column in score_columns:
        rated[column] = pd.to_numeric(rated[column], errors="coerce")
        invalid = rated[column].notna() & ~rated[column].between(1, 5)
        if invalid.any():
            raise ValueError(f"Điểm {column} trong phiếu nghe phải từ 1 đến 5.")
    rated["lost_or_changed_speech"] = pd.to_numeric(rated["lost_or_changed_speech"], errors="coerce")
    changed = rated["lost_or_changed_speech"]
    if (changed.notna() & ~changed.isin([0, 1])).any():
        raise ValueError("lost_or_changed_speech phải là 0 hoặc 1.")
    complete = rated[score_columns].notna().all(axis=1)
    rated["status"] = np.where(complete, "ok", "unrated")
    rated_paired = common_success_rows(rated)
    if len(rated_paired):
        listening_summary = rated_paired.groupby("method")[score_columns].mean().add_prefix("listening_")
        listening_summary["listening_samples"] = rated_paired.groupby("method").size()
        flags = rated_paired.groupby("method")["lost_or_changed_speech"]
        listening_summary["listening_changed_speech_samples"] = flags.sum(min_count=1)
        listening_summary["listening_changed_speech_rated_samples"] = flags.count()
        comparison = comparison.join(listening_summary)
    print(f"Phiếu nghe: {int(complete.sum())}/{len(rated)} dòng đã đủ 4 điểm; chỉ gộp mẫu có đủ điểm ở mọi phương pháp.")
comparison.to_csv(RUN_DIR / "comparison.csv")
comparison.reset_index().to_json(RUN_DIR / "comparison.json", orient="records", force_ascii=False, indent=2)
display(comparison.round(4))
failures = results_df[results_df["status"] != "ok"]
if len(failures):
    display(failures[["method", "sample_id", "status", "error"]])
print("Kết quả khảo sát:", RUN_DIR)

# %% [markdown]
# ### Ghi nhận sau khi nghe và đo
#
# - Phương pháp giảm vang rõ nhất: …
# - Phương pháp giữ giọng tự nhiên nhất: …
# - Các sample bị mất/đổi âm tiết: …
# - WER/CER thay đổi trên cùng tập mẫu: …
# - Chi phí thời gian/VRAM có chấp nhận được: …
# - Ứng viên cần thử tiếp và lý do: …
#
# ### Nguồn và điều kiện diễn giải
#
# - [NARA-WPE](https://github.com/fgnt/nara_wpe): thuật toán giảm vang, không dùng model học sẵn.
# - [noisereduce](https://github.com/timsainb/noisereduce): spectral gating; `prop_decrease` là mức áp dụng mask, không phải phần trăm nhiễu thật bị loại.
# - [RNNoise](https://github.com/xiph/rnnoise): demo nhận RAW PCM16 mono 48 kHz; notebook thực hiện chuyển đổi và flush frame.
# - [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet): xử lý native 48 kHz và bù trễ.
# - [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio): FRCRN/MossFormerGAN 16 kHz; chất lượng tiếng Việt cần khảo sát thực tế.
# - [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance): khử nhiễu và phục hồi là hai chế độ riêng; phục hồi có thể đổi chi tiết giọng.
# - [SGMSE](https://github.com/sp-uhh/sgmse): hai checkpoint 16 kHz khác nhau cho WSJ0-REVERB và VoiceBank-DEMAND.
# - [SAM-Audio](https://github.com/facebookresearch/sam-audio): cần quyền tải weights, xuất target/residual; có thể giữ cả người nói nền.
# - [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS): điểm dự đoán SIG/BAK/OVRL; không thay thế nghe và kiểm tra lời nói.
#
# Phiên bản gói và thông tin model của mỗi lần chạy được lưu ở `audio/<method>/method_info.json`.
# Checkpoint download/cache có thể dùng nhiều dung lượng; nó không tương đương RAM/VRAM khi suy luận.

# %% [markdown]
# ## 11. Đóng gói kết quả để tải từ Kaggle Output
#
# ZIP chứa audio của từng sample/phương pháp, CSV/JSON, phiếu nghe, biểu đồ, cấu hình và log trong run.
# Model/cache/môi trường không được đóng gói. Dùng Save Version/Save & Run All để giữ output của notebook.
# Sau khi điền lại phiếu nghe, chạy lại mục 10 và cell này nếu muốn ZIP cập nhật theo điểm mới.

# %%
EXPORT_RUN_ZIP = True
if EXPORT_RUN_ZIP:
    from IPython.display import FileLink
    archive_path = Path(shutil.make_archive(str(ARTIFACT_ROOT / RUN_DIR.name), "zip", root_dir=RUN_DIR))
    print("ZIP:", archive_path)
    print(f"Dung lượng: {archive_path.stat().st_size / 1024**2:.1f} MiB")
    print("Tải file từ Output của Kaggle sau khi lưu version.")
    if Path.cwd().resolve() == KAGGLE_WORKING_ROOT.resolve():
        display(FileLink(str(archive_path.relative_to(KAGGLE_WORKING_ROOT)), result_html_prefix="Download: "))
else:
    print("Kết quả vẫn được lưu tại:", RUN_DIR)
