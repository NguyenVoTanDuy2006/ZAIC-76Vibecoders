# %% [markdown]
# # Qwen3-ASR 1.7B · PiSSA · CE + semantic SCST · Kaggle T4 ×2
# Bật Internet, chọn GPU T4 ×2, gắn ZAIC train. Source được clone từ GitHub vào /kaggle/temp.
# Đẩy module này lên branch SOURCE_REF trước khi chạy. Không cần Research/ViMD adapter.
# Tạo Kaggle Secret HF_TOKEN nếu cần; không đặt token trực tiếp vào script.

# %%
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from IPython.display import FileLink, display

DATA_ROOT = None  # Ví dụ: /kaggle/input/zaic-data/train+public_test/train
SOURCE_ROOT = None  # Chỉ đặt nếu muốn dùng source local đã gắn vào Kaggle Input.
SOURCE_REPOSITORY = "https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git"
SOURCE_REF = (
    "benchmark_base_asr_model"  # Nên thay bằng commit đã kiểm thử khi dùng GitHub
)
RESUME_BUNDLE = None  # /kaggle/input/previous-run/zaic-pissa-scst-v1-resume.zip
RUN_ID = "zaic-pissa-scst-v1"  # Đổi tên khi đổi cấu hình/dữ liệu
RUN_SMOKE = True
TEMP = Path("/kaggle/temp/zaic-qwen-pissa")
WORKING = Path("/kaggle/working/zaic-qwen-pissa")
TEMP.mkdir(parents=True, exist_ok=True)


def checked(command, **kwargs):
    return subprocess.run([str(v) for v in command], check=True, **kwargs)


# %% [markdown]
# ## Nạp source vào /kaggle/temp
# Clone nhánh SOURCE_REF vào /kaggle/temp; Kaggle Internet cần được bật.


# %%
def resolve_source():
    if SOURCE_ROOT:
        source = Path(SOURCE_ROOT)
    else:
        repo = TEMP / "repository"
        if not (repo / ".git").exists():
            checked(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    SOURCE_REPOSITORY,
                    repo,
                ]
            )
        checked(
            [
                "git",
                "-C",
                repo,
                "sparse-checkout",
                "set",
                "src/fine-tuning_with_zaic_data",
            ]
        )
        checked(["git", "-C", repo, "fetch", "--depth", "1", "origin", SOURCE_REF])
        checked(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"])
        source = repo / "src/fine-tuning_with_zaic_data"
        print(
            "Source commit:",
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "HEAD"], text=True
            ).strip(),
        )
    if not (source / "zaic_pissa/__init__.py").is_file():
        raise FileNotFoundError(
            "Source chưa có module mới. Hãy push code lên SOURCE_REF rồi chạy lại notebook."
        )
    target = TEMP / "source"
    if source.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", ".git", "*.zip"),
        )
    return target


SOURCE = resolve_source()
sys.path.insert(0, str(SOURCE))
from zaic_pissa.kaggle_setup import find_dataset, make_environment

# %% [markdown]
# ## pip · Python 3.12 · Torch 2.6 CUDA 12.4
# Kernel Kaggle có thể là Python 3.13; train chạy bằng Python 3.12 riêng qua subprocess.
# Venv thiếu pip được sửa trước khi cài dependencies. Không dùng uv hay qwen-asr SDK.

# %%
PYTHON = make_environment(SOURCE, TEMP)
ENV = dict(
    os.environ,
    PYTHONPATH=str(SOURCE),
    HF_HOME=str(TEMP / "hf"),
    TOKENIZERS_PARALLELISM="false",
    OMP_NUM_THREADS="2",
    MPLCONFIGDIR=str(TEMP / "matplotlib"),
)
try:
    from kaggle_secrets import UserSecretsClient

    secret = UserSecretsClient().get_secret("HF_TOKEN")
    if secret:
        ENV["HF_TOKEN"] = secret.strip()
except Exception:
    print("Không có Kaggle Secret HF_TOKEN; dùng quyền truy cập public hiện có.")
checked(
    [
        PYTHON,
        "-c",
        'import torch; assert torch.cuda.device_count()==2, "Chọn T4 x2"; '
        "print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)])",
    ],
    env=ENV,
)

# %% [markdown]
# ## Cấu hình và split cố định theo conversation
# Giữ nguyên audio/transcript. Bao gồm user audio trong history, không lấy assistant/tool làm target.
# Input chỉ đọc; kết quả lưu dưới /kaggle/working. Test chỉ chạy sau khi hoàn tất 5 epoch.

# %%
dataset = find_dataset(override=DATA_ROOT)
CONFIG = TEMP / "run-config.json"
CONFIG.write_text(
    json.dumps(
        {
            "dataset": {"root": str(dataset)},
            "runtime": {"cache_dir": str(TEMP / "models")},
            "output": {"root": str(WORKING), "run_id": RUN_ID},
        },
        indent=2,
    )
)
RUN = WORKING / RUN_ID


def cli(command, *arguments, distributed=False):
    launch = [PYTHON, "-m"]
    if distributed:
        launch += [
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
        ]
    checked(launch + ["zaic_pissa", command, "--config", CONFIG, *arguments], env=ENV)


if RESUME_BUNDLE and not (RUN / "checkpoints/latest.pt").is_file():
    cli("restore", "--bundle", RESUME_BUNDLE)
cli("prepare")
display(json.loads((RUN / "dataset.json").read_text()))

# %% [markdown]
# ## Smoke trên hai GPU trước lượt train chính
# Chạy CE+SCST, một nhóm accumulation ngắn, xuất/nạp lại adapter trên base gốc.
# Smoke có run_id riêng và dùng train clips; không xem test. Nếu lỗi, cell dừng trước train chính.

# %%
if RUN_SMOKE:
    smoke = WORKING / f"{RUN_ID}-smoke"
    if smoke.exists():
        # Preserve previous diagnostics instead of overwriting an earlier smoke.
        import time

        smoke.rename(WORKING / f"{RUN_ID}-smoke-previous-{time.time_ns()}")
    cli("smoke", distributed=True)

# %% [markdown]
# ## 5 epoch CE + 0.1 SCST
# Lần đầu chấm base validation trước khi gắn PiSSA. Mỗi epoch chấm validation all/latest.
# Chọn best bằng normalized PhoBERT F1 trên latest; tie-break WER, CER, epoch sớm hơn.
# Khi session ngắt giữa epoch, resume từ epoch hoàn tất gần nhất, cùng world size.

# %%
arguments = ["--resume"] if (RUN / "checkpoints/latest.pt").is_file() else []
try:
    cli("train", *arguments, distributed=True)
finally:
    # Export the last completed epoch even if a later epoch fails.
    if (RUN / "checkpoints/latest.pt").is_file():
        cli("bundle")

# %% [markdown]
# ## So sánh test: base gốc và best adapter nạp mới
# Cùng processor, inference, normalization và PhoBERT. Báo cáo WER/CER raw+normalized,
# BERTScore F1, coverage và thời gian cho all/latest; test không tham gia chọn checkpoint.

# %%
cli("evaluate", distributed=True)
cli("bundle")
comparison = json.loads((RUN / "test/comparison.json").read_text())
import pandas as pd

display(
    pd.DataFrame(
        [
            {"model": model, "subset": subset, **metrics}
            for model in ["base", "best"]
            for subset, metrics in comparison[model].items()
        ]
    )
)

for path in sorted((WORKING / "bundles").glob(f"{RUN_ID}-*.zip")):
    display(FileLink(str(path)))
