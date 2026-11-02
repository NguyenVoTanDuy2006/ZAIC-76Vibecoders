# %% [markdown]
# # Prediction ASR cho LifeMate — Qwen3-ASR-1.7B + PiSSA
# Chạy bằng Python của môi trường native ASR Transformers 5 / PEFT đã có trong repo.
# Local từ repo root: python src/asr/notebooks/integration/predict_asr.py
# Kaggle: mở script dạng percent cells; đặt ZAIC_PROJECT_ROOT nếu cwd không phải repo.

# %%
import os
from pathlib import Path
import sys

start = Path(os.environ.get("ZAIC_PROJECT_ROOT",
    str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd()))).resolve()
PROJECT_ROOT = next((root for root in [start, *start.parents]
                     if (root / "pyproject.toml").is_file() and (root / "src/lifemate").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Đặt ZAIC_PROJECT_ROOT tới thư mục ZAIC-76Vibecoders")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asr.integration.predict import predict_lifemate
from lifemate.paths import default_data_root, asr_output_dir


# %%
def main():
    # Sửa đường dẫn theo local/Kaggle. Không giới hạn ID: chạy toàn bộ split.
    DATA_ROOT = default_data_root()
    SPLIT = "train"
    MODEL_CONFIG = PROJECT_ROOT / "src/asr/benchmark/configs/models/qwen_vimd_pissa_full.json"
    MODEL_PATH = None     # Thư mục base checkpoint local; None = dùng HF cache
    ADAPTER_PATH = None   # None = đường dẫn PiSSA trong MODEL_CONFIG; đổi sang best_adapter/ nếu cần
    OUTPUT_DIR = asr_output_dir(SPLIT, PROJECT_ROOT)

    report = predict_lifemate(
        DATA_ROOT, SPLIT, MODEL_CONFIG, OUTPUT_DIR,
        model_path=MODEL_PATH, adapter_path=ADAPTER_PATH,
        local_files_only=True, device="cuda:0",
    )
    print("ASR file cho agent:", report["prediction_path"])
    print("Conversations:", report["conversations"], "| Audio clips:", report["audio_clips"])


# %%
if __name__ == "__main__":
    main()
