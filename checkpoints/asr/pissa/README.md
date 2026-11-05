# Fine-tuning ASR với PiSSA

Nhánh `fine-tuning-asr-with-pissa` chứa mã huấn luyện Qwen3-ASR-1.7B bằng PiSSA,
checkpoint tốt nhất của lượt chạy ZAIC đã hoàn tất và cấu hình để suy luận.
Mã huấn luyện ở [`src/asr/finetuning`](../../../src/asr/finetuning/), launcher
Kaggle ở [`train_qwen_pissa_scst_kaggle.py`](../../../src/asr/notebooks/finetuning/train_qwen_pissa_scst_kaggle.py).

## Checkpoint và kết quả

`best_adapter/` là **adapter LoRA đã chuyển đổi từ PiSSA**, nạp trực tiếp trên
base Qwen gốc. Không nạp nó như raw PiSSA, và không cần `pissa_initial/` để suy
luận. Base là `Qwen/Qwen3-ASR-1.7B-hf` tại revision
`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`; trọng số base không nằm trong
repo này. `model_config.json` dùng được với `asr.integration.predict` và tự tìm
`best_adapter/` theo vị trí file cấu hình.

Lượt chạy `zaic-pissa-scst-v1` có 3 epoch, chọn **epoch 3** theo PhoBERT F1 đã
chuẩn hóa trên các audio latest thuộc validation: **0.85875**. Trên test 523
audio, normalized WER của base là **0.25915**, của adapter là **0.15471**.
Các chỉ số đầy đủ nằm trong `best.json` và `test_comparison.json`; provenance
runtime/source ở `provenance.json`. Đây là kết quả của lượt chạy hiện có, không
phải kết quả huấn luyện lại trên nhánh này.

File trọng số `best_adapter/adapter_model.safetensors` (19.2 MB) và tokenizer
`best_adapter/tokenizer.json` (11.4 MB) lưu bằng Git LFS. Sau khi clone nhánh:

```bash
git switch fine-tuning-asr-with-pissa
git lfs pull
```

Kiểm tra `best_adapter/adapter_model.safetensors` là file nhị phân, không phải
con trỏ LFS; SHA-256 mong đợi:
`d98e4bb651103aaced5c5f110b6e815ef5bfd84a8e97b79f85819e26c02e2c52`.
`checkpoints/latest.pt` là state huấn luyện để resume, không cần để suy luận
và không được phát hành trong nhánh này.

## Chạy suy luận

Dùng Python 3.12 với môi trường native Transformers 5/PEFT của
[`requirements.txt`](../../../src/asr/finetuning/requirements.txt). Cần GPU để
chạy model 1.7B và base checkpoint đúng revision trong Hugging Face cache hoặc
trong thư mục local. Đặt `DATA_ROOT` đến thư mục **cha** chứa `train/`,
`public_test/`; mỗi split có `<split>.jsonl` và `audio/`. Ví dụ từ repo root:

```bash
export PYTHONPATH=src
export LIFEMATE_DATA_ROOT=/path/to/train+public_test
python - <<'PY'
from pathlib import Path
from asr.integration.predict import predict_lifemate
from lifemate.paths import asr_output_dir, default_data_root

split = "public_test"
result = predict_lifemate(
    data_root=default_data_root(),
    split=split,
    model_config_path=Path("checkpoints/asr/pissa/model_config.json"),
    output_dir=asr_output_dir(split, Path.cwd()),
    model_path=None,  # hoặc /path/to/base; None chỉ dùng cache có sẵn
    local_files_only=True,
    device="cuda:0",
)
print(result["prediction_path"])
PY
```

Hàm này dự đoán audio hiện tại và toàn bộ user-history audio; file
`asr_predictions.jsonl` là đầu vào chuẩn cho LifeMate. Trên Kaggle, dữ liệu đọc
từ `/kaggle/input`, output mặc định ở `/kaggle/working/lifemate/asr/<split>/`.
Nếu base chưa có trong cache, tải đúng revision từ Hugging Face trước khi chạy
hoặc truyền đường dẫn snapshot local qua `model_path`.

## Huấn luyện lại trên Kaggle

1. Import `src/asr/notebooks/finetuning/train_qwen_pissa_scst_kaggle.py` như
   Python percent notebook. Chọn **T4 ×2**, bật Internet và Add Input dataset ZAIC
   chứa `train/train.jsonl` cùng `train/audio/`. Thêm Kaggle Secret `HF_TOKEN` nếu
   model cần xác thực.
2. Trong cell cấu hình, đặt `SOURCE_REF = "fine-tuning-asr-with-pissa"` khi nhánh
   đã có trên GitHub, hoặc đặt `SOURCE_ROOT` đến source đã gắn dưới
   `/kaggle/input`. Đặt `DATA_ROOT` nếu Kaggle tìm thấy nhiều dataset phù hợp.
3. Đặt `RUN_ID` mới nếu muốn tạo lượt train mới, rồi **Run All**. Launcher cài
   dependencies bằng pip vào Python 3.12, chạy smoke hai GPU, train 3 epoch,
   đánh giá base/best và tạo ZIP tại `/kaggle/working/zaic-qwen-pissa/bundles/`.

Config mặc định: PiSSA rank 8, alpha 16 trên `q_proj`/`v_proj`, CE + 0.1 SCST,
batch hiệu dụng 16, AdamW 5e-5. Best được chọn trên validation latest bằng
normalized PhoBERT F1; test không tham gia chọn checkpoint. Chi tiết về split,
resume, export và CLI có trong
[`src/asr/finetuning/README.md`](../../../src/asr/finetuning/README.md).

Khi chạy lại, checkpoint mới nằm trong
`/kaggle/working/zaic-qwen-pissa/<RUN_ID>/best_adapter/`; nó không tự thay đổi
adapter đã phát hành trong nhánh này.
