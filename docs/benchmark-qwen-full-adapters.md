# Benchmark Qwen ASR 1.7B PiSSA / AdaLoRA trên Zalo train

Hai model Kaggle được đọc từ `configs/models/qwen_vimd_pissa_full.json`
và `qwen_vimd_adalora_full.json` trong package `asr.benchmark`,
giữ nguyên cấu trúc config Research: `model.key`, `backend`, `model_id`,
`revision`, `adapter_path`, `display_name`, `dtype`, `trust_remote_code`.
PiSSA đã chuyển thành LoRA trên base gốc; AdaLoRA giữ `rank_pattern` đã học.
Cả hai dùng `Qwen/Qwen3-ASR-1.7B-hf`, revision
`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`.

## Kaggle — bản chạy cho thí nghiệm này

Import script `src/asr/notebooks/benchmark/benchmark-qwen-full-adapters-kaggle.py`
vào Kaggle. Bản Kaggle dùng `.py` với các cell `# %%`, cài bằng pip.

1. Add Input dataset Zalo chứa `train.jsonl` và `audio/`.
2. Add Input hai Kaggle Model versions của `nvtd1796/qwen3-asr-vimd-adapters`:
   `pytorch/pissa-full/1` và `pytorch/adalora-full/1`.
   Mỗi version chứa trực tiếp `adapter_config.json`, `adapter_model.safetensors`.
   Không cần thư mục run, `best_epoch.txt` hoặc `run_config.json` cho chế độ config.
3. Bật Internet, chọn GPU (T4 ×2 hoặc một GPU), cấp quyền Secret `HF_TOKEN`
   nếu dùng token. Không upload `.env`.
4. Nếu mount khác, điền `ADAPTER_PATH_OVERRIDES` theo `model.key`.
   Nếu có config riêng, đặt `MODEL_CONFIG_PATHS_OVERRIDE` thành danh sách hai
   file JSON. Nếu có nhiều tập train, đặt `DATASET_ROOT_OVERRIDE`.
5. Run All: chuẩn bị → smoke test hai mẫu cho cả hai adapter → full train →
   bảng so sánh và nghe theo sample.

Đường dẫn adapter mặc định từ config:

```text
/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/pissa-full/1
/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/adalora-full/1
```

Script tự sparse-clone source benchmark đã khóa commit `9a96f3a`, chuẩn bị
Python 3.12 kể cả khi kernel là 3.13, tạo venv riêng, tự bổ sung pip nếu thiếu.
Torch được khóa 2.6.0 CUDA 12.4 theo [wheel chính thức](https://pytorch.org/get-started/previous-versions/#v260);
Transformers/PEFT giữ phiên bản huấn luyện. Không cài vào môi trường kernel.

Input chỉ đọc từ `/kaggle/input`. Config, prediction, log và ZIP nằm dưới
`/kaggle/working/qwen-adapter-benchmark`; source, Python, venv, cache base nằm
dưới `/kaggle/temp/qwen-adapter-benchmark-configs`, không được đóng vào ZIP kết quả.
`GPU_IDS=None` tự dùng các GPU nhìn thấy. Giữ RUN_ID để resume trong cùng
session; đổi RUN_ID khi đổi đầu vào/cấu hình. Session mới phải chuẩn bị lại
cache/môi trường tạm; resume hiện không tự import ZIP từ session trước.

## Môi trường riêng

Các adapter được train bằng Transformers 5.17.0 và PEFT 0.21.1. Môi trường
baseline dùng `qwen-asr==0.0.6` / Transformers 4.57.6, nên không cài đè vào đó.
Notebook tạo một virtualenv riêng bằng pip; CLI có thể dùng môi trường riêng:

```bash
python3 -m venv /tmp/qwen-adapter-benchmark
/tmp/qwen-adapter-benchmark/bin/python -m pip install -r src/asr/benchmark/requirements.txt
```

Notebook local: `src/asr/notebooks/benchmark/benchmark-qwen-full-adapters-local.ipynb`.
Chọn GPU IDs hiện có trong cell cấu hình. Hai adapter chạy lần lượt để giải
phóng VRAM giữa các model; mỗi model chia manifest theo tổng thời lượng lên
số GPU được cấu hình. Một GPU là mặc định, hai GPU cũng được hỗ trợ.

## CLI

Từ thư mục gốc repository, trên máy có các Kaggle Models đã mount, chuẩn bị
bằng config JSON:

```bash
PYTHONPATH=src python -m asr.benchmark.workflow.benchmark_qwen_adapters \
  --model-configs src/asr/benchmark/configs/models/qwen_vimd_pissa_full.json \
                  src/asr/benchmark/configs/models/qwen_vimd_adalora_full.json \
  --dataset-root /kaggle/input/your-dataset/train \
  --run-id qwen-full-train-config-v1 --prepare-only
```

Chế độ local cũ vẫn chọn `epoch_<best_epoch>_adapter` từ hai thư mục full
chứa `best_epoch.txt` và metadata run:

```bash
PYTHONPATH=src /tmp/qwen-adapter-benchmark/bin/python -m asr.benchmark.workflow.benchmark_qwen_adapters \
  --models-root /run/media/fernir/Data/Documents/HCMUS/Research/models \
  --run-id qwen-full-train-v1 --prepare-only
```

Chạy inference trên GPU bằng cùng lệnh, bỏ `--prepare-only`, thêm
`--env-file .env --gpu-ids 0` (hoặc `--gpu-ids 0 1`). Token chỉ được đọc vào
môi trường tiến trình, không ghi vào báo cáo. Omit `--limit` để chạy toàn bộ
train; smoke test có thể dùng `--limit 2 --run-id qwen-adapter-smoke-v1`.
Smoke test và full benchmark phải dùng run ID khác nhau.

Mặc định đọc `data/train+public_test/train/train.jsonl`; chỉ đánh giá các
audio được tham chiếu trong manifest, không quét toàn bộ cây WAV. Không khử
nhiễu hoặc EQ. Giải mã greedy, Vietnamese, FP16 trên GPU, tối đa 768 token,
seed 1234, batch 1. Model và processor dùng chung snapshot base đã khóa
revision, tải một lần cho cả hai adapter.

## Kết quả và chạy tiếp

Output mặc định: `output/qwen-adapter-benchmark/runs/<run-id>/`:

- `manifest.jsonl`, `resolved_config.json`, `metadata.json`, `adapter_benchmark.json`.
- Từng model: `predictions.jsonl`, `metrics.jsonl`, `summary.json`, `worker_*/errors.jsonl`.
- So sánh: `comparison.json`, `comparison.csv`, `comparison.md`.

Đo WER/CER raw và normalized ở mức corpus: tổng edit / tổng số đơn vị tham
chiếu. Normalization giữ dấu tiếng Việt và cách viết số, chuyển chữ thường,
bỏ dấu câu, gộp khoảng trắng; WER tách theo khoảng trắng, CER tính cả khoảng
trắng. Ghi coverage, latency và real-time factor theo pipeline hiện có.
Không bật DNSMOS hoặc BERTScore trong experiment này.

Chạy lại cùng run ID để tiếp tục các mẫu chưa có prediction. Cấu hình,
manifest hoặc SHA-256 adapter thay đổi sẽ từ chối trộn kết quả. Chương trình
ghi báo cáo nhưng trả lỗi nếu chưa đủ mẫu; kiểm tra lỗi rồi chạy tiếp. Không
diễn giải WER thấp trên một phần mẫu thành kết quả toàn bộ tập train.

## Kiểm thử

```bash
PYTHONPATH=src python -m unittest discover -s src/asr/tests/benchmark/unit -p 'test_*.py' -q
PYTHONPATH=src python -m unittest discover -s src/asr/tests/benchmark/integration -p 'test_qwen_hf_peft.py' -v
```

Integration test cần môi trường Transformers 5 / PEFT; dùng processor offline
qua `QWEN_TEST_PROCESSOR_PATH`, hoặc tự bỏ qua nếu chưa cung cấp. Test dùng
Qwen cấu hình nhỏ và trọng số ngẫu nhiên để kiểm tra nạp adapter/inference;
không phải phép đo độ chính xác của hai adapter thực.

API inference tham chiếu [Transformers Qwen3 ASR](https://huggingface.co/docs/transformers/model_doc/qwen3_asr).
