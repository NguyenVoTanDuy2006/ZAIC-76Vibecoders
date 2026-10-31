# Benchmark Qwen ASR 1.7B PiSSA / AdaLoRA trên Zalo train

Hai model được chọn từ `qwen_pissa_full` và `qwen_adalora_full`, theo
`best_epoch.txt`. Với artifacts hiện tại, cả hai là `epoch_2_adapter`.
Không dùng bản `small`, không dùng `pissa_initial` hoặc checkpoint optimizer.
PiSSA có metadata `converted_standard_lora`, nạp trên base gốc; AdaLoRA giữ
`rank_pattern` đã học. Cả hai dùng `Qwen/Qwen3-ASR-1.7B-hf`, revision
`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`.

## Kaggle — bản chạy cho thí nghiệm này

Import script `output/jupyter-notebook/benchmark-qwen-full-adapters-kaggle.py`
vào Kaggle. Bản Kaggle dùng `.py` với các cell `# %%`, cài bằng pip.

1. Add Input dataset Zalo chứa `train.jsonl` và `audio/`.
2. Add Input chứa hai run `qwen_pissa_full`, `qwen_adalora_full`; có thể là
   hai datasets riêng. Giữ `best_epoch.txt`, metadata run và checkpoint được
   chọn. PiSSA cần `training_method.json` trong checkpoint; AdaLoRA cần
   `run_config.json` để xác định revision base.
3. Bật Internet, chọn GPU (T4 ×2 hoặc một GPU), cấp quyền Secret `HF_TOKEN`
   nếu dùng token. Không upload `.env`.
4. Nếu tên folder khác hoặc gắn nhiều bản dataset, điền `DATASET_ROOT_OVERRIDE`,
   `PISSA_RUN_OVERRIDE`, `ADALORA_RUN_OVERRIDE` trong cell cấu hình.
5. Run All: chuẩn bị → smoke test hai mẫu cho cả hai adapter → full train →
   bảng so sánh và nghe theo sample.

Script tự sparse-clone source benchmark đã khóa commit `d564ee3`, chuẩn bị
Python 3.12 kể cả khi kernel là 3.13, tạo venv riêng, tự bổ sung pip nếu thiếu.
Torch được khóa 2.6.0 CUDA 12.4 theo [wheel chính thức](https://pytorch.org/get-started/previous-versions/#v260);
Transformers/PEFT giữ phiên bản huấn luyện. Không cài vào môi trường kernel.

Input chỉ đọc từ `/kaggle/input`. Config, prediction, log và ZIP nằm dưới
`/kaggle/working/qwen-adapter-benchmark`; source, Python, venv, cache base nằm
dưới `/kaggle/temp/qwen-adapter-benchmark`, không được đóng vào ZIP kết quả.
`GPU_IDS=None` tự dùng các GPU nhìn thấy. Giữ RUN_ID để resume trong cùng
session; đổi RUN_ID khi đổi đầu vào/cấu hình. Session mới phải chuẩn bị lại
cache/môi trường tạm; resume hiện không tự import ZIP từ session trước.

## Môi trường riêng

Các adapter được train bằng Transformers 5.17.0 và PEFT 0.21.1. Môi trường
baseline dùng `qwen-asr==0.0.6` / Transformers 4.57.6, nên không cài đè vào đó.
Notebook tạo một virtualenv riêng bằng pip; CLI có thể dùng môi trường riêng:

```bash
python3 -m venv /tmp/qwen-adapter-benchmark
/tmp/qwen-adapter-benchmark/bin/python -m pip install -r requirements/qwen-adapter-benchmark.txt
```

Notebook local: `output/jupyter-notebook/benchmark-qwen-full-adapters-local.ipynb`.
Chọn GPU IDs hiện có trong cell cấu hình. Hai adapter chạy lần lượt để giải
phóng VRAM giữa các model; mỗi model chia manifest theo tổng thời lượng lên
số GPU được cấu hình. Một GPU là mặc định, hai GPU cũng được hỗ trợ.

## CLI

Từ thư mục gốc repository, chuẩn bị và kiểm tra dữ liệu trên CPU:

```bash
PYTHONPATH=src /tmp/qwen-adapter-benchmark/bin/python -m benchmark_baseline_asr.workflow.benchmark_qwen_adapters \
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
PYTHONPATH=src python -m unittest discover -s src/benchmark_baseline_asr/tests/unit -p 'test_*.py' -q
PYTHONPATH=src python -m unittest discover -s src/benchmark_baseline_asr/tests/integration -p 'test_qwen_hf_peft.py' -v
```

Integration test cần môi trường Transformers 5 / PEFT; dùng processor offline
qua `QWEN_TEST_PROCESSOR_PATH`, hoặc tự bỏ qua nếu chưa cung cấp. Test dùng
Qwen cấu hình nhỏ và trọng số ngẫu nhiên để kiểm tra nạp adapter/inference;
không phải phép đo độ chính xác của hai adapter thực.

API inference tham chiếu [Transformers Qwen3 ASR](https://huggingface.co/docs/transformers/model_doc/qwen3_asr).
