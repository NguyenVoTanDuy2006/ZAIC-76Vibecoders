# Benchmark ASR: source Python và tiện ích tải source từ GitHub

## 1. Mục tiêu và môi trường

Chỉ xây dựng source Python; không tạo notebook Kaggle. Notebook do bạn viết sau sẽ gọi các hàm hoặc CLI có sẵn.

Các quyết định giữ nguyên:

- So sánh bốn model PhoWhisper-medium/large và Qwen3-ASR-0.6B/1.7B.
- Đánh giá 1.545 audio mới nhất của train bằng CER/WER raw và normalized.
- Chạy lần lượt từng model trên Kaggle T4 ×2.
- Mỗi GPU nạp một bản model và xử lý một shard dữ liệu cân bằng thời lượng.
- Lấy HF_TOKEN từ Kaggle Secrets.
- Cấu hình bằng các file JSON riêng; kết nối module qua workflow/.

Repo GitHub đã xác định:

https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git

Ref mặc định là nhánh hiện tại benchmark_base_asr_model; cho phép đổi sang branch/tag/commit qua JSON.

## 2. Tải và xóa source trên Kaggle

Thêm thư mục source_manager/ ở root repo, độc lập với package benchmark. Module này chỉ dùng Python standard library và lệnh Git, để hoạt động trước khi cài dependency ASR.

source_manager/
├── __init__.py
├── config.json
├── config.py          # đọc và kiểm tra JSON
├── download.py        # tải source GitHub
├── cleanup.py         # xóa source đã tải
└── cli.py

Các hàm public:

download_source(config_path) -> DownloadedSource
cleanup_source(config_path) -> CleanupResult

DownloadedSource chứa đường dẫn source, repo URL và commit thực tế đã checkout.

Ví dụ cấu hình:

{
  "repo_url": "https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git",
  "ref": "benchmark_base_asr_model",
  "source_dir": "/kaggle/working/_source/ZAIC-76Vibecoders",
  "include_paths": [
    "src",
    "source_manager",
    "pyproject.toml",
    "uv.lock",
    "README.md"
  ]
}

Hành vi download:

- Dùng Git sparse checkout để chỉ lấy source và các file phục vụ cài đặt; không checkout dataset, model hoặc artifact.
- Trả về đường dẫn và commit đã tải để ghi vào metadata benchmark.
- Nếu source đã tồn tại và đúng repo/ref, tái sử dụng; không ghi đè một thư mục không do tiện ích quản lý.
- Ghi marker nhận diện thư mục do tiện ích tạo.
- Báo lỗi rõ ràng nếu ref chưa được push hoặc GitHub từ chối truy cập.

Hành vi cleanup:

- Xóa riêng thư mục source được cấu hình và các tệp tạm do quá trình download tạo.
- Chỉ cho phép xóa thư mục con của /kaggle/working/_source/ có marker hợp lệ; từ chối xóa /kaggle/working hoặc đường dẫn artifact.
- Gọi lại sau khi source đã bị xóa vẫn thành công.
- Không tự chạy khi benchmark hoàn tất; bạn gọi hàm khi đã kiểm tra kết quả, thuận tiện debug nếu có lỗi.

Tiện ích này phải được đưa vào phiên Kaggle trước, ví dụ qua file đính kèm hoặc tải thư mục nhỏ này từ GitHub. Nó không phụ thuộc package benchmark nên dùng được để tải toàn bộ phần source còn lại.

Package được cài theo cách thông thường, không dùng editable install, để sau khi cleanup source thì package đã cài vẫn có thể import.

## 3. Cây thư mục benchmark và luồng chạy

src/benchmark_baseline_asr/
├── __init__.py
├── __main__.py
│
├── configs/
│   ├── experiment.json
│   ├── dataset.json
│   ├── audio.json
│   ├── download.json
│   ├── secrets.json
│   ├── runtime.json
│   ├── execution.json
│   ├── normalization.json
│   ├── metrics.json
│   ├── reporting.json
│   ├── environments/
│   │   └── kaggle.json
│   └── models/
│       ├── phowhisper_medium.json
│       ├── phowhisper_large.json
│       ├── qwen3_asr_0_6b.json
│       └── qwen3_asr_1_7b.json
│
├── domain/
│   ├── __init__.py
│   ├── entities.py
│   ├── ports.py
│   └── errors.py
│
├── application/
│   ├── __init__.py
│   ├── config_service.py
│   ├── dataset_service.py
│   ├── sharding_service.py
│   ├── download_service.py
│   ├── audio_service.py
│   ├── transcription_service.py
│   ├── prediction_merge_service.py
│   ├── normalization_service.py
│   ├── evaluation_service.py
│   └── reporting_service.py
│
├── infrastructure/
│   ├── __init__.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── json_loader.py
│   ├── secrets/
│   │   ├── __init__.py
│   │   └── kaggle_secrets.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── train_jsonl.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── waveform.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── huggingface.py
│   │   ├── factory.py
│   │   ├── phowhisper.py
│   │   └── qwen_asr.py
│   ├── execution/
│   │   ├── __init__.py
│   │   └── process_runner.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── reproducibility.py
│   │   └── timing.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── text_normalizer.py
│   │   └── error_metrics.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── artifacts.py
│   └── reporting/
│       ├── __init__.py
│       └── exporters.py
│
├── workflow/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── prepare_dataset.py
│   ├── download_models.py
│   ├── transcribe_worker.py
│   ├── transcribe.py
│   ├── evaluate.py
│   ├── compare.py
│   └── benchmark.py
│
└── interfaces/
    ├── __init__.py
    └── cli.py

Domain chứa kiểu dữ liệu và interface; application xử lý từng bước; infrastructure thực hiện tương tác thư viện/file/GPU; workflow nối các bước. CLI chỉ chuyển yêu cầu vào workflow.

Luồng chạy từ notebook tương lai:

download_source()
        ↓
Cài package và dependency
        ↓
Chọn JSON config Kaggle
        ↓
prepare_dataset → download_models
        ↓
Với từng model:
  worker GPU 0 + worker GPU 1
        ↓
  ghép prediction → giải phóng hai worker
        ↓
evaluate → compare
        ↓
Kiểm tra artifact
        ↓
cleanup_source()
        ↓
Tải output từ Kaggle

benchmark là lệnh gọi các workflow xử lý benchmark theo thứ tự. Tải/xóa source là tiện ích riêng, không được tự gọi bên trong benchmark.

## 4. Cấu hình, kết quả và quy tắc xử lý

Mỗi JSON phụ trách một nhóm cấu hình: dataset, audio, download model, secrets, runtime, thực thi hai GPU, normalization, metrics, reporting và từng model. experiment.json tham chiếu các file này; profile Kaggle xác định các đường dẫn tuyệt đối.

Phân bố dữ liệu trên Kaggle:

| Nội dung | Vị trí mặc định |
|---|---|
| Dataset đầu vào | /kaggle/input/<dataset>/..., khai báo trong JSON |
| Source tải từ GitHub | /kaggle/working/_source/ZAIC-76Vibecoders |
| Model và cache tải xuống | /tmp/asr_benchmark/ |
| Kết quả benchmark | /kaggle/working/asr_benchmark/runs/<run_id>/ |

Đặt model/cache ngoài /kaggle/working để tránh đưa weights vào output tải về. Sau cleanup, source tải xuống cũng không còn trong output.

Quy tắc thực thi:

- Chia dữ liệu một lần theo tổng thời lượng, dùng cùng hai shard cho bốn model.
- Hai process riêng, mỗi process thấy một GPU; FP16, batch 1.
- Nạp model local, warmup rồi đồng bộ điểm bắt đầu xử lý.
- Greedy decoding, seed cố định, nhận dạng tiếng Việt; không đưa reference vào model.
- Nếu lỗi OOM, ghi lỗi; không tự thay dtype hoặc chuyển sang chia model.
- HF token lấy từ secret HF_TOKEN, chỉ giữ trong bộ nhớ và truyền cho downloader.
- Ghi commit source, revision model, cấu hình thực tế, phần cứng và phiên bản thư viện vào metadata.

Quy tắc đánh giá:

- Tính CER/WER raw và normalized.
- Normalized dùng Unicode NFC, chữ thường, thay dấu câu bằng khoảng trắng và gộp khoảng trắng; giữ dấu tiếng Việt và cách viết số.
- WER tách theo khoảng trắng; CER tính cả khoảng trắng.
- CER/WER toàn tập lấy tổng số lỗi chia tổng độ dài reference.
- Lưu latency từng audio, thời gian hoàn tất toàn tập và RTF hai GPU; tách thời gian download/nạp model.
- Lưu prediction sau mỗi audio để resume. Chỉ tái sử dụng khi manifest, model revision và cấu hình inference khớp.
- Báo rõ coverage; không xếp hạng lượt chạy thiếu mẫu cùng lượt chạy đầy đủ.

Artifact gồm manifest/shard, cấu hình không chứa secret, metadata, prediction/error của từng worker, prediction đã ghép, metrics từng mẫu, summary từng model và bảng so sánh JSON/CSV/Markdown.

## 5. Kiểm thử và nghiệm thu

- Tải source từ repo/ref vào thư mục mới; kiểm tra commit và sparse checkout không lấy dữ liệu thừa.
- Cleanup đúng source đã tải; gọi lại an toàn; từ chối đường dẫn artifact hoặc thư mục không có marker.
- Sau cleanup, artifact vẫn còn và package đã cài vẫn import được.
- Manifest đủ 1.545 mẫu; shard không trùng/mất ID và ghép đúng transcript.
- Kiểm tra normalization/CER/WER bằng ví dụ biết trước.
- Dùng adapter giả kiểm tra hai worker, lỗi từng mẫu, merge và resume.
- Smoke test bốn model trên hai T4 trước khi chạy toàn tập.
- evaluate/compare chạy độc lập từ prediction mà không nạp model.
- Kết quả hoàn chỉnh có đủ bốn model, bốn chỉ số CER/WER và báo cáo tốc độ.

Giả định: repo/ref có thể truy cập từ Kaggle và đã được push trước khi tải; Internet được bật khi cài dependency/download; notebook và quyền truy cập dataset do bạn thiết lập sau.