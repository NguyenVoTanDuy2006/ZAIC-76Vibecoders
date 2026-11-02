# Test LLM trên Kaggle trước khi push evaluation

Có thể giữ nguyên GitHub hiện tại: upload source local thành Kaggle Dataset
private, rồi import source đó trước source của checkout GitHub. Không cần commit,
push hay sửa remote. Gói source tự đủ cho agent/evaluation nên cũng có thể chạy
không clone GitHub.

## File đã chuẩn bị

- ZIP upload: `D:\ZAIC\lifemate-kaggle-eval.zip`.
- Launcher percent cells: `notebooks/lifemate/evaluate_llm_kaggle.py`.
- Script tạo lại gói: `scripts/package_kaggle_evaluation.py`.

ZIP chứa toàn source `lifemate` hiện tại, config, launcher và **154 validation
samples** đã tách input/gold. Có fingerprints để kiểm tra đúng source và data.
Không chứa model, audio, environment, credentials, train hoặc local-test gold.
Đây là payload chỉ phục vụ validation; không chạy `verify` toàn bộ 3 split hoặc
local test từ payload rút gọn này. Validation và evaluation kiểm tra các artifact
của validation riêng nên không cần đóng gói hai tập còn lại.

## Các bước trên Kaggle

1. Tạo Kaggle Dataset **private**, upload ZIP và tạo version. Nếu Kaggle giữ file
   ZIP nguyên thay vì giải nén, giải nén nó vào `/kaggle/working` rồi gán
   `PAYLOAD_ROOT` tới thư mục có `payload_manifest.json`.
2. Tạo notebook mới cho LLM. Add Input dataset code ở bước 1 và dataset BTC gốc
   đang dùng. Dữ liệu BTC phải có `train/train.jsonl`, `train_tools_catalog.json`,
   `train_environment/` và `zaic_tools/` đúng release. Không cần chạy ASR.
3. Bật GPU. Bật Internet nếu để launcher cài dependencies hoặc tải Qwen.
4. Copy các cell trong `evaluate_llm_kaggle.py` vào notebook rồi chạy từ đầu.
   Hoặc dùng một cell `%run` với đường dẫn thực tế tới file này trong Add Input.
5. Đọc thư mục kết quả được in ra dưới
   `/kaggle/working/lifemate-llm-evaluation/<mode>_<run_id>/`.

Launcher tự tìm payload và data root; nếu Add Input nhiều bộ dữ liệu, gán rõ
`PAYLOAD_ROOT`, `DATA_ROOT`. Nó kiểm tra fingerprint trước khi tải model, in
đường dẫn module evaluation đang import và từ chối dùng module cũ trong kernel.
Nếu notebook đã import `lifemate` từ checkout GitHub, restart session trước.

Không cần chép đè repo GitHub: `sys.path` ưu tiên `<payload>/src`. Một lần chỉ
dùng source agent/evaluator của cùng payload, tránh ghép evaluation mới với
`agent.py` cũ. `get_tool_env_factory` vẫn nạp thư viện BTC từ data root, không từ
payload và không từ GitHub.

## Hai chế độ

```python
MODE = "oracle_text"  # mặc định: kiểm tra riêng LLM/agent
SPEECH_FILE = None
```

Luồng này là **transcript chuẩn current/history → Qwen THẬT → tools THẬT trong
ToolEnv → evaluate** trên toàn 154 validation samples. Model tự chọn tool và
arguments, nhận kết quả tool rồi tiếp tục. Không dùng reference calls để trả lời.
Gold calls/final state chỉ đi vào evaluator. Transcript chuẩn dùng có chủ đích
trong chế độ oracle, chỉ trên validation, để tách ảnh hưởng ASR.

```python
MODE = "asr_text"
SPEECH_FILE = "/kaggle/input/your-asr-output/asr_predictions.jsonl"
```

Chế độ thứ hai dùng file ASR có sẵn của đội, cùng contract current/history. Đây
là chất lượng agent sau text ASR; không chạy lại model hoặc metrics ASR.
Giữ nguyên Qwen, prompt/config để so sánh hai chế độ.

## Qwen và dependencies

- Model: `Qwen/Qwen3-4B-Instruct-2507`, FP16 trên `cuda:0`, cấu hình agent hiện tại.
- `MODEL_PATH=None`: launcher resolve một HF commit và tải đúng commit vào
  `/kaggle/working/lifemate-llm-evaluation/model`, sau đó agent nạp từ path local.
  Cần Internet và dung lượng cho weights. Model public không cần token.
- Nếu đã Add Input checkpoint, gán `MODEL_PATH` tới thư mục có `config.json`,
  tokenizer và `.safetensors`. Không cần tải lại weights.
- `INSTALL_DEPENDENCIES=True`: cài Transformers **5.17.0**, Accelerate và HF Hub;
  giữ nguyên PyTorch/CUDA của Kaggle. Không cài bộ ASR SDK/PEFT.
- Dùng notebook LLM riêng; nếu Transformers đã import trước khi thay phiên bản,
  restart session. Có thể đặt `INSTALL_DEPENDENCIES=False` khi môi trường đã phù
  hợp. Loader kiểm tra hỗ trợ `prefill_chunk_size` và GPU trước khi inference.

Không khẳng định mọi GPU/context đều đủ VRAM. Nếu gặp OOM, lưu lỗi và cấu hình
rồi điều chỉnh ngân sách context/generation có chủ đích ở config; kết quả giữa
các cấu hình phải được xem như các thí nghiệm khác nhau.

## Đầu ra

- `summary.json`: mode, revision Qwen, số mẫu và điểm local.
- `agent/run.json`, `predictions.jsonl`, `results.jsonl`, `traces/`: config/input
  provenance, log/hash thật và các failure.
- `evaluation/metrics.json`: score, tool/argument/hash diagnostics, loss buckets.
- `evaluation/samples.jsonl`: lỗi cụ thể của từng mẫu.
- `evaluation/report.md`: ưu tiên sửa theo điểm mất.

Kết quả này là điểm Qwen thực sự chạy trên validation với convention scorer
local hiện tại. Nó khác self-check 100/100 dùng gold calls trước đó, và vẫn chưa
được đối chiếu với scorer chính thức của BTC. Launcher không tự mở local test.

Tạo lại ZIP sau khi sửa code local (chọn filename output mới nếu đã có):

```powershell
Set-Location D:\ZAIC\ZAIC-76Vibecoders
& C:\Users\ACER\anaconda3\python.exe scripts/package_kaggle_evaluation.py --output D:\ZAIC\lifemate-kaggle-eval-v2.zip
```

Phần chuẩn bị gói đã kiểm tra CPU/ToolEnv tại local. Chưa chạy notebook này trên
GPU Kaggle trong phiên làm việc này; cần Add Input và chạy trên tài khoản của bạn.
