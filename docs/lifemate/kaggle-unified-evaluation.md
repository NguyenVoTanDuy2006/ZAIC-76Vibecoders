# Chạy Agent + ToolEnv + evaluation trên Kaggle

Notebook sử dụng: `notebooks/lifemate/test_lifemate.ipynb`.
Notebook chứa một snapshot source của toàn gói `lifemate` và cấu hình evaluation,
được kiểm tra SHA256 trước import. Chỉ cần upload notebook này và Add Input dataset
BTC. Không cần upload riêng `agent.py`, source ZIP hoặc clone GitHub.

## Chuẩn bị trên Kaggle

1. Import bản mới của `test_lifemate.ipynb` vào Kaggle Notebook.
2. Add Input dataset BTC chứa thư mục cha của `train/` và `public_test/`.
   Phần thử nghiệm sử dụng `train/train.jsonl`, catalog, audio và SDK/environment
   tương ứng; không chọn public test.
3. Bật GPU. Bật Internet nếu dùng model ID chưa cache hoặc để cell dependencies
   cài thư viện. `ALLOW_MODEL_DOWNLOADS=True` cho phép tải checkpoint khi người
   dùng chạy notebook; thao tác build notebook tại local không tải model.
4. Nếu đã import LifeMate từ source cũ trong kernel, Restart Session trước.
5. Chạy các cell theo thứ tự từ mục 1 đến mục 8.

Dataset không được nhúng vào notebook. SDK `zaic_tools` luôn nạp từ dataset BTC
được gắn vào Kaggle, không phải từ snapshot source của đội.

## Cấu hình cần chú ý

```python
DATA_ROOT = None
SAMPLE_COUNT = 10
SAMPLE_IDS = None
ASR_MODE = "oracle_text"
SPEECH_FILE = None
MODEL_SOURCE = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = None
DEVICE = "cuda:0"
AGENT_OPTIONS = {}
```

- `DATA_ROOT=None`: tự tìm bộ dữ liệu trong `/kaggle/input`; nếu có nhiều bộ,
  gán đường dẫn cha chứa `train/`, ví dụ
  `/kaggle/input/datasets/khangl123/zalo-ai/train+public_test`.
- `SAMPLE_COUNT=1`: smoke test; `10`: thử ngắn; `None`: toàn bộ validation.
  Đây là các mẫu thuộc validation được tách cố định từ train, **không phải 10 dòng
  đầu train hoặc 10% cuối train**. Split vật lý cho ToolEnv vẫn là `train`.
- `SAMPLE_IDS`: chọn rõ IDs thuộc validation nếu cần tái lập; lấy IDs từ selection
  manifest được in ra. Không giả định ID liên tục.
- `MODEL_SOURCE`: có thể là thư mục checkpoint đã Add Input có `config.json`,
  tokenizer và weights. Khi có checkpoint local/cache đầy đủ có thể đặt
  `ALLOW_MODEL_DOWNLOADS=False` để chạy offline.
- `MODEL_REVISION`: commit checkpoint HF muốn pin; `None` phải được ghi nhận là
  chưa pin, không tự coi tên model là danh tính weights bất biến.
- `AGENT_OPTIONS={}`: dùng mặc định Agent hiện tại, 48 lượt sinh model và 12 tool
  calls; brief/write check/terminal check tính chung vào số lượt sinh model.

Notebook giữ hai chế độ tách biệt:

```python
ASR_MODE = "oracle_text"  # transcript chuẩn current/history, không chạy ASR
```

Chế độ này do người dùng chủ động chọn để kiểm tra Agent. Gold calls/final state
không đi vào Actor hoặc dùng làm output giả; chúng chỉ vào evaluator sau inference.

```python
ASR_MODE = "asr_text"
SPEECH_FILE = "/kaggle/input/your-asr-output/asr_predictions.jsonl"
```

Chế độ này nhận file ASR của đội với `id/current_text/history_user_texts` và
`history_index` gốc. Không chạy ASR model hoặc thay prediction thiếu bằng oracle.
Muốn so sánh ảnh hưởng transcript, giữ cùng source, model/config và subset IDs.

## Luồng và từng cell

| Mục | Việc thực hiện |
|---|---|
| 1 | Chọn dữ liệu, subset, transcript và model |
| 2 | Giải nén source snapshot, kiểm tra hash, chọn đúng import |
| 3 | Chuẩn bị grouping/split, selection chung và preflight trước model |
| 4 | Cài dependencies khi cần; giữ Torch/CUDA của Kaggle |
| 5 | Nạp Qwen một lần hoặc dùng lại backend có cùng source/revision/device |
| 6 | AgentSession -> SDK ToolEnv -> lưu log/hash -> validate/replay -> evaluate |
| 7 | In summary và links tải artifact |
| 7b | In đầy đủ metrics.json, tất cả mẫu trong samples.jsonl và report.md |
| 8 | Xem lỗi từng mẫu và status |
| 9 | Đọc trace tùy chọn, không chạy model hoặc tool lại |

Preflight chỉ dùng thư viện chuẩn và metadata/SDK, không nạp weights. Chuẩn bị
lần đầu có thể mất thời gian vì grouping và kiểm tra toàn bộ audio train.
Inference và evaluator cùng dùng selection manifest; chạy 10 mẫu sẽ chấm đúng
10 mẫu. Không sửa trực tiếp integrity của bundle gốc để tạo subset.

Mỗi mẫu chỉ có một ToolEnv inference. Chỉ execution module gọi `env.call()`.
Submission lấy nguyên `env.get_tool_calls()` và `env.get_state_hash()`, giữ call
thất bại đã vào SDK. Evaluator dùng các env replay độc lập, không reset session
inference. Các proposal bị chặn trước thực thi chỉ nằm trong trace.

## Đọc kết quả

Mỗi phép thử có một thư mục mới dưới `/kaggle/working/lifemate-unified-evaluation`.
Summary chứa đường dẫn tới selection/config, prediction, trace và evaluation.
Các artifact quan trọng:

- `summary.json`: trạng thái tổng hợp và điểm local.
- `predictions.jsonl`: ba trường `id`, `tool_calls`, `state_hash` của SDK.
- `results.jsonl` và `traces/`: từng mẫu, lỗi, model output và actual tool result.
- `metrics.json`: điểm và diagnostics trên đúng mẫu số selection.
- `samples.jsonl`: `K/G/M/H`, calls thiếu/thừa, arguments/state khác nhau.
- `report.md`: nhóm lỗi và điểm mất để ưu tiên kiểm tra.

Muốn xem toàn bộ nội dung ba file evaluation ngay trong notebook, chạy mục **7b**.
Cell tự lấy thư mục từ `LAST_RESULT` của run vừa hoàn tất. Để đọc run đã lưu sau
restart, đặt `EVALUATION_DIR` tới thư mục chứa cả ba file. Thao tác này chỉ đọc
kết quả, không chạy lại Qwen/ToolEnv hoặc tính điểm lại.

Hai phép đối chiếu hash có ý nghĩa khác nhau:

1. Submitted hash bằng replay hash: log và hash của pipeline nhất quán.
2. Replay hash bằng gold hash: trạng thái cuối đúng với reference.

`finished` chỉ là tình trạng vận hành. Điểm submission được tính từ calls/hash
hợp lệ; runtime status và trace health báo riêng. `finished` với hash sai vẫn có
thể 0 điểm. Dừng ở `step_limit` với submission đúng không tự động bị trừ về 0 chỉ
vì tên status. Prediction thiếu hoặc không replay được không được giả thành `[]`.

Quy ước local hiện dùng fuzzy NFC/lowercase/whitespace token F1 **> 0.6**, ghép
one-to-one không xét thứ tự, và call F1 `2M/(K+G)` qua điều kiện state hash theo hình
người dùng cung cấp. File `docs/requirement/evaluation.md` ghi ngưỡng `>= 0.6`
và thiếu hệ số 2 trong công thức; chưa dùng bản text mâu thuẫn đó để âm thầm đổi
scorer. Kết quả luôn là ước lượng local, chưa được xác minh với scorer BTC.

## Chạy lại và mở rộng

Sau lượt smoke, đổi `SAMPLE_COUNT` ở mục 1, chạy mục 3 để tạo plan mới, rồi mục
5 -> 6 -> 7 -> 8. Cell 5 dùng lại model nếu cùng source/revision/device. Không cần
chạy lại cell dependencies khi model đã nạp và thư viện không đổi.

Đọc trace sau restart mà không nạp Qwen:

```python
TRACE_SAMPLE_ID = "ID_trong_run_da_luu"
TRACE_RESULTS_FILE = "/kaggle/working/.../results.jsonl"
```

Chạy cell trace cuối notebook. Kết quả cũ được giữ nguyên; không thực thi lại tool.

Sau khi sửa source local, tạo lại snapshot:

```powershell
python scripts/build_lifemate_test_notebook.py
python scripts/build_lifemate_test_notebook.py --check
```

Upload lại notebook mới và restart Kaggle session. Source local thay đổi không tự
đi vào notebook đã upload. Không trộn cell source giữa các run đang so sánh.
CPU/scripted/SDK tests xác minh dây nối; điểm Qwen thật và thời gian GPU chỉ có
sau khi chạy notebook trên Kaggle. Latency Agent từ transcript cache không phải
latency toàn bộ ASR -> Agent.
