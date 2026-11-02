# Qwen3-ASR 1.7B · PiSSA · ZAIC CE + SCST

Module độc lập, chạy **Kaggle T4 ×2**, bắt đầu từ Qwen gốc, không nạp adapter ViMD và không import `Research`. Launcher là [Python percent script](../notebooks/finetuning/train_qwen_pissa_scst_kaggle.py). Plan đã chốt nằm ở `docs/superpowers/plans/2026-11-01-zaic-qwen-pissa-scst.md`.

## Chạy trên Kaggle

1. Push thư mục này lên branch `asr` của GitHub.
2. Import `src/asr/notebooks/finetuning/train_qwen_pissa_scst_kaggle.py` vào Kaggle, chọn **T4 ×2**, bật Internet và gắn dataset ZAIC chứa `train.jsonl` cùng `audio/`. Notebook clone branch GitHub vào `/kaggle/temp/zaic-qwen-pissa/repository`, lấy riêng thư mục ASR, rồi cài và chạy từ bản clone đó. Nếu có nhiều dataset train, đặt `DATA_ROOT` ở cell đầu. Thêm **Kaggle Secret `HF_TOKEN`** nếu cần; token không được lưu vào file output.
3. Run All: cài bằng **pip** vào Python 3.12 riêng → chuẩn bị manifest → smoke hai GPU → base validation → train 3 epoch → test base/best → ZIP kết quả.

Kernel Kaggle có thể là Python 3.13; subprocess chạy Python 3.12 riêng. Nếu máy không có Python 3.12, bootstrap dùng [pbs-installer](https://github.com/frostming/pbs-installer). Torch 2.6.0/cu124, Transformers 5.17.0, PEFT 0.21.1 và scorer được pin trong `requirements.txt`. Không cài `qwen-asr` SDK vì stack này dùng native Qwen trong Transformers 5. Venv đã tồn tại nhưng thiếu pip được sửa bằng `ensurepip`, sau đó host pip `--python` nếu cần.

Source, venv và cache model nằm trong `/kaggle/temp/zaic-qwen-pissa`; input đọc từ `/kaggle/input`; run nằm ở `/kaggle/working/zaic-qwen-pissa/<run_id>`. Không xóa source/cache khi train đang chạy.

## Cấu hình đã chốt

| Thành phần | Giá trị |
|---|---|
| Base | `Qwen/Qwen3-ASR-1.7B-hf` @ `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` |
| PiSSA | r=8, alpha=16, dropout=0, bias=none, `pissa_niter_4`; q/v encoder + decoder |
| Train | 3 epoch, AdamW 5e-5, decay .01, linear schedule, warmup 3%, clip 1 |
| Batch | 1/GPU × 2 GPU × accumulation 8 = 16; nhóm cuối dùng đúng số microbatch còn lại |
| Precision | FP16 autocast + scaler, non-reentrant gradient checkpointing |
| Seed | 1234; PiSSA initialization chung seed cho hai rank, giữ RNG của lượt train |
| Scorer | Frozen FP32 `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6`, layer 12 |
| Decode | Vietnamese, max_new_tokens=768; greedy validation, một sampled + một greedy/batch |

Các config tách thành `configs/model.json`, `data.json`, `training.json`, ghép bởi `experiment.json`. File override chỉ cần chứa các giá trị muốn thay; field sai tên bị từ chối. Mặc định được thiết kế cho T4 ×2. `fp32/world_size=1/device=cpu` dành cho kiểm thử mô hình nhỏ.

Mỗi batch có cả CE và SCST:

```text
A = stop_gradient(F1(sampled, reference) - F1(current_model_greedy, reference))
SCST = - A * sum(log p(actual_sampled_token_ids | audio, prefix))
loss = assistant_token_mean_CE + 0.1 * SCST
```

Advantage giữ nguyên, không scale/clip/chuẩn hóa theo độ dài. Sampling temperature=1, top_p=1, top_k=0. Replay dùng đúng token đã sinh, không decode rồi tokenize lại; cùng mask structural tokens với rollout. Tính đến EOS đầu tiên, bỏ prompt/padding/phần sau EOS. CE backward trước, giải phóng graph, rồi SCST backward để giảm peak memory. Gradient vẫn đi qua audio encoder; scorer không có gradient. Rank có advantage bằng 0 vẫn chạy backward để DDP đồng bộ.

Reward chuẩn hóa NFC, lowercase, punctuation thành space, gộp whitespace rồi PyVi; giữ cách viết số. Không IDF/rescale. Hypothesis rỗng có reward 0 (cả hai rỗng: 1). Văn bản dài được chia cửa sổ và ghép content embeddings để so khớp toàn bộ, đánh dấu `semantic_chunked`: đây là xấp xỉ do từng cửa sổ có context riêng, không phải PhoBERT context dài. Scorer lỗi/nonfinite làm dừng run, không thay bằng reward giả.

## Dữ liệu và đánh giá

Lấy latest + user audio trong history thành các cặp audio/transcript độc lập. Không dùng assistant/tool history làm nhãn. Giữ nguyên audio, transcript, cách viết số; không khử nhiễu. Kiểm tra WAV mono PCM16/16kHz, path, label và PCM hash.

Split theo conversation: sort ID → shuffle Python Random(1234) → test ceil(10%) đầu → validation ceil(10%) tiếp → train còn lại. Dataset hiện tại có:

| Split | Clips | Latest | Giờ |
|---|---:|---:|---:|
| Train | 4,043 | 1,235 | 5.46345 |
| Validation | 470 | 155 | 0.65574 |
| Test | 523 | 155 | 0.71006 |

Conversation/PCM không chồng giữa split; không khẳng định speaker-disjoint vì dataset không cung cấp speaker ID. Train dùng DistributedSampler có padding để hai rank cùng số backward; padding được ghi rõ trong log. Validation/test chia strided không padding, kiểm tra đủ ID và không trùng.

Chấm base validation **trước khi gắn PiSSA**. Mỗi epoch chấm latest và all: raw/normalized WER, CER theo tổng edit counts toàn corpus, CER gồm khoảng trắng, mean BERTScore F1, coverage, truncation trong predictions, latency/RTF. Best lấy **normalized PhoBERT F1 trên validation latest**; tie trong 1e-6 lấy normalized WER thấp hơn, CER thấp hơn, rồi epoch sớm hơn. Không early stopping. Test chỉ chạy khi đủ 3 epoch, dùng base gốc và best adapter nạp mới, cùng inference/scorer; test không chọn checkpoint.

## Output và resume

```text
<run_id>/
  config.json, provenance.json, dataset.json
  manifests/{train,validation,test}.jsonl
  pissa_initial/                  # factors ban đầu, phục vụ conversion
  adapters/epoch-01..03/          # LoRA rank 16, nạp trên original base
  best_adapter/, best_model.json, best.json
  validation/{base,epoch-01..03}/{predictions.jsonl,summary.json}
  test/{base,best}/, test/comparison.json
  history.json
  checkpoints/latest.pt          # chỉ giữ native state mới nhất
```

PiSSA live rank 8 dùng residual base; export ghép current delta và initial delta thành LoRA rank 16 trên base gốc. Không nạp raw PiSSA factors lên original base. Export giữ nguyên weights/config/train flags/RNG của model đang train.

`latest.pt` chứa initial factors, residual weights, current factors, optimizer/scheduler/scaler, RNG từng rank, history/best/progress, config/data fingerprint (bao gồm cấu hình scorer). Ghi file tạm rồi replace atomic. Một epoch chỉ được commit sau khi có adapter và validation. Khi resume, `latest.pt` là nguồn trạng thái chính; output epoch chưa commit được chuyển vào `orphaned/`, giữ lại để kiểm tra, không đưa vào ZIP.

Run có khóa tiến trình để chặn hai lượt train/resume/evaluate/bundle cùng ghi. Rank 0 giữ khóa trong DDP; OS tự nhả khóa khi tiến trình chết. Bundle giữ cùng khóa, chỉ lấy epoch đã commit và tái tạo history/best từ checkpoint, kể cả khi các JSON phụ chưa kịp ghi sau sự cố.

Hai file ZIP dưới `.../bundles/`:

- `*-reports.zip`: reports, manifests, converted adapters; không chứa native checkpoint/base/venv/secrets.
- `*-resume.zip`: thêm initial adapter và latest native checkpoint, đủ artifact để tiếp tục.

Để tiếp tục sau session Kaggle ngắt: tải resume ZIP của epoch hoàn tất gần nhất, gắn lại làm Input, đặt `RESUME_BUNDLE`, giữ nguyên code trên GitHub, config và world size, rồi Run All. Mount data/output/cache có thể đổi; PCM/labels/splits và cấu hình train/scorer phải giữ nguyên. Session bị dừng giữa epoch sẽ chạy lại epoch đó. File provenance kiểm tra source; dùng cùng commit của lượt trước. Checkpoint native là pickle: chỉ dùng artifact do bạn tin cậy tạo ra.

CLI kiểm tra các dependency pins trước khi ghi provenance và từ chối runtime khác khi resume/evaluate. Torch chấp nhận bản CPU để kiểm thử và bản CUDA 12.4 cho Kaggle; bản PyPI `2.6.0` và wheel `2.6.0+cu124` được coi cùng build CUDA. Cho phép đổi Python patch trong nhánh 3.12, giữ runtime gốc trong provenance và ghi patch hiện tại ở `last_runtime`; Python 3.13 bị từ chối.

## CLI và kiểm thử

Đặt `PYTHONPATH=src`, dùng Python của environment đã cài requirements:

```bash
python -m asr.finetuning prepare --config overrides.json
python -m torch.distributed.run --standalone --nproc_per_node=2 --module asr.finetuning smoke --config overrides.json
python -m torch.distributed.run --standalone --nproc_per_node=2 --module asr.finetuning train --config overrides.json
# Thêm --resume để tiếp tục ở epoch boundary.
python -m torch.distributed.run --standalone --nproc_per_node=2 --module asr.finetuning evaluate --config overrides.json
python -m asr.finetuning bundle --config overrides.json
python -m unittest discover -s src/asr/tests/finetuning -p 'test_*.py' -v
```

Suite gồm fixture WAV, masking/gradient sign, frozen reward/full text chunking, PiSSA export/reload, atomic checkpoint/resume, native Qwen nhỏ và hai tiến trình CPU/Gloo. Test native processor/PhoBERT dùng cache offline nếu có (override `ZAIC_QWEN_PROCESSOR`, `ZAIC_PHOBERT_PATH`). Test hai tiến trình cần quyền TCP nội bộ. CPU test không chứng minh peak memory/thời gian trên T4; launcher yêu cầu smoke trên hai GPU trước lượt train chính. Kết quả chất lượng phải được xác nhận bằng benchmark, không suy ra từ việc loss giảm.
