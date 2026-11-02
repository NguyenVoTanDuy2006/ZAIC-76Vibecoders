# ASR source và notebook

```text
src/asr/
├── common/        # Chuẩn hóa transcript, edit distance/counts, duyệt audio lượt user
├── benchmark/     # Config, backend và workflow so sánh ASR
├── finetuning/    # Trainer Qwen3-ASR PiSSA, cấu hình và runtime riêng
├── integration/   # Sinh/đọc prediction để bàn giao cho LifeMate
├── notebooks/
│   ├── benchmark/   # Baseline, adapter và khảo sát âm thanh
│   ├── finetuning/  # Kaggle percent script huấn luyện
│   └── integration/ # Kaggle percent script sinh ASR cho agent
└── tests/
    ├── benchmark/
    └── finetuning/
```

`asr.common` chỉ dùng Python standard library để cả benchmark, trainer và
prediction contract cùng dùng. `asr.benchmark` và `asr.finetuning` có môi trường
dependency khác nhau; cài theo README của từng phần. `asr.integration` dùng
benchmark adapter để suy luận, còn `src/lifemate/` giữ agent và ToolEnv.

Notebook local dùng `.ipynb`; Kaggle launcher mới dùng `.py` với cell `# %%`
và cài dependency bằng pip. Script adapter benchmark giữ SHA lịch sử để tái lập
thí nghiệm và tự nhận diện package tại SHA đó. Output/model/cache vẫn ở vị trí
được các config và launcher quy định, không nằm trong source tree này.

Chạy test từ repo root với `PYTHONPATH=src`:

```bash
python -m unittest discover -s src/asr/tests/benchmark/unit -p 'test_*.py' -v
python -m unittest discover -s src/asr/tests/finetuning -p 'test_*.py' -v
python -m unittest discover -s tests/lifemate -p 'test_*.py' -v
```
