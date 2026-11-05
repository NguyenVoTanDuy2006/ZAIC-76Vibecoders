"""Đường dẫn dùng chung cho LifeMate trên local và Kaggle.

DATA_ROOT là thư mục CHA chứa train/, public_test/, không phải thư mục train/.
File ASR là đầu vào đã có của agent; các hàm ở đây chỉ chọn path, không chạy ASR.
"""
from __future__ import annotations

import os
from pathlib import Path
import re


def find_project_root():
    # Repo chứa pyproject.toml và src/lifemate; đây KHÔNG phải thư mục dữ liệu.
    # Kaggle: đặt ZAIC_PROJECT_ROOT nếu repo nằm ngoài các thư mục cha của notebook.
    explicit = os.environ.get("ZAIC_PROJECT_ROOT")
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        here = Path.cwd().resolve()
        candidates = [here, *here.parents, Path(__file__).resolve().parents[2]]
    for root in candidates:
        if (root / "pyproject.toml").is_file() and (root / "src/lifemate").is_dir():
            return root
    raise FileNotFoundError("Không tìm thấy repo. Đặt ZAIC_PROJECT_ROOT tới ZAIC-76Vibecoders.")


def _on_kaggle():
    return Path("/kaggle").is_dir()


def default_data_root():
    # Đổi chỗ đặt dữ liệu: sửa DATA_ROOT trong launcher, hoặc đặt biến môi trường
    # LIFEMATE_DATA_ROOT trước khi chạy cell cấu hình. Không thêm /train ở cuối.
    explicit = os.environ.get("LIFEMATE_DATA_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    if _on_kaggle():
        # Dataset đã Add Input vào Kaggle; /kaggle/input chỉ dùng để đọc.
        return Path("/kaggle/input/datasets/khangl123/zalo-ai/train+public_test")
    # Local Windows: dữ liệu BTC đã giải nén trên máy hiện tại.
    return Path("C:/zalo_ai/train+public_test")


def output_root(project_root=None):
    if _on_kaggle():
        # Kaggle ghi kết quả vào /kaggle/working, không ghi vào /kaggle/input.
        return Path("/kaggle/working/lifemate")
    # Local luôn ghi theo repo, kể cả khi mở notebook ở src/lifemate/notebooks/.
    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    return root / "output/lifemate"


def asr_output_dir(split, project_root=None):
    if not isinstance(split, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", split):
        raise ValueError("Tên split không hợp lệ")
    return output_root(project_root) / "asr" / split


def asr_prediction_file(split, project_root=None):
    # Path mặc định khi file bàn giao được tạo cùng máy/phiên chạy.
    # File từ notebook Kaggle khác: sửa ASR_FILE trong launcher tới file đã Add Input.
    return asr_output_dir(split, project_root) / "asr_predictions.jsonl"


def agent_output_dir(project_root=None):
    # Runner sẽ tạo thêm <run_id>/ để tách kết quả và traces của từng lần chạy.
    return output_root(project_root) / "agent"
