"""Dataset metadata, ASR handoff and split-aware ToolEnv package discovery."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import importlib
import json
import os
import sys

from lifemate.paths import asr_prediction_file
from .config import AgentInputError


def discover_data_root(search_root="/kaggle/input", explicit=None):
    """Find metadata only; do not traverse audio or environment directories."""
    if explicit:
        root = Path(explicit)
        if not (root / "train" / "train.jsonl").is_file():
            raise AgentInputError(f"Không tìm thấy train/train.jsonl tại {root}")
        return root
    roots = set()
    for folder, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d != "audio" and not d.endswith("_environment") and d != ".git"]
        if "train.jsonl" in files:
            roots.add(Path(folder).parent)
    if len(roots) != 1:
        raise AgentInputError(f"Tìm thấy {len(roots)} bộ dữ liệu; hãy đặt DATA_ROOT rõ ràng")
    return roots.pop()


def load_split_items(data_root, split):
    # DATA_ROOT/<split>/<split>.jsonl; không truyền DATA_ROOT đã có /train ở cuối.
    with (Path(data_root) / split / f"{split}.jsonl").open(encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream if line.strip()]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise AgentInputError("Split có ID trùng")
    return items


def load_catalog(data_root, split):
    # Catalog công cụ thuộc đúng split, nằm cạnh file metadata JSONL.
    with (Path(data_root) / split / f"{split}_tools_catalog.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def get_tool_env_factory(data_root, split):
    """Never silently reuse another split's zaic_tools module."""
    # Nạp code BTC từ DATA_ROOT/<split>/zaic_tools; chỉ ToolEnv đọc trạng thái env.
    package = (Path(data_root) / split / "zaic_tools").resolve()
    previous = sys.modules.get("zaic_tools")
    if previous is not None:
        origin = Path(previous.__file__).resolve().parent
        if origin != package:
            raise AgentInputError("Đã import zaic_tools từ split khác. Khởi động kernel mới khi đổi split.")
    else:
        sys.path.insert(0, str(package.parent))
    module = importlib.import_module("zaic_tools")
    if Path(module.__file__).resolve().parent != package:
        raise AgentInputError("Import nhầm zaic_tools; hãy khởi động kernel mới")
    return module.ToolEnv


def prepare_request(item):
    """Allowlist fields instead of forwarding the raw labeled training row."""
    history = []
    for turn in item.get("history", []):
        role = turn.get("role")
        if role not in ("user", "assistant"):
            raise AgentInputError(f"History role không hỗ trợ: {role}")
        allowed = ("role", "audio") if role == "user" else ("role", "text", "tool_calls", "tool_results")
        history.append({key: deepcopy(turn[key]) for key in allowed if key in turn})
    return {"id": item["id"], "context": deepcopy(item["context"]), "history": history}


def speech_from_train(item, split):
    if split != "train":
        raise AgentInputError("Transcript chuẩn chỉ được dùng ở chế độ thử nghiệm trên train")
    if "audio_text" not in item:
        raise AgentInputError("Mẫu không có transcript chuẩn")
    speech = {
        "id": item["id"], "current_text": item["audio_text"],
        "history_user_texts": [
            {"history_index": i, "text": turn["audio_text"]}
            for i, turn in enumerate(item.get("history", [])) if turn["role"] == "user"
        ],
    }
    validate_speech(prepare_request(item), speech)
    return speech


def validate_speech(request, speech):
    if request["id"] != speech.get("id"):
        raise AgentInputError("ID ASR không khớp mẫu")
    if not isinstance(speech.get("current_text"), str):
        raise AgentInputError("current_text phải là chuỗi")
    expected = {i for i, t in enumerate(request["history"]) if t["role"] == "user"}
    entries = speech.get("history_user_texts")
    if not isinstance(entries, list):
        raise AgentInputError("history_user_texts phải là list")
    seen = set()
    for turn in entries:
        index = turn.get("history_index")
        if type(index) is not int or index not in expected or index in seen:
            raise AgentInputError("history_index ASR sai, thừa hoặc trùng")
        if not isinstance(turn.get("text"), str):
            raise AgentInputError("Văn bản history phải là chuỗi")
        seen.add(index)
    if seen != expected:
        raise AgentInputError("Thiếu transcript ASR của một hoặc nhiều lượt history")


def load_speech_file(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["id"] in result:
                    raise AgentInputError("File ASR có ID trùng")
                result[row["id"]] = row
    return result


def load_asr(data_root, split, predictions_path=None):
    """Đọc prediction ASR đã xuất; thiếu ID/history thì dừng trước khi nạp Qwen."""
    from lifemate.asr.predictions import FileTranscriber

    # Chỉ đọc file bàn giao, không nạp/chạy model ASR trong luồng agent.
    # predictions_path cho phép đọc file chuyển từ Kaggle sang local hoặc ngược lại.
    path = Path(predictions_path) if predictions_path is not None else asr_prediction_file(split)
    return FileTranscriber(path, load_split_items(data_root, split))


def prepare_audio_input(item, data_root, split):
    """Chỉ chuyển đường dẫn audio cho ASR; tuyệt đối không chuyển label/gold text."""
    split_root = (Path(data_root) / split).resolve()

    def full_audio_path(relative):
        if not isinstance(relative, str) or not relative:
            raise AgentInputError("Mẫu thiếu đường dẫn audio")
        path = (split_root / relative).resolve()
        if not path.is_relative_to(split_root):
            raise AgentInputError("Audio nằm ngoài thư mục split")
        return str(path)

    return {
        "id": item["id"], "audio": full_audio_path(item["audio"]),
        "history_user_audio": [
            {"history_index": index, "audio": full_audio_path(turn.get("audio"))}
            for index, turn in enumerate(item.get("history", [])) if turn["role"] == "user"
        ],
    }
