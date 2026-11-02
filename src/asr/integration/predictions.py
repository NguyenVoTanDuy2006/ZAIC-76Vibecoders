"""ASR -> LifeMate handoff. Standard library only; no gold-text fallback."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from asr.common.manifest import user_audio_turns


class PredictionContractError(ValueError):
    pass


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_items(data_root, split):
    """data_root is the directory containing train/, public_test/, ..."""
    rows = read_jsonl(Path(data_root) / split / f"{split}.jsonl")
    ids = [row["id"] for row in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise PredictionContractError("Dataset rỗng hoặc có ID không hợp lệ/trùng")
    return rows


def build_audio_manifest(items, data_root, split):
    """Flatten current + user-history audio. Never forward labels/reference text.

    History IDs match the existing fine-tuning package: <id>_history_<index:03d>.
    history_index always refers to the original history, including assistant turns.
    """
    split_root = (Path(data_root) / split).resolve()
    manifest, seen, conversations = [], set(), set()
    for item in items:
        cid = item["id"]
        if not isinstance(cid, str) or not cid or cid in conversations:
            raise PredictionContractError("ID conversation không hợp lệ/trùng")
        conversations.add(cid)
        try:
            for _, sample_id, index, relative in user_audio_turns(item, strict_roles=True):
                if sample_id in seen or not isinstance(relative, str) or not relative:
                    raise PredictionContractError(f"Audio ID/path không hợp lệ: {sample_id}")
                raw = Path(relative)
                path = (split_root / raw).resolve()
                if raw.is_absolute() or not raw.parts or raw.parts[0] != "audio" or not path.is_relative_to(split_root):
                    raise PredictionContractError(f"Audio phải nằm trong audio/ của split: {sample_id}")
                if not path.is_file():
                    raise FileNotFoundError(f"Thiếu audio: {path}")
                manifest.append({"sample_id": sample_id, "conversation_id": cid,
                                 "history_index": index, "audio_path": str(path),
                                 "audio": path.relative_to(split_root).as_posix()})
                seen.add(sample_id)
        except ValueError as exc:
            if str(exc).startswith("Invalid history role:"):
                raise PredictionContractError(f"History role không hợp lệ: {cid}") from exc
            raise
    return manifest


def index_audio_predictions(predictions, manifest):
    """Accept baseline 'text' or fine-tuning 'prediction'; ignore gold 'reference'."""
    expected = {row["sample_id"] for row in manifest}
    indexed = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise PredictionContractError("Prediction phải là JSON object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected or sample_id in indexed:
            raise PredictionContractError(f"Prediction ID thừa/trùng: {sample_id}")
        text = row.get("text", row.get("prediction"))
        if not isinstance(text, str):
            raise PredictionContractError(f"Thiếu text/prediction: {sample_id}")
        if "text" in row and "prediction" in row and row["text"] != row["prediction"]:
            raise PredictionContractError(f"text và prediction mâu thuẫn: {sample_id}")
        indexed[sample_id] = text
    return indexed


def assemble_speech_rows(items, manifest, predictions):
    """One speech row per conversation; require every current/history ASR clip."""
    items = list(items)
    indexed = index_audio_predictions(predictions, manifest)
    missing = {row["sample_id"] for row in manifest} - set(indexed)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise PredictionContractError(f"Thiếu ASR của {len(missing)} audio (kể cả history): {preview}")
    rows = {item["id"]: {"id": item["id"], "current_text": None, "history_user_texts": []} for item in items}
    for clip in manifest:
        row = rows[clip["conversation_id"]]
        text = indexed[clip["sample_id"]]
        if clip["history_index"] is None:
            row["current_text"] = text
        else:
            row["history_user_texts"].append({"history_index": clip["history_index"], "text": text})
    result = [rows[item["id"]] for item in items]
    validate_speech_rows(result, items)
    return result


def validate_speech_rows(rows, items):
    """Reject missing/extra/duplicate conversation IDs and incomplete history."""
    expected = {item["id"]: item for item in items}
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "current_text", "history_user_texts"}:
            raise PredictionContractError("Dòng ASR phải có đúng id/current_text/history_user_texts")
        cid = row["id"]
        if not isinstance(cid, str) or cid not in expected or cid in indexed:
            raise PredictionContractError(f"ASR conversation ID sai/thừa/trùng: {cid}")
        if not isinstance(row["current_text"], str) or not isinstance(row["history_user_texts"], list):
            raise PredictionContractError(f"Sai kiểu text/history: {cid}")
        indices = {i for i, t in enumerate(expected[cid].get("history", [])) if t.get("role") == "user"}
        seen = set()
        for turn in row["history_user_texts"]:
            if not isinstance(turn, dict) or set(turn) != {"history_index", "text"}:
                raise PredictionContractError(f"Sai schema history ASR: {cid}")
            index = turn["history_index"]
            if type(index) is not int or index not in indices or index in seen or not isinstance(turn["text"], str):
                raise PredictionContractError(f"Sai/trùng history_index hoặc text: {cid}")
            seen.add(index)
        if seen != indices:
            raise PredictionContractError(f"Thiếu ASR history: {cid}")
        indexed[cid] = deepcopy(row)
    if set(indexed) != set(expected):
        raise PredictionContractError(f"ASR file thiếu {len(set(expected) - set(indexed))} conversation")
    return indexed


def write_speech_file(path, rows):
    """Exclusive create; caller assembles/validates all rows before publishing."""
    path = Path(path)
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return path


def convert_prediction_files(data_root, split, prediction_paths, output_path):
    """Convert existing friend's prediction files; latest-only files fail on history."""
    items = load_items(data_root, split)
    manifest = build_audio_manifest(items, data_root, split)
    predictions = [row for path in prediction_paths for row in read_jsonl(path)]
    rows = assemble_speech_rows(items, manifest, predictions)
    return write_speech_file(output_path, rows)


class FileTranscriber:
    """Callable accepted by the agent's ASR hook; no ASR/torch imports here."""
    def __init__(self, prediction_path, items):
        self.rows = validate_speech_rows(read_jsonl(prediction_path), items)

    def __call__(self, audio_input):
        cid = audio_input.get("id")
        if cid not in self.rows:
            raise PredictionContractError(f"Không có ASR cho ID {cid}")
        row = self.rows[cid]
        expected = {turn["history_index"] for turn in audio_input["history_user_audio"]}
        actual = {turn["history_index"] for turn in row["history_user_texts"]}
        if actual != expected:
            raise PredictionContractError(f"ASR history không khớp audio_input: {cid}")
        return deepcopy(row)
