"""Strict, standard-library artifact IO and reproducibility helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def loads(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{index}: blank JSONL line")
        try:
            row = loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
        except ValueError as error:
            raise ValueError(f"{path}:{index}: {error}") from error
        rows.append(row)
    return rows


def index_rows(rows):
    indexed = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in indexed:
            raise ValueError(f"Missing/invalid/duplicate ID: {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parents[1]
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def artifact_hashes(folder):
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name not in ("audit.jsonl",)}


def call_error(call):
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return "call must contain exactly name and arguments"
    if not isinstance(call["name"], str) or not call["name"]:
        return "call name must be a nonempty string"
    if not isinstance(call["arguments"], dict):
        return "call arguments must be an object"
    return None
