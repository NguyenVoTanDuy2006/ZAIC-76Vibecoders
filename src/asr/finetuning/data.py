from __future__ import annotations

import hashlib
import json
import math
import random
import re
import wave
from pathlib import Path

from .artifacts import fingerprint, write_json, write_jsonl
from asr.common.manifest import user_audio_turns


def prepare_records(root, manifest="train.jsonl", include_history=True):
    root = Path(root).expanduser().resolve()
    records, seen = [], set()
    with (root / manifest).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("id")
            if (
                not isinstance(cid, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cid)
                or cid in seen
            ):
                raise ValueError(f"invalid/duplicate conversation ID: {cid}")
            seen.add(cid)
            turns = user_audio_turns(row) if include_history else [(True, cid, None, row.get("audio"))]
            for latest, sample_id, history_index, relative in turns:
                turn = row if history_index is None else row["history"][history_index]
                reference = turn.get("audio_text")
                if (
                    not isinstance(reference, str)
                    or not reference.strip()
                    or not isinstance(relative, str)
                ):
                    raise ValueError(f"missing transcript/audio: {sample_id}")
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or path == root:
                    raise ValueError(f"audio escapes root: {relative}")
                with wave.open(str(path), "rb") as wav:
                    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                        1,
                        2,
                        16000,
                    ) or wav.getnframes() == 0:
                        raise ValueError(
                            f"expected nonempty mono PCM16 16kHz WAV: {path}"
                        )
                    pcm = wav.readframes(wav.getnframes())
                    if len(pcm) != wav.getnframes() * 2:
                        raise ValueError(f"truncated WAV: {path}")
                    duration = wav.getnframes() / 16000
                records.append(
                    {
                        "sample_id": sample_id,
                        "conversation_id": cid,
                        "is_latest": latest,
                        "audio": str(path.relative_to(root)),
                        "reference": reference,
                        "duration_seconds": duration,
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    }
                )
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("empty or duplicate sample IDs")
    return records


def split_records(records, seed=1234, validation_fraction=0.1, test_fraction=0.1):
    ids = sorted({r["conversation_id"] for r in records})
    random.Random(seed).shuffle(ids)
    nv, nt = (
        math.ceil(len(ids) * validation_fraction),
        math.ceil(len(ids) * test_fraction),
    )
    if nv < 1 or nt < 1 or nv + nt >= len(ids):
        raise ValueError("not enough conversations for three nonempty splits")
    test, validation = set(ids[:nt]), set(ids[nt : nt + nv])
    result = {"train": [], "validation": [], "test": []}
    hashes = {}
    for row in records:
        key = (
            "test"
            if row["conversation_id"] in test
            else "validation"
            if row["conversation_id"] in validation
            else "train"
        )
        old = hashes.setdefault(row["pcm_sha256"], key)
        if old != key:
            raise ValueError(f"duplicate PCM across splits: {row['sample_id']}")
        result[key].append(row)
    return result


def prepare_dataset(config, destination):
    records = prepare_records(
        config["dataset"]["root"],
        config["dataset"]["manifest"],
        config["dataset"]["include_history"],
    )
    splits = split_records(records, **config["split"])
    identity = fingerprint(splits)
    destination = Path(destination)
    marker = destination / "dataset.json"
    if marker.is_file() and json.loads(marker.read_text())["fingerprint"] != identity:
        raise ValueError("dataset/split changed; use a new run_id")
    summary = {
        "fingerprint": identity,
        "splits": {
            k: {
                "clips": len(v),
                "latest": sum(r["is_latest"] for r in v),
                "hours": sum(r["duration_seconds"] for r in v) / 3600,
            }
            for k, v in splits.items()
        },
    }
    for key, rows in splits.items():
        write_jsonl(destination / "manifests" / f"{key}.jsonl", rows)
    write_json(marker, summary)
    return splits, summary


class AudioDataset:
    def __init__(self, rows, root):
        self.rows, self.root = rows, Path(root).expanduser().resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import soundfile as sf

        row = self.rows[index]
        waveform, sr = sf.read(self.root / row["audio"], dtype="float32")
        if sr != 16000 or waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError(f"invalid waveform: {row['sample_id']}")
        return {"record": row, "waveform": waveform}


def identity_collate(items):
    # Keep the HF processor/tokenizer in the training process, not loader workers.
    return items
