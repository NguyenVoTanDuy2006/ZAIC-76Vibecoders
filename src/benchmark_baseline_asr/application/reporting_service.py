from __future__ import annotations

import csv
from pathlib import Path

from benchmark_baseline_asr.infrastructure.storage.artifacts import read_json, write_json


def build_comparison(run_dir: str | Path, model_keys: list[str]) -> list[dict]:
    root = Path(run_dir)
    rows = []
    for key in model_keys:
        summary = read_json(root / key / "summary.json")
        rows.append({"model": key, **summary})
    write_json(root / "comparison.json", rows)
    _write_csv(root / "comparison.csv", rows)
    _write_markdown(root / "comparison.md", rows)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    columns = ["model", "raw_cer", "raw_wer", "normalized_cer", "normalized_wer", "coverage", "rtf_sum_latency"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
