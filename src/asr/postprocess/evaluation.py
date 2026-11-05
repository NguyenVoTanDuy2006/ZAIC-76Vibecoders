"""Corpus-level CER/WER for correction variants without semantic-model loading."""
from __future__ import annotations

from asr.common.text import normalize_text
from asr.finetuning.metrics import error_rates
from asr.common.metrics import edit_distance


def score_variant(source_rows: list[dict], prediction_rows: list[dict]) -> dict:
    expected = {row["sample_id"] for row in source_rows}
    actual = [row["sample_id"] for row in prediction_rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("incomplete or duplicate correction predictions")
    by_id = {row["sample_id"]: row["prediction"] for row in prediction_rows}
    if any(not isinstance(value, str) for value in by_id.values()):
        raise ValueError("correction prediction must be text")
    result = {}
    for group_name, group in (
        ("all", source_rows),
        ("latest", [row for row in source_rows if row["is_latest"]]),
    ):
        metrics = {"samples": len(group)}
        improved = worsened = changed = 0
        for row in group:
            reference = normalize_text(row["reference"]).split()
            original = normalize_text(row["prediction"]).split()
            candidate = normalize_text(by_id[row["sample_id"]]).split()
            before = edit_distance(reference, original)
            after = edit_distance(reference, candidate)
            improved += after < before
            worsened += after > before
            changed += candidate != original
        metrics.update({"changed_samples": changed, "improved_samples": improved,
                        "worsened_samples": worsened})
        for prefix in ("raw", "normalized"):
            pairs = [(row["reference"], by_id[row["sample_id"]]) for row in group]
            if prefix == "normalized":
                pairs = [(normalize_text(ref), normalize_text(hyp)) for ref, hyp in pairs]
            metrics.update({f"{prefix}_{key}": value for key, value in error_rates(pairs).items()})
        result[group_name] = metrics
    return result
