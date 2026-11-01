from __future__ import annotations

import math
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(
        "".join(
            " " if unicodedata.category(c).startswith("P") else c for c in text
        ).split()
    )


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(
                min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def error_rates(pairs):
    word_edits = char_edits = words = chars = 0
    for ref, hyp in pairs:
        word_edits += edit_distance(ref.split(), hyp.split())
        char_edits += edit_distance(ref, hyp)
        words += len(ref.split())
        chars += len(ref)
    return {
        "wer": word_edits / words if words else (0.0 if not word_edits else None),
        "cer": char_edits / chars if chars else (0.0 if not char_edits else None),
        "word_edits": word_edits,
        "reference_words": words,
        "char_edits": char_edits,
        "reference_chars": chars,
    }


def is_better(candidate, previous):
    if previous is None:
        return True
    a, b = candidate["latest"], previous["latest"]
    diff = a["normalized_bertscore_f1"] - b["normalized_bertscore_f1"]
    if abs(diff) > 1e-6:
        return diff > 0
    return (a["normalized_wer"], a["normalized_cer"], candidate["epoch"]) < (
        b["normalized_wer"],
        b["normalized_cer"],
        previous["epoch"],
    )


def summarize_predictions(rows, expected):
    actual = [r["sample_id"] for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("incomplete or duplicate evaluation coverage")
    result = {}
    for name, group in [("all", rows), ("latest", [r for r in rows if r["is_latest"]])]:
        if not group:
            raise ValueError(f"empty evaluation subset: {name}")
        metrics = {"samples": len(group), "coverage": 1.0}
        for prefix in ["raw", "normalized"]:
            pairs = [
                (r["reference"], r["prediction"])
                if prefix == "raw"
                else (normalize(r["reference"]), normalize(r["prediction"]))
                for r in group
            ]
            metrics.update({f"{prefix}_{k}": v for k, v in error_rates(pairs).items()})
            scores = [r[f"{prefix}_bertscore_f1"] for r in group]
            if not all(math.isfinite(s) for s in scores):
                raise ValueError("nonfinite semantic metric")
            metrics[f"{prefix}_bertscore_f1"] = sum(scores) / len(scores)
        seconds = sum(r["latency_seconds"] for r in group)
        metrics["latency_mean_seconds"] = seconds / len(group)
        metrics["rtf_sum_latency"] = seconds / sum(r["duration_seconds"] for r in group)
        result[name] = metrics
    return result
