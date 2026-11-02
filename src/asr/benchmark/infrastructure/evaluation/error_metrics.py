from __future__ import annotations

from asr.benchmark.domain.entities import ErrorRates
from asr.common.metrics import edit_counts as _edit_counts


def corpus_error_rates(pairs: list[tuple[str, str]], *, units: str = "character") -> ErrorRates:
    if units not in {"character", "word"}:
        raise ValueError("units must be 'character' or 'word'")
    sub = delete = insert = denominator = 0
    for reference, hypothesis in pairs:
        ref = list(reference) if units == "character" else reference.split()
        hyp = list(hypothesis) if units == "character" else hypothesis.split()
        a, b, c = _edit_counts(ref, hyp)
        sub += a
        delete += b
        insert += c
        denominator += len(ref)
    return ErrorRates(sub, delete, insert, denominator)
