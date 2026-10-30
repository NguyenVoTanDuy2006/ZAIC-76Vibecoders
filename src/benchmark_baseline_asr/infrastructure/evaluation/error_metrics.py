from __future__ import annotations

from benchmark_baseline_asr.domain.entities import ErrorRates


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for ref_index, ref_unit in enumerate(reference, 1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_unit in enumerate(hypothesis, 1):
            if ref_unit == hyp_unit:
                current.append(previous[hyp_index - 1])
                continue
            sub = previous[hyp_index - 1]
            delete = previous[hyp_index]
            insert = current[hyp_index - 1]
            options = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            current.append(min(options, key=lambda counts: (counts[0], counts[1], counts[2], counts[3])))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


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
