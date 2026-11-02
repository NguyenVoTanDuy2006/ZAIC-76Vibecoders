"""Levenshtein edit counts shared by ASR evaluation paths."""


def edit_counts(reference, hypothesis):
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
            current.append(min(options, key=lambda counts: counts))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def edit_distance(reference, hypothesis):
    return sum(edit_counts(reference, hypothesis))
