"""Paired group bootstrap of agent scores on the same validation samples."""
from __future__ import annotations

from collections import defaultdict
import random

from .evaluate import _percentile
from .io import index_rows, read_rows, write_json


def compare(left_path, right_path, output, iterations=2000, seed=2026):
    left, right = index_rows(read_rows(left_path)), index_rows(read_rows(right_path))
    if not left or left.keys() != right.keys():
        raise ValueError("Comparison requires identical nonempty ID sets")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be positive")
    groups = defaultdict(list)
    for sid in left:
        if left[sid].get("group_id") != right[sid].get("group_id"):
            raise ValueError(f"Group mismatch: {sid}")
        groups[left[sid].get("group_id") or sid].append((sid, right[sid]["score"] - left[sid]["score"]))
    rng, keys, boot = random.Random(seed), list(groups), []
    for _ in range(iterations):
        sampled = [value for key in rng.choices(keys, k=len(keys)) for _, value in groups[key]]
        boot.append(100 * sum(sampled) / len(sampled))
    changes = [{"id": sid, "before": left[sid]["score"], "after": right[sid]["score"],
                "delta": right[sid]["score"] - left[sid]["score"]} for sid in left]
    result = {"samples": len(left), "groups": len(groups), "seed": seed, "iterations": iterations,
        "delta_points": 100 * sum(r["delta"] for r in changes) / len(changes),
        "ci95_points": [_percentile(boot, .025), _percentile(boot, .975)],
        "improved": sum(r["delta"] > 0 for r in changes), "worsened": sum(r["delta"] < 0 for r in changes),
        "unchanged": sum(r["delta"] == 0 for r in changes), "samples_detail": changes,
        "interpretation": "Paired group bootstrap; a score gap alone does not establish ASR or agent causality"}
    write_json(output, result)
    return result
