"""Local call convention, explicitly versioned until checked against BTC scorer."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import unicodedata


@dataclass(frozen=True)
class MatchConvention:
    name: str = "local-nfc-whitespace-multiset-strict-v1"
    fuzzy_threshold: float = 0.6
    tokenizer: str = "whitespace"

    def __post_init__(self):
        if self.tokenizer != "whitespace" or not 0 <= self.fuzzy_threshold <= 1:
            raise ValueError("Unsupported tokenizer/threshold")

    def metadata(self):
        return {**asdict(self), "threshold_comparison": ">", "official_scorer_verified": False,
                "exact": "typed recursive JSON equality; identical key sets; array order preserved",
                "fuzzy": "NFC + lowercase + whitespace split; multiset token F1; accents retained"}


def token_f1(left, right):
    if not isinstance(left, str) or not isinstance(right, str):
        return 0.0
    def tokens(text):
        return Counter(unicodedata.normalize("NFC", text.lower()).split())
    a, b = tokens(left), tokens(right)
    size = sum(a.values()) + sum(b.values())
    return 2 * sum((a & b).values()) / size if size else 1.0


def exact(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def argument_differences(left, right, schema, convention=None, path="arguments"):
    convention = convention or MatchConvention()
    if schema.get("x-match") == "fuzzy":
        score = token_f1(left, right)
        if isinstance(left, str) and isinstance(right, str) and score > convention.fuzzy_threshold:
            return []
        return [{"path": path, "kind": "fuzzy_mismatch", "submitted": left, "gold": right,
                 "token_f1": score, "threshold": convention.fuzzy_threshold}]
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(left.keys() | right.keys()):
            child_path = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append({"path": child_path,
                                    "kind": "missing_argument" if key not in left else "extra_argument",
                                    "submitted": left.get(key), "gold": right.get(key)})
            else:
                differences.extend(argument_differences(left[key], right[key],
                    schema.get("properties", {}).get(key, {}), convention, child_path))
        return differences
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [d for i, (a, b) in enumerate(zip(left, right))
                for d in argument_differences(a, b, schema.get("items", {}), convention, f"{path}[{i}]")]
    return [] if exact(left, right) else [{"path": path, "kind": "exact_mismatch",
                                         "submitted": left, "gold": right}]


def maximum_matching(left, right, compatible):
    """Maximum-cardinality bipartite matching via augmenting paths, never greedy."""
    edges = [[j for j, b in enumerate(right) if compatible(a, b)] for a in left]
    owners = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j], seen):
                owners[j] = i
                return True
        return False
    for i in range(len(left)):
        augment(i, set())
    return sorted((i, j) for j, i in owners.items())


def match_calls(submitted, gold, tools, convention=None):
    def compatible(a, b):
        return a["name"] == b["name"] and not argument_differences(
            a["arguments"], b["arguments"], tools.get(a["name"], {}).get("parameters", {}), convention)
    return maximum_matching(submitted, gold, compatible)


def prf(matched, submitted, gold):
    return {"precision": matched / submitted if submitted else (1.0 if not gold else 0.0),
            "recall": matched / gold if gold else (1.0 if not submitted else 0.0),
            "f1": 2 * matched / (submitted + gold) if submitted + gold else 1.0}
