"""Evaluator-only official ToolEnv replay; never reads environment files itself."""
from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path

from lifemate.agent import get_tool_env_factory, load_catalog
from .io import call_error, code_fingerprint, sha256, write_json, write_rows


class OfficialRuntime:
    def __init__(self, data_root, split="train"):
        self.root, self.split = Path(data_root).resolve(), split
        self.factory = get_tool_env_factory(self.root, split)
        module = importlib.import_module("zaic_tools")
        self.canon = module.canon
        self.catalog = load_catalog(self.root, split)
        self.tools = module.catalog_tools(self.catalog)
        self.schema_error = module.schema_error
        self.identity = {"physical_split": split,
            "catalog_sha256": sha256(self.root / split / f"{split}_tools_catalog.json"),
            "runtime": code_fingerprint(self.root / split / "zaic_tools")}

    def make_env(self, sample_id):
        return self.factory(self.split, sample_id, root=self.root)

    def replay(self, sample_id, calls):
        env = self.make_env(sample_id)  # isolated evaluator replay, never agent inference
        results = [env.call(call["name"], deepcopy(call["arguments"])) for call in calls]
        return {"tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
                "state": self.canon(env.get_state()), "results": results}

    def reference(self, reference):
        label, sid = reference["label"], reference["id"]
        for call in label["tool_calls"]:
            if call_error(call):
                raise ValueError(f"{sid}: invalid reference call: {call_error(call)}")
        env = self.make_env(sid)
        initial = env.get_state()
        results = [env.call(c["name"], deepcopy(c["arguments"])) for c in label["tool_calls"]]
        # final_state is partial by APP. Merge into initial full state, then use BTC canon.
        expected = deepcopy(initial)
        expected.update(deepcopy(label["final_state"]))
        actual = self.canon(env.get_state())
        expected = self.canon(expected)
        differences = state_differences(actual, expected)
        return {"id": sid, "group_id": reference.get("group_id"), "tool_calls": label["tool_calls"],
            "state_hash": env.get_state_hash(), "canonical_state": actual,
            "reference_valid": not differences and not any(r.get("ok") is False for r in results),
            "reference_state_differences": differences, "reference_results": results,
            "reference_tool_results_equal": results == label.get("tool_results")}


def state_differences(submitted, expected, path="state"):
    if isinstance(submitted, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(submitted.keys() | expected.keys()):
            if key not in submitted or key not in expected:
                result.append({"path": f"{path}.{key}", "submitted": submitted.get(key), "gold": expected.get(key)})
            else:
                result.extend(state_differences(submitted[key], expected[key], f"{path}.{key}"))
        return result
    if isinstance(submitted, list) and isinstance(expected, list) and len(submitted) == len(expected):
        return [d for i, (a, b) in enumerate(zip(submitted, expected)) for d in state_differences(a, b, f"{path}[{i}]")]
    return [] if submitted == expected and isinstance(submitted, bool) == isinstance(expected, bool) else [
        {"path": path, "submitted": submitted, "gold": expected}]


def build_gold(references, runtime, output):
    rows = [runtime.reference(row) for row in references]
    errors = [r["id"] for r in rows if not r["reference_valid"]]
    output = Path(output)
    write_rows(output / "gold.jsonl", rows)
    write_json(output / "provenance.json", {**runtime.identity, "samples": len(rows), "invalid_references": errors})
    if errors:
        raise ValueError(f"Reference replay disagrees with label.final_state/schema for {len(errors)} samples; see {output}")
    return rows
