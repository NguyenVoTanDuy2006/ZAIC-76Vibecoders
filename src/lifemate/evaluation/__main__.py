"""python -m lifemate.evaluation --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare
from .data import access_split, freeze, prepare, verify
from .evaluate import evaluate
from .io import index_rows, read_json, read_rows, sha256, write_json, write_rows
from .matching import MatchConvention
from .runner import run
from .runtime import OfficialRuntime, build_gold
from .validate import load_traces, read_predictions, validate_prediction
from .selection import select_rows


def parser():
    result = argparse.ArgumentParser(description="Validate/evaluate LifeMate agents; ASR metrics remain in the team's existing pipeline")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "build-gold", "run", "validate", "evaluate", "freeze", "compare"):
        p = commands.add_parser(name)
        if name in ("prepare", "verify", "build-gold", "run", "validate", "evaluate"):
            p.add_argument("--data-root", required=True, type=Path, help="Parent of train/ and public_test/")
        if name not in ("prepare", "compare"):
            p.add_argument("--bundle", required=True, type=Path)
        if name not in ("verify",):
            p.add_argument("--output", required=True, type=Path, help="Fresh output path; existing artifacts are not overwritten")
        if name in ("build-gold", "run", "validate", "evaluate"):
            p.add_argument("--split", choices=("validation", "local_test"), default="validation")
            p.add_argument("--allow-local-test", action="store_true")
            p.add_argument("--frozen", type=Path)
            p.add_argument("--selection", type=Path, help="Immutable subset manifest shared with inference")
        if name in ("validate", "evaluate"):
            p.add_argument("--predictions", required=True, type=Path)
            p.add_argument("--traces", type=Path, help="results.jsonl or traces directory; autodetected next to predictions")
        if name == "evaluate":
            p.add_argument("--fuzzy-threshold", type=float, default=.6, help="Local convention override, strictly greater than threshold")
        if name == "prepare":
            p.add_argument("--seed", type=int, default=2026)
        if name in ("run", "freeze"):
            p.add_argument("--config", type=Path, required=name == "freeze")
        if name == "run":
            p.add_argument("--speech", type=Path, help="Existing ASR current/history predictions; no ASR model runs here")
            p.add_argument("--oracle-text", action="store_true", help="Gold text diagnostic on validation only")
        if name == "compare":
            p.add_argument("--left", required=True, type=Path, help="Before samples.jsonl")
            p.add_argument("--right", required=True, type=Path, help="After samples.jsonl")
            p.add_argument("--iterations", type=int, default=2000)
            p.add_argument("--seed", type=int, default=2026)
    return result


def _selected(args):
    folder = access_split(args.bundle, args.split, allow_local_test=args.allow_local_test,
                          frozen=args.frozen, purpose=args.command)
    if args.split == "local_test" and args.command == "evaluate" and args.fuzzy_threshold != .6:
        raise ValueError("Local test uses the fixed BTC fuzzy threshold 0.6; tune conventions only on validation")
    manifest = read_json(folder / "manifest.json")
    items = read_rows(folder / "inputs.jsonl")
    if [r["id"] for r in items] != manifest["ids"]:
        raise ValueError("Input IDs do not match manifest")
    runtime = OfficialRuntime(args.data_root, manifest["physical_split"])
    metadata = read_json(args.bundle / "metadata.json")
    if metadata["catalog_sha256"] != runtime.identity["catalog_sha256"] or metadata["source_sha256"] != sha256(
            args.data_root / "train/train.jsonl"):
        raise ValueError("Data/catalog changed since prepare")
    integrity = read_json(args.bundle / "integrity.json")
    for name in ("manifest.json", "inputs.jsonl", "references.jsonl"):
        relative = f"{args.split}/{name}"
        if sha256(folder / name) != integrity[relative]:
            raise ValueError(f"Prepared artifact changed: {relative}")
    items = select_rows(items, folder / "manifest.json", args.selection)
    return folder, items, runtime


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.data_root, args.output, args.seed)
            print(json.dumps({"splits": result["splits"], "groups": result["groups"], "source_sha256": result["source_sha256"]}, ensure_ascii=False))
        elif args.command == "verify":
            print(json.dumps(verify(args.bundle, args.data_root)))
        elif args.command == "freeze":
            freeze(args.bundle, args.config, args.output)
            print(str(args.output))
        elif args.command == "compare":
            result = compare(args.left, args.right, args.output, args.iterations, args.seed)
            print(json.dumps({k: v for k, v in result.items() if k != "samples_detail"}))
        elif args.command == "run":
            print(json.dumps(run(args.bundle, args.split, args.data_root, args.speech, args.output,
                config_path=args.config, allow_local_test=args.allow_local_test,
                frozen=args.frozen, oracle=args.oracle_text, selection_path=args.selection)))
        else:
            folder, items, runtime = _selected(args)
            if args.command == "build-gold":
                references = index_rows(read_rows(folder / "references.jsonl"))
                gold = build_gold([references[row["id"]] for row in items], runtime, args.output)
                print(json.dumps({"verified_gold": len(gold)}))
                return 0
            trace_path = args.traces
            if trace_path is None:
                for candidate in (args.predictions.parent / "results.jsonl", args.predictions.parent / "traces"):
                    if candidate.exists():
                        trace_path = candidate
                        break
            traces = load_traces(trace_path)
            if args.command == "validate":
                predictions, issues = read_predictions(args.predictions, [r["id"] for r in items])
                checks = [validate_prediction(row["id"], predictions.get(row["id"]), row, runtime, traces) for row in items]
                write_rows(args.output / "validation.jsonl", checks)
                valid = not issues and all(c["valid"] and not c["schema_errors"] for c in checks)
                summary = {"valid": valid, "samples": len(checks), "file_issues": issues,
                    "invalid_predictions": sum(not c["valid"] for c in checks),
                    "schema_error_calls": sum(len(c["schema_errors"]) for c in checks),
                    "trace_coverage_checked": traces is not None, "runtime": runtime.identity}
                write_json(args.output / "validation_summary.json", summary)
                print(json.dumps({k: v for k, v in summary.items() if k != "runtime"}))
                return 0 if valid else 1
            references = index_rows(read_rows(folder / "references.jsonl"))
            gold = [runtime.reference(references[row["id"]]) for row in items]
            result = evaluate(items, gold, args.predictions, runtime, args.output, traces=traces,
                              convention=MatchConvention(fuzzy_threshold=args.fuzzy_threshold))
            write_json(args.output / "selection.json", {"split": args.split,
                "ids": [row["id"] for row in items],
                "selection_sha256": sha256(args.selection) if args.selection else None,
                "manifest_sha256": sha256(folder / "manifest.json"), "references_sha256": sha256(folder / "references.jsonl")})
            print(json.dumps({"score": result["score"], "label": result["score_label"],
                              "counts": result["counts"], "file_issues": result["prediction_file_issues"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
