"""Kaggle orchestration: one AgentSession runner, durable outputs, replay and score."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import uuid

from lifemate.agent import AgentConfig, ModelLoadConfig, SYSTEM_PROMPT
from . import data
from .evaluate import evaluate
from .io import code_fingerprint, index_rows, read_json, read_rows, sha256, write_json, write_rows
from .runner import load_run_config, prepare_run, run
from .selection import create_selection
from .validate import load_traces, read_predictions, validate_prediction


def _run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def prepare_experiment(data_root, work_dir, *, count=10, sample_ids=None, speech_path=None,
                       oracle=True, agent_options=None, model_source=None, model_revision=None,
                       device="cuda:0", seed=2026, bundle=None):
    """Prepare train-derived validation and check inputs before model downloads.

    count=None selects the complete validation split. Explicit IDs must belong
    to that split. Gold lives only in prepared references; the runtime receives
    clean inputs plus oracle speech (opt-in) or the existing ASR handoff file.
    """
    import hashlib

    if type(oracle) is not bool:
        raise TypeError("oracle must be an explicit boolean")
    root, work = Path(data_root).resolve(), Path(work_dir).resolve()
    if not (root / "train/train.jsonl").is_file():
        raise FileNotFoundError(root / "train/train.jsonl")
    # Construct and validate the entire effective config, not a second set of defaults.
    config = AgentConfig(**(agent_options or {}))
    source = "Qwen/Qwen3-4B-Instruct-2507" if model_source is None else model_source
    model = ModelLoadConfig(model_source=str(source) if isinstance(source, Path) else source,
                            revision=model_revision, device=device)
    settings = {"model_source": model.model_source,
                "model_revision": model_revision, "device": device, "quantization": "fp16",
                "weights_fingerprint": None, "generation": asdict(config),
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
    if not oracle and (speech_path is None or not Path(speech_path).is_file()):
        raise FileNotFoundError("asr_text requires an existing current/history ASR predictions file")
    if bundle is None:
        bundle = work / "bundles" / (sha256(root / "train/train.jsonl")[:16] + f"_seed{seed}")
        if bundle.exists():
            data.verify(bundle, root)
        else:
            data.prepare(root, bundle, seed)
    else:
        bundle = Path(bundle).resolve()
    plan_dir = work / "plans" / _run_id()
    config_path, selection_path = plan_dir / "config.json", plan_dir / "selection.json"
    write_json(config_path, settings)
    load_run_config(config_path)
    selection = create_selection(bundle, "validation", selection_path,
        sample_ids=sample_ids, limit=count if sample_ids is None else None)
    prepared = prepare_run(bundle, "validation", root, speech_path, config_path=config_path,
        oracle=oracle, selection_path=selection_path, require_complete_speech=True)
    preflight = {"passed": True, "logical_split": "validation", "physical_split": "train",
                 "selected_samples": len(selection["ids"]), "speech_errors": prepared["speech_errors"],
                 "effective_generation": asdict(prepared["config"]),
                 "effective_model": asdict(prepared["model"]),
                 "code": code_fingerprint(), "runtime": prepared["runtime"].identity,
                 "selection_sha256": sha256(selection_path), "config_sha256": sha256(config_path),
                 "bundle_integrity_sha256": sha256(bundle / "integrity.json"),
                 "speech_sha256": sha256(speech_path) if not oracle else None,
                 "mode": "oracle_text" if oracle else "asr_text"}
    write_json(plan_dir / "preflight.json", preflight)
    plan = {"data_root": str(root), "bundle": str(bundle), "config": str(config_path),
            "selection": str(selection_path), "output_dir": str(plan_dir), "oracle": oracle,
            "speech_path": str(Path(speech_path).resolve()) if speech_path is not None else None,
            "selected_ids": selection["ids"], "preflight": preflight}
    write_json(plan_dir / "plan.json", plan)
    return plan


def run_experiment(plan, *, backend=None):
    """Execute once, then validate and score the SAME IDs even when some fail.

    Calling again makes a new run with new inference envs, while an injected
    Qwen backend can retain its weights. No sample is retried within a run.
    """
    expected = plan["preflight"]
    if type(plan["oracle"]) is not bool or expected["mode"] != ("oracle_text" if plan["oracle"] else "asr_text"):
        raise ValueError("ASR/oracle mode changed after preflight; prepare a fresh plan")
    if sha256(Path(plan["bundle"]) / "integrity.json") != expected["bundle_integrity_sha256"]:
        raise ValueError("Prepared bundle changed after preflight; prepare a fresh plan")
    for path, key in ((plan["selection"], "selection_sha256"), (plan["config"], "config_sha256")):
        if sha256(path) != expected[key]:
            raise ValueError("Experiment inputs changed after preflight; prepare a fresh plan")
    if code_fingerprint()["sha256"] != expected["code"]["sha256"]:
        raise ValueError("Source changed after preflight; prepare a fresh plan")
    if not plan["oracle"] and sha256(plan["speech_path"]) != expected["speech_sha256"]:
        raise ValueError("ASR speech changed after preflight; prepare a fresh plan")
    prepared = prepare_run(plan["bundle"], "validation", plan["data_root"], plan["speech_path"],
        config_path=plan["config"], oracle=plan["oracle"], selection_path=plan["selection"],
        require_complete_speech=True)
    if prepared["runtime"].identity != expected["runtime"]:
        raise ValueError("ToolEnv runtime/catalog changed after preflight; prepare a fresh plan")
    destination = Path(plan["output_dir"]) / ("run_" + _run_id())
    destination.mkdir(parents=True)
    print("Run output:", destination)
    write_json(destination / "selection.json", read_json(plan["selection"]))
    report = run(plan["bundle"], "validation", plan["data_root"], plan["speech_path"], destination / "agent",
        config_path=plan["config"], oracle=plan["oracle"], selection_path=plan["selection"], backend=backend)
    runtime, items = prepared["runtime"], prepared["items"]
    ids = [row["id"] for row in items]
    traces = load_traces(report["traces"])
    predictions, file_issues = read_predictions(report["predictions"], ids)
    checks = [validate_prediction(row["id"], predictions.get(row["id"]), row, runtime, traces) for row in items]
    write_rows(destination / "validation/validation.jsonl", checks)
    validation = {"samples": len(items), "file_issues": file_issues,
                  "submission_valid_samples": sum(c["submission_valid"] for c in checks),
                  "strict_valid_samples": sum(c["valid"] and not c["schema_errors"] for c in checks),
                  "schema_error_calls": sum(len(c["schema_errors"]) for c in checks),
                  "trace_coverage_checked": True}
    write_json(destination / "validation/validation_summary.json", validation)
    # Evaluation-only reference replay happens after inference; no gold call
    # or final_state is passed into AgentSession or its backend.
    references = index_rows(read_rows(Path(plan["bundle"]) / "validation/references.jsonl"))
    gold = [runtime.reference(references[sid]) for sid in ids]
    write_rows(destination / "evaluation/gold_replay.jsonl", gold)
    metrics = evaluate(items, gold, report["predictions"], runtime, destination / "evaluation", traces=traces)
    run_metadata = read_json(destination / "agent/run.json")
    summary = {"selected_samples": len(ids), "selected_ids": ids,
               "prediction_rows": len(predictions), "trace_rows": len(traces),
               "statuses": dict(Counter(t["status"] for t in traces.values())),
               "local_estimated_score_100": metrics["score"], "score_label": metrics["score_label"],
               "scoring_convention": metrics["scoring_convention"],
               "hash_accuracy_nonempty": metrics["hash_accuracy_nonempty"],
               "runtime_health": metrics["runtime_health"], "validation": validation,
               "mode": expected["mode"], "asr_inference_in_this_run": False,
               "backend_type": type(backend).__name__ if backend is not None else "HFBackend",
               "loaded_model": run_metadata["loaded_model"],
               "source_sha256": expected["code"]["sha256"],
               "selection_sha256": expected["selection_sha256"],
               "config_sha256": expected["config_sha256"],
               "output_dir": str(destination)}
    summary_path = destination / "summary.json"
    write_json(summary_path, summary)
    return {**report, "summary": summary, "summary_path": str(summary_path),
            "evaluation_dir": str(destination / "evaluation"), "output_dir": str(destination)}
