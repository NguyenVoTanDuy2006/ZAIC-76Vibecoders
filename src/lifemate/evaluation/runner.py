"""Agent-only validation runner: cached ASR text -> existing Qwen loop -> ToolEnv."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from lifemate.agent import (AgentConfig, AgentSession, ModelLoadConfig, SYSTEM_PROMPT,
    load_qwen_backend, load_toolenv, select_available_tools, validate_speech)
from .data import access_split
from .io import code_fingerprint, index_rows, read_json, read_rows, sha256, write_json
from .runtime import OfficialRuntime
from .selection import select_rows


def load_run_config(config_path=None):
    """Fail on incompatible settings before touching weights or creating sessions."""
    settings = read_json(config_path) if config_path else {}
    config = AgentConfig(**settings.get("generation", {}))
    model = ModelLoadConfig(model_source=settings.get("model_source", "Qwen/Qwen3-4B-Instruct-2507"),
        revision=settings.get("model_revision"), device=settings.get("device", "cuda:0"), local_files_only=True)
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if settings.get("prompt_sha256", prompt_hash) != prompt_hash:
        raise ValueError("Configured prompt hash does not match SYSTEM_PROMPT")
    if settings.get("quantization", "fp16") != "fp16":
        raise ValueError("Current Qwen backend implements fp16")
    return settings, config, model, prompt_hash


def prepare_run(bundle, split, data_root, speech_path=None, *, config_path=None,
                allow_local_test=False, frozen=None, oracle=False, selection_path=None,
                require_complete_speech=False):
    """Standard-library preflight, with no model generation or ToolEnv session."""
    if oracle and split != "validation":
        raise ValueError("Oracle text is diagnostic-only on validation")
    settings, config, model, prompt_hash = load_run_config(config_path)
    folder = access_split(bundle, split, allow_local_test=allow_local_test, frozen=frozen, purpose="run_agent")
    manifest = read_json(folder / "manifest.json")
    items = read_rows(folder / "inputs.jsonl")
    if [r["id"] for r in items] != manifest["ids"]:
        raise ValueError("Manifest/input ID mismatch")
    integrity = read_json(Path(bundle) / "integrity.json")
    for name in ("manifest.json", "inputs.jsonl", "references.jsonl"):
        if sha256(folder / name) != integrity[f"{split}/{name}"]:
            raise ValueError(f"Prepared artifact changed: {split}/{name}")
    items = select_rows(items, folder / "manifest.json", selection_path)
    runtime = OfficialRuntime(data_root, manifest["physical_split"])
    metadata = read_json(Path(bundle) / "metadata.json")
    if runtime.identity["catalog_sha256"] != metadata["catalog_sha256"] or sha256(
            Path(data_root) / "train/train.jsonl") != metadata["source_sha256"]:
        raise ValueError("Source/catalog differs from prepared bundle")
    if oracle:
        speech = {r["id"]: r["oracle_speech"] for r in read_rows(folder / "references.jsonl")}
    else:
        if speech_path is None:
            raise ValueError("Supply --speech: output of the team's existing ASR")
        speech = index_rows(read_rows(speech_path))
        # A full physical train ASR file is permitted; select only manifest IDs.
    if split == "local_test":
        snapshot = read_json(frozen)
        if config_path is None or sha256(config_path) != snapshot["config_sha256"]:
            raise ValueError("Local test must use the frozen config file unchanged")
        if settings.get("quantization") != "fp16":
            raise ValueError("Current Qwen backend implements fp16; frozen quantization must be fp16")
    speech_errors = []
    for row in items:
        select_available_tools(row, runtime.catalog)
        try:
            if row["id"] not in speech:
                raise ValueError("Missing ASR handoff row")
            validate_speech(row, speech[row["id"]])
        except ValueError as error:
            speech_errors.append({"id": row["id"], "error": str(error)})
    if require_complete_speech and speech_errors:
        raise ValueError(f"Incomplete current/history speech for {len(speech_errors)} selected samples: "
                         + ", ".join(row["id"] for row in speech_errors[:10]))
    return {"items": items, "speech": speech, "runtime": runtime, "settings": settings,
            "config": config, "model": model, "prompt_hash": prompt_hash,
            "manifest": manifest, "speech_errors": speech_errors}


def run(bundle, split, data_root, speech_path, output, *, config_path=None,
        allow_local_test=False, frozen=None, oracle=False, backend=None, selection_path=None):
    prepared = prepare_run(bundle, split, data_root, speech_path, config_path=config_path,
        allow_local_test=allow_local_test, frozen=frozen, oracle=oracle, selection_path=selection_path)
    items, speech, runtime = prepared["items"], prepared["speech"], prepared["runtime"]
    settings, config, model = prepared["settings"], prepared["config"], prepared["model"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if backend is None:
        backend = load_qwen_backend(model, config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"logical_split": split, "physical_split": prepared["manifest"]["physical_split"],
        "mode": "oracle_text_diagnostic" if oracle else "asr_text", "ids": [r["id"] for r in items],
        "selection_sha256": sha256(selection_path) if selection_path else None,
        "speech_sha256": None if oracle else sha256(speech_path), "config": settings,
        "effective_generation": asdict(config), "effective_model": asdict(model),
        "prompt_sha256": prepared["prompt_hash"], "code": code_fingerprint(), "runtime": runtime.identity,
        "loaded_model": {"name_or_path": getattr(getattr(backend, "model", None), "name_or_path", None),
                         "resolved_revision": getattr(getattr(getattr(backend, "model", None), "config", None), "_commit_hash", None)}})
    make_env = load_toolenv(data_root, prepared["manifest"]["physical_split"])
    predictions, traces = [], []
    prediction_path, results_path = output / "predictions.jsonl", output / "results.jsonl"
    # Create before the first sample and flush after each result, so interrupted
    # GPU runs keep durable outputs without rerunning completed actions.
    with prediction_path.open("x", encoding="utf-8"), results_path.open("x", encoding="utf-8"):
        pass
    for row in items:
        sid, started, env, session = row["id"], time.perf_counter(), None, None
        try:
            current = speech.get(sid)
            if current is None:
                raise ValueError(f"Missing ASR handoff row: {sid}")
            validate_speech(row, current)
            tools = select_available_tools(row, runtime.catalog)
            env = make_env(sid)  # exactly one inference ToolEnv; no reset/retry
            session = AgentSession(row, current, tools, env, backend, config)
            session.started = time.perf_counter()
            result = session.run()
            result["input_audit"] = {"speech": deepcopy(current),
                "request_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "available_tools": [t["name"] for t in tools]}
        except Exception as error:
            result = {"id": sid, "status": "input_or_runtime_error", "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
                "trace": deepcopy(session.trace) if session is not None else [], "submission": None}
            if env is not None:
                # If calls ran before failure, preserve the real log/hash.
                try:
                    result["submission"] = {"id": sid, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
                except Exception:
                    pass
        if result.get("submission") is not None:
            predictions.append(result["submission"])
        result["id"] = sid
        traces.append(result)
        # Persist every sample before continuing; a failed sample is never a fabricated [].
        write_json(output / "traces" / f"{len(traces)-1:06d}.json", result)
        for path, record in ((results_path, result), (prediction_path, result.get("submission"))):
            if record is not None:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(sid, result["status"], "calls:",
              len(result["submission"]["tool_calls"]) if result.get("submission") else "missing")
    return {"predictions": str(output / "predictions.jsonl"), "traces": str(output / "results.jsonl"),
        "samples": len(items), "successful": sum(r["status"] in ("finished", "waiting_for_user") for r in traces)}
