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
from .io import code_fingerprint, index_rows, read_json, read_rows, sha256, write_json, write_rows
from .runtime import OfficialRuntime


def run(bundle, split, data_root, speech_path, output, *, config_path=None,
        allow_local_test=False, frozen=None, oracle=False, backend=None):
    if oracle and split != "validation":
        raise ValueError("Oracle text is diagnostic-only on validation")
    folder = access_split(bundle, split, allow_local_test=allow_local_test, frozen=frozen, purpose="run_agent")
    manifest = read_json(folder / "manifest.json")
    items = read_rows(folder / "inputs.jsonl")
    if [r["id"] for r in items] != manifest["ids"]:
        raise ValueError("Manifest/input ID mismatch")
    integrity = read_json(Path(bundle) / "integrity.json")
    for name in ("manifest.json", "inputs.jsonl", *( ("references.jsonl",) if oracle else () )):
        if sha256(folder / name) != integrity[f"{split}/{name}"]:
            raise ValueError(f"Prepared artifact changed: {split}/{name}")
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
    settings = read_json(config_path) if config_path else {}
    config = AgentConfig(**settings.get("generation", {}))
    model = ModelLoadConfig(model_source=settings.get("model_source", "Qwen/Qwen3-4B-Instruct-2507"),
        revision=settings.get("model_revision"), device=settings.get("device", "cuda:0"), local_files_only=True)
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if settings.get("prompt_sha256", prompt_hash) != prompt_hash:
        raise ValueError("Configured prompt hash does not match SYSTEM_PROMPT")
    if split == "local_test":
        snapshot = read_json(frozen)
        if config_path is None or sha256(config_path) != snapshot["config_sha256"]:
            raise ValueError("Local test must use the frozen config file unchanged")
        if settings.get("quantization") != "fp16":
            raise ValueError("Current Qwen backend implements fp16; frozen quantization must be fp16")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"logical_split": split, "physical_split": manifest["physical_split"],
        "mode": "oracle_text_diagnostic" if oracle else "asr_text", "ids": manifest["ids"],
        "speech_sha256": None if oracle else sha256(speech_path), "config": settings,
        "effective_generation": asdict(config), "effective_model": asdict(model),
        "prompt_sha256": prompt_hash, "code": code_fingerprint(), "runtime": runtime.identity})
    make_env = load_toolenv(data_root, manifest["physical_split"])
    if backend is None:
        backend = load_qwen_backend(model, config)
    predictions, traces = [], []
    for row in items:
        sid, started, env = row["id"], time.perf_counter(), None
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
                "elapsed_seconds": time.perf_counter() - started, "trace": [], "submission": None}
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
    write_rows(output / "predictions.jsonl", predictions)
    write_rows(output / "results.jsonl", traces)
    return {"predictions": str(output / "predictions.jsonl"), "traces": str(output / "results.jsonl"),
        "samples": len(items), "successful": sum(r["status"] in ("finished", "waiting_for_user") for r in traces)}
