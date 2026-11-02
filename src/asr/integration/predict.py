"""Run the existing ASR adapters on current AND history audio for LifeMate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from asr.integration.predictions import (
    PredictionContractError, assemble_speech_rows, build_audio_manifest,
    index_audio_predictions, load_items, read_jsonl, validate_speech_rows, write_speech_file,
)


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_model_settings(config_path, adapter_path=None):
    """Reuse friend's baseline configs or a SINGLE native PiSSA/AdaLoRA config."""
    config_path = Path(config_path).resolve()
    document = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if "model" in document:
        spec = document["model"]
        if spec.get("backend") != "qwen3_asr" or spec.get("trust_remote_code", False):
            raise ValueError("Native adapter config cần backend=qwen3_asr và trust_remote_code=false")
        settings = {"adapter": "qwen_hf_peft", "repo_id": spec["model_id"],
                    "revision": spec.get("revision"), "adapter_path": spec.get("adapter_path"),
                    "dtype": spec.get("dtype", "float16"), "language": "Vietnamese",
                    "max_new_tokens": spec.get("max_new_tokens", 768)}
    else:
        settings = deepcopy(document)
    if settings.get("adapter") not in {"phowhisper", "qwen_asr", "qwen_hf_peft"}:
        raise ValueError("Model config phải dùng một ASR adapter có sẵn của repo")
    if adapter_path is not None:
        settings["adapter_path"] = str(Path(adapter_path).expanduser().resolve())
    if "model" in document and not settings.get("adapter_path"):
        raise ValueError("Thiếu adapter_path cho native ASR adapter; không tự chuyển sang base model")
    if settings.get("adapter_path"):
        if settings["adapter"] != "qwen_hf_peft":
            raise ValueError("PEFT adapter cần backend qwen_hf_peft")
        checkpoint = Path(settings["adapter_path"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = config_path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        from asr.benchmark.infrastructure.models.peft_checkpoints import adapter_digests
        settings["adapter_path"] = str(checkpoint)
        settings["adapter_sha256"] = adapter_digests(checkpoint)
        peft = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        if peft.get("peft_type") not in {"LORA", "ADALORA"}:
            raise ValueError("Adapter phải là converted PiSSA/LoRA hoặc AdaLoRA")
        if peft.get("peft_type") == "LORA" and peft.get("init_lora_weights") is not True:
            raise ValueError("PiSSA phải export thành standard LoRA trên base gốc trước khi inference")
        if peft.get("revision") and peft["revision"] != settings.get("revision"):
            raise ValueError("Adapter revision không khớp base model config")
    return settings


def resolve_base_snapshot(settings, model_path=None, *, local_files_only=True, cache_dir=None):
    """Default offline/cache only. No credentials are stored in output/config."""
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Thiếu checkpoint ASR local (config.json): {path}")
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=settings["repo_id"], revision=settings.get("revision"),
        local_files_only=local_files_only, cache_dir=cache_dir,
    )).resolve()


def predict_lifemate(data_root, split, model_config_path, output_dir, *, model_path=None,
                     adapter_path=None, device="cuda:0", local_files_only=True,
                     cache_dir=None, adapter_factory=None, audio_loader=None):
    """Full split -> audio_predictions.jsonl -> asr_predictions.jsonl.

    Current/history are both inferred. Only waveform goes into the ASR adapter;
    gold labels/transcripts never do. The audio cache supports restart with the
    same dataset/model identity. A failed clip stops the run without publishing
    the final speech file; rerun retries missing clips. This is a staged ASR run,
    not a measurement of the complete competition ASR+agent latency.
    """
    items = load_items(data_root, split)
    manifest = build_audio_manifest(items, data_root, split)
    settings = load_model_settings(model_config_path, adapter_path)
    base_path = resolve_base_snapshot(settings, model_path, local_files_only=local_files_only, cache_dir=cache_dir)
    runtime = {"device": device, "dtype": settings.get("dtype", "float16"),
               "do_sample": False, "num_beams": 1, "seed": 1234}
    audio_config = {"sample_rate": 16000, "resample_nonmatching_audio": False}
    identity = {"version": 1, "split": split, "model": settings, "base_path": str(base_path),
                "runtime": runtime, "audio": audio_config,
                "clips": [{"sample_id": clip["sample_id"], "audio": clip["audio"],
                           "sha256": _digest(clip["audio_path"])} for clip in manifest]}
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "prediction_metadata.json"
    cache_path = output / "audio_predictions.jsonl"
    final_path = output / "asr_predictions.jsonl"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != identity:
            raise PredictionContractError("Dataset/model/adapter đã đổi; chọn output_dir mới")
    else:
        if cache_path.exists() or final_path.exists():
            raise PredictionContractError("Output có predictions nhưng thiếu metadata; dùng thư mục mới")
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, ensure_ascii=False, indent=2)

    predictions = read_jsonl(cache_path) if cache_path.exists() else []
    cached = index_audio_predictions(predictions, manifest)
    if final_path.exists():
        rows = assemble_speech_rows(items, manifest, predictions)
        existing = read_jsonl(final_path)
        validate_speech_rows(existing, items)
        if existing != rows:
            raise PredictionContractError("Speech file không khớp audio prediction cache")
        return {"prediction_path": str(final_path), "audio_prediction_path": str(cache_path),
                "conversations": len(items), "audio_clips": len(manifest), "new_audio_clips": 0}

    pending = [clip for clip in manifest if clip["sample_id"] not in cached]
    if pending:
        if adapter_factory is None:
            from asr.benchmark.infrastructure.models.factory import create_adapter
            adapter_factory = create_adapter
        if audio_loader is None:
            from asr.benchmark.application.audio_service import prepare_audio
            audio_loader = prepare_audio
        adapter = adapter_factory(settings["adapter"])
        try:
            adapter.load(str(base_path), deepcopy(settings), runtime)  # existing ASR load(), once
            for count, clip in enumerate(pending, 1):
                started = time.perf_counter()
                try:
                    result = adapter.transcribe(audio_loader(clip["audio_path"], audio_config))
                    if not isinstance(result.text, str):
                        raise PredictionContractError("ASR phải trả Transcription.text dạng string")
                    row = {"sample_id": clip["sample_id"], "text": result.text,
                           "language": getattr(result, "language", None),
                           "latency_seconds": time.perf_counter() - started}
                except Exception as error:
                    with (output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"sample_id": clip["sample_id"],
                            "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False) + "\n")
                    raise
                with cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.append(row)
                print(f"ASR {count}/{len(pending)}: {clip['sample_id']}")
        finally:
            adapter.unload()  # release ASR weights before the Qwen agent is loaded
    rows = assemble_speech_rows(items, manifest, predictions)
    write_speech_file(final_path, rows)
    return {"prediction_path": str(final_path), "audio_prediction_path": str(cache_path),
            "conversations": len(items), "audio_clips": len(manifest), "new_audio_clips": len(pending)}
