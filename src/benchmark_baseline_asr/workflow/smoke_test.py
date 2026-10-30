from __future__ import annotations

from benchmark_baseline_asr.domain.entities import DatasetSample
from benchmark_baseline_asr.infrastructure.models.factory import create_adapter
from benchmark_baseline_asr.infrastructure.runtime.reproducibility import configure_determinism
from benchmark_baseline_asr.infrastructure.runtime.timing import timed_call


def _prepare_audio(audio_path, audio_config):
    from benchmark_baseline_asr.application.audio_service import prepare_audio

    return prepare_audio(audio_path, audio_config)


def smoke_test_models(
    samples: list[DatasetSample],
    models: list[dict],
    resolved_models: dict[str, dict],
    audio_config: dict,
    runtime_config: dict,
) -> list[dict]:
    """Load and run every configured model on one deterministic, representative sample."""
    if not samples:
        raise ValueError("Cannot smoke-test ASR models: the prepared dataset has no samples.")
    if not models:
        raise ValueError("Cannot smoke-test ASR models: no models are configured.")

    sample = sorted(samples, key=lambda row: (row.duration_seconds, row.sample_id))[len(samples) // 2]
    audio = _prepare_audio(sample.audio_path, audio_config)
    runtime = dict(runtime_config)
    runtime["device"] = "cuda:0"
    configure_determinism(
        runtime.get("seed", 0),
        runtime.get("cpu_threads_per_worker", 2),
    )

    results = []
    for model in models:
        model_key = model["key"]
        model_config = model["settings"]
        model_path = resolved_models[model_key]["path"]
        adapter = create_adapter(model_config["adapter"])
        try:
            adapter.load(model_path, model_config, runtime)
            transcription, elapsed_seconds = timed_call(
                lambda: adapter.transcribe(audio), device=runtime["device"]
            )
            results.append(
                {
                    "model": model_key,
                    "sample_id": sample.sample_id,
                    "text": transcription.text,
                    "latency_seconds": elapsed_seconds,
                }
            )
        finally:
            adapter.unload()
    return results
