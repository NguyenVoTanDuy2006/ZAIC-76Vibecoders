from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from benchmark_baseline_asr.infrastructure.evaluation.error_metrics import corpus_error_rates
from benchmark_baseline_asr.infrastructure.evaluation.text_normalizer import normalize_text
from benchmark_baseline_asr.infrastructure.storage.artifacts import read_jsonl, write_json, write_jsonl
from benchmark_baseline_asr.infrastructure.storage.artifacts import read_json


def evaluate_predictions(
    manifest_path: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path,
    normalization_config: dict | None = None,
) -> dict:
    manifest = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    predictions = {row["sample_id"]: row for row in read_jsonl(prediction_path)}
    per_sample = []
    pairs = defaultdict(list)
    for sample_id, prediction in predictions.items():
        if sample_id not in manifest:
            raise ValueError(f"prediction references unknown sample ID: {sample_id}")
        reference, hypothesis = manifest[sample_id]["reference"], prediction["text"]
        pairs["raw_char"].append((reference, hypothesis))
        pairs["raw_word"].append((reference, hypothesis))
        normalized_reference = normalize_text(reference, normalization_config)
        normalized_hypothesis = normalize_text(hypothesis, normalization_config)
        pairs["normalized_char"].append((normalized_reference, normalized_hypothesis))
        pairs["normalized_word"].append((normalized_reference, normalized_hypothesis))
        per_sample.append({
            "sample_id": sample_id,
            "reference": reference,
            "prediction": hypothesis,
            "reference_normalized": normalized_reference,
            "prediction_normalized": normalized_hypothesis,
            "latency_seconds": prediction.get("latency_seconds"),
            "audio_duration_seconds": manifest[sample_id]["duration_seconds"],
        })
    raw_cer = corpus_error_rates(pairs["raw_char"], units="character")
    raw_wer = corpus_error_rates(pairs["raw_word"], units="word")
    norm_cer = corpus_error_rates(pairs["normalized_char"], units="character")
    norm_wer = corpus_error_rates(pairs["normalized_word"], units="word")
    successful_audio = sum(row["audio_duration_seconds"] for row in per_sample)
    inference_seconds = sum((row["latency_seconds"] or 0) for row in per_sample)
    latencies = sorted(row["latency_seconds"] for row in per_sample if row["latency_seconds"] is not None)
    summary = {
        "expected_samples": len(manifest), "successful_samples": len(predictions),
        "coverage": len(predictions) / len(manifest) if manifest else 1.0,
        "raw_cer": raw_cer.rate, "raw_wer": raw_wer.rate,
        "normalized_cer": norm_cer.rate, "normalized_wer": norm_wer.rate,
        "raw_cer_edits": _edit_summary(raw_cer), "raw_wer_edits": _edit_summary(raw_wer),
        "normalized_cer_edits": _edit_summary(norm_cer), "normalized_wer_edits": _edit_summary(norm_wer),
        "latency_mean_seconds": inference_seconds / len(latencies) if latencies else None,
        "latency_p50_seconds": _percentile(latencies, 0.50), "latency_p95_seconds": _percentile(latencies, 0.95),
        "successful_audio_seconds": successful_audio,
        "inference_seconds_sum": inference_seconds,
        "rtf_sum_latency": inference_seconds / successful_audio if successful_audio else None,
    }
    out = Path(output_dir)
    timing_path = out / "run_timing.json"
    if timing_path.is_file():
        timing = read_json(timing_path)
        wall_seconds = timing.get("wall_seconds")
        summary["inference_wall_seconds"] = wall_seconds
        summary["rtf_wall"] = wall_seconds / successful_audio if wall_seconds and successful_audio else None
        summary["audio_seconds_per_wall_second"] = successful_audio / wall_seconds if wall_seconds else None
        summary["worker_model_load_seconds"] = [row.get("model_load_seconds") for row in timing.get("workers", [])]
        summary["worker_warmup_seconds"] = [row.get("warmup_seconds") for row in timing.get("workers", [])]
    errors = []
    for error_path in out.glob("worker_*/errors.jsonl"):
        errors.extend(read_jsonl(error_path))
    successful_ids = set(predictions)
    failed_ids = {row.get("sample_id") for row in errors} - successful_ids
    summary["failed_samples"] = len(failed_ids)
    summary["result_complete"] = summary["successful_samples"] == summary["expected_samples"]
    write_jsonl(out / "metrics.jsonl", per_sample)
    write_json(out / "summary.json", summary)
    return summary


def _edit_summary(result):
    return {"substitutions": result.substitutions, "deletions": result.deletions,
            "insertions": result.insertions, "reference_units": result.reference_units}


def _percentile(values: list[float], fraction: float):
    if not values:
        return None
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]
