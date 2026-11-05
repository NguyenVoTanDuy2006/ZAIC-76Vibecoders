"""Run the seven-way ASR text correction comparison on saved predictions."""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from asr.finetuning.artifacts import read_jsonl, write_json, write_jsonl
from asr.finetuning.metrics import error_rates
from asr.common.text import normalize_text
from .checks import hard_check
from .evaluation import score_variant
from .generation import TransformersLM, generate_correction, judge_correction

RUN_VERSION = 1
VARIANTS = (
    "asr_original",
    "zero_shot_direct", "zero_shot_hard", "zero_shot_hard_soft",
    "few_shot_direct", "few_shot_hard", "few_shot_hard_soft",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_cache(path: Path, allowed: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id not in allowed or sample_id in result:
            raise ValueError(f"cache has unknown/duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _validate_source(rows: list[dict]) -> None:
    ids = [row.get("sample_id") for row in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("source has empty/invalid/duplicate sample_id")
    for row in rows:
        if not isinstance(row.get("prediction"), str) or not isinstance(row.get("reference"), str):
            raise ValueError(f"source missing prediction/reference: {row['sample_id']}")
        if not isinstance(row.get("conversation_id"), str) or type(row.get("is_latest")) is not bool:
            raise ValueError(f"source missing conversation metadata: {row['sample_id']}")


def _history_index(row: dict) -> int | None:
    if row["is_latest"]:
        return None
    stem = row["conversation_id"] + "_history_"
    suffix = row["sample_id"][len(stem):] if row["sample_id"].startswith(stem) else ""
    if not suffix.isdigit():
        raise ValueError(f"invalid history sample ID: {row['sample_id']}")
    return int(suffix)


def _histories(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["conversation_id"]].append(row)
    result = {}
    for group in groups.values():
        past = sorted((row for row in group if not row["is_latest"]), key=_history_index)
        if len({_history_index(row) for row in past}) != len(past):
            raise ValueError("duplicate history index")
        for row in group:
            index = _history_index(row)
            applicable = past if index is None else [turn for turn in past if _history_index(turn) < index]
            result[row["sample_id"]] = [
                {"history_index": _history_index(turn), "text": turn["prediction"]}
                for turn in applicable
            ]
    return result


def _select_examples(train_rows: list[dict], predictions: dict[str, dict]) -> list[dict]:
    candidates = []
    for row in train_rows:
        if row["sample_id"] not in predictions or not row["is_latest"]:
            continue
        original = predictions[row["sample_id"]]["prediction"]
        reference = row["reference"]
        if not 8 <= len(original.split()) <= 35 or not reference.strip():
            continue
        ref, hyp = normalize_text(reference), normalize_text(original)
        rate = error_rates([(ref, hyp)])["wer"]
        candidates.append((rate, row["sample_id"], original, reference, row["conversation_id"]))
    exact = sorted((item for item in candidates if item[0] == 0), key=lambda x: x[1])
    errors = sorted((item for item in candidates if item[0] and 0.05 <= item[0] <= 0.35),
                    key=lambda x: (abs(x[0] - 0.2), x[1]))
    if not exact or len(errors) < 2:
        raise ValueError("train support pool lacks one exact and two moderate-error examples")
    selected = [exact[0]]
    for item in errors:
        if item[4] not in {x[4] for x in selected}:
            selected.append(item)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("train support pool lacks examples from distinct conversations")
    return [{"sample_id": item[1], "asr_text": item[2], "corrected_text": item[3]}
            for item in selected]


def collect_fewshot_examples(
    *, train_manifest: Path, train_audio_root: Path, asr_config: Path,
    asr_model_path: Path, adapter_path: Path, output_dir: Path,
    asr_factory=None, audio_loader=None, pool_size: int = 128,
) -> list[dict]:
    """Infer train audio only; references enter only the fixed training examples."""
    rows = read_jsonl(train_manifest)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate train manifest sample IDs")
    # Keep latest turns for self-contained demonstrations.
    eligible = sorted((row for row in rows if row["is_latest"]),
                      key=lambda row: hashlib.sha256(row["sample_id"].encode()).hexdigest())
    chosen = eligible[:pool_size]
    allowed = {row["sample_id"] for row in chosen}
    cache_path = output_dir / "train_support_predictions.jsonl"
    cached = _jsonl_cache(cache_path, allowed)
    if len(cached) < len(chosen):
        from asr.integration.predict import load_model_settings
        from asr.benchmark.infrastructure.models.qwen_hf_peft import QwenHFPEFTAdapter

        if audio_loader is None:
            from asr.benchmark.application.audio_service import prepare_audio
            audio_loader = prepare_audio

        settings = load_model_settings(asr_config, adapter_path)
        asr = asr_factory() if asr_factory else QwenHFPEFTAdapter()
        try:
            asr.load(str(asr_model_path), settings, {"device": "cuda:0", "dtype": "float16"})
            for row in chosen:
                sample_id = row["sample_id"]
                if sample_id in cached:
                    continue
                path = (train_audio_root / row["audio"]).resolve()
                if not path.is_relative_to(train_audio_root.resolve()):
                    raise ValueError(f"train audio escapes root: {sample_id}")
                result = asr.transcribe(audio_loader(path, {"sample_rate": 16000,
                                                         "resample_nonmatching_audio": False}))
                record = {"sample_id": sample_id, "prediction": result.text}
                _append(cache_path, record)
                cached[sample_id] = record
        finally:
            asr.unload()
    if len(cached) != len(chosen):
        raise ValueError("incomplete train support predictions")
    examples = _select_examples(chosen, cached)
    write_json(output_dir / "fewshot_examples.json", examples)
    return examples


def run_experiment(
    *, predictions_path: str | Path, train_manifest: str | Path,
    train_audio_root: str | Path, asr_config: str | Path,
    asr_model_path: str | Path, adapter_path: str | Path,
    lm_model_path: str | Path, output_dir: str | Path,
    lm_factory=None, asr_factory=None, audio_loader=None,
    support_pool_size: int = 128, expected_samples: int = 470,
    expected_latest: int = 155,
) -> dict:
    paths = {name: Path(value).expanduser().resolve() for name, value in {
        "predictions": predictions_path, "train_manifest": train_manifest,
        "train_audio_root": train_audio_root, "asr_config": asr_config,
        "asr_model_path": asr_model_path, "adapter_path": adapter_path,
        "lm_model_path": lm_model_path, "output_dir": output_dir,
    }.items()}
    for name in ("predictions", "train_manifest", "asr_config"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    for name in ("train_audio_root", "asr_model_path", "adapter_path", "lm_model_path"):
        if not paths[name].is_dir():
            raise FileNotFoundError(paths[name])
    source = read_jsonl(paths["predictions"])
    _validate_source(source)
    if len(source) != expected_samples:
        raise ValueError(f"expected {expected_samples} validation clips, found {len(source)}")
    latest = sum(row["is_latest"] for row in source)
    if latest != expected_latest:
        raise ValueError(f"expected {expected_latest} latest turns, found {latest}")
    baseline_rows = [{"sample_id": row["sample_id"], "prediction": row["prediction"]}
                     for row in source]
    baseline = score_variant(source, baseline_rows)
    previous_summary = paths["predictions"].with_name("summary.json")
    if previous_summary.is_file():
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        for group in ("all", "latest"):
            for metric in ("raw_cer", "normalized_cer", "raw_wer", "normalized_wer"):
                if abs(baseline[group][metric] - previous[group][metric]) > 1e-10:
                    raise ValueError(f"baseline metric mismatch: {group}.{metric}")
    history = _histories(source)
    identity = {"version": RUN_VERSION, "source_sha256": _sha256(paths["predictions"]),
                "train_manifest_sha256": _sha256(paths["train_manifest"]),
                "asr_config_sha256": _sha256(paths["asr_config"]),
                "asr_model_path": str(paths["asr_model_path"]),
                "train_audio_root": str(paths["train_audio_root"]),
                "adapter_path": str(paths["adapter_path"]),
                "adapter_sha256": _sha256(paths["adapter_path"] / "adapter_model.safetensors"),
                "lm_model_path": str(paths["lm_model_path"]),
                "lm_config_sha256": _sha256(paths["lm_model_path"] / "config.json"),
                "generation_code_sha256": _sha256(Path(__file__).with_name("generation.py")),
                "checks_code_sha256": _sha256(Path(__file__).with_name("checks.py")),
                "support_pool_size": support_pool_size,
                "expected_samples": expected_samples, "expected_latest": expected_latest}
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "run_metadata.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != identity:
            raise ValueError("run inputs changed; choose a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("nonempty output directory lacks run_metadata.json")
        write_json(metadata, identity)
    examples = collect_fewshot_examples(
        train_manifest=paths["train_manifest"], train_audio_root=paths["train_audio_root"],
        asr_config=paths["asr_config"], asr_model_path=paths["asr_model_path"],
        adapter_path=paths["adapter_path"], output_dir=output,
        asr_factory=asr_factory, audio_loader=audio_loader, pool_size=support_pool_size,
    )
    ids = {row["sample_id"] for row in source}
    generations = {}
    for mode in ("zero_shot", "few_shot"):
        generations[mode] = _jsonl_cache(output / "generation" / f"{mode}.jsonl", ids)
    judge_cache = {mode: _jsonl_cache(output / "judge" / f"{mode}.jsonl", ids)
                   for mode in generations}
    lm = None

    def get_lm():
        nonlocal lm
        if lm is None:
            lm = lm_factory() if lm_factory else TransformersLM(str(paths["lm_model_path"]))
        return lm

    for mode in ("zero_shot", "few_shot"):
        cache_path = output / "generation" / f"{mode}.jsonl"
        for row in source:
            sid = row["sample_id"]
            if sid in generations[mode]:
                continue
            candidate, status = generate_correction(
                get_lm(), row["prediction"], history[sid], examples if mode == "few_shot" else []
            )
            record = {"sample_id": sid, "candidate": candidate, "status": status}
            _append(cache_path, record)
            generations[mode][sid] = record
    results = {name: [] for name in VARIANTS}
    audit = []
    for row in source:
        sid, original = row["sample_id"], row["prediction"]
        results["asr_original"].append({"sample_id": sid, "prediction": original})
        for mode in ("zero_shot", "few_shot"):
            generated = generations[mode][sid]
            candidate = generated["candidate"]
            okay, hard_reason = hard_check(original, candidate, history[sid])
            if generated["status"] != "generated":
                okay, hard_reason = False, generated["status"]
            hard_text = candidate if okay else original
            decision, soft_reason = "skipped", "unchanged_or_hard_rejected"
            if okay and normalize_text(candidate) != normalize_text(original):
                cached = judge_cache[mode].get(sid)
                if cached is None:
                    decision, soft_reason = judge_correction(get_lm(), original, candidate, history[sid])
                    cached = {"sample_id": sid, "candidate": candidate,
                              "decision": decision, "reason": soft_reason}
                    _append(output / "judge" / f"{mode}.jsonl", cached)
                    judge_cache[mode][sid] = cached
                elif cached.get("candidate") != candidate:
                    raise ValueError(f"stale judge cache for {sid}")
                decision, soft_reason = cached["decision"], cached["reason"]
            soft_text = candidate if okay and decision == "accept" else original
            for suffix, text in (("direct", candidate), ("hard", hard_text),
                                 ("hard_soft", soft_text)):
                results[f"{mode}_{suffix}"].append({"sample_id": sid, "prediction": text})
            audit.append({"sample_id": sid, "mode": mode, "original": original,
                          "candidate": candidate, "generation_status": generated["status"],
                          "hard_accept": okay, "hard_reason": hard_reason,
                          "soft_decision": decision, "soft_reason": soft_reason})
    summaries = {}
    for name, predictions in results.items():
        write_jsonl(output / "predictions" / f"{name}.jsonl", predictions)
        summaries[name] = score_variant(source, predictions)
    write_jsonl(output / "audit.jsonl", audit)
    write_json(output / "check_counts.json", {
        mode: {
            "hard": dict(Counter("accept" if row["hard_accept"] else "reject"
                                 for row in audit if row["mode"] == mode)),
            "soft": dict(Counter(row["soft_decision"] for row in audit if row["mode"] == mode)),
        } for mode in ("zero_shot", "few_shot")
    })
    write_json(output / "comparison.json", summaries)
    best = min(VARIANTS, key=lambda name: (
        summaries[name]["all"]["normalized_wer"],
        summaries[name]["all"]["normalized_cer"],
        VARIANTS.index(name),
    ))
    baseline_wer = summaries["asr_original"]["all"]["normalized_wer"]
    best_wer = summaries[best]["all"]["normalized_wer"]
    write_json(output / "selection.json", {
        "metric": "all.normalized_wer", "best_variant": best,
        "baseline_wer": baseline_wer, "best_wer": best_wer,
        "absolute_wer_reduction": baseline_wer - best_wer,
        "relative_wer_reduction": (baseline_wer - best_wer) / baseline_wer if baseline_wer else 0.0,
    })
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "group", "samples", "raw_cer",
            "normalized_cer", "raw_wer", "normalized_wer"])
        writer.writeheader()
        for name, groups in summaries.items():
            for group, metrics in groups.items():
                writer.writerow({"variant": name, "group": group, **{
                    field: metrics[field] for field in ("samples", "raw_cer", "normalized_cer",
                                                       "raw_wer", "normalized_wer")}})
    return summaries
