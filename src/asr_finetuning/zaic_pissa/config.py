from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from .artifacts import fingerprint

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment.json"


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path, seen):
    path = Path(path).expanduser().resolve()
    if path in seen:
        raise ValueError(f"circular config include: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config must be an object: {path}")
    includes = value.pop("includes", [])
    if not isinstance(includes, list):
        raise ValueError("includes must be a list")
    result = {}
    for relative in includes:
        result = merge(result, _read(path.parent / relative, seen | {path}))
    return merge(result, value)


def load_config(path=None):
    result = _read(DEFAULT_CONFIG, set())
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        result = merge(result, _read(path, set()))
    validate_config(result)
    return result


def validate_config(c):
    sections = {
        "model",
        "dataset",
        "split",
        "pissa",
        "train",
        "scst",
        "semantic",
        "decode",
        "runtime",
        "output",
    }
    if set(c) != sections:
        raise ValueError(f"config sections must be {sorted(sections)}")
    defaults = _read(DEFAULT_CONFIG, set())
    for section in sections:
        if not isinstance(c[section], dict) or set(c[section]) != set(
            defaults[section]
        ):
            raise ValueError(f"unknown or missing fields in {section}")
    if c["model"]["adapter_path"] is not None:
        raise ValueError(
            "PiSSA starts from an original base; use --resume for native training state"
        )
    if c["model"]["trust_remote_code"] or c["model"]["backend"] != "qwen3_asr":
        raise ValueError("use native qwen3_asr without remote code")
    for section in ["model", "semantic"]:
        if not c[section]["model_id"] or not re.fullmatch(
            r"[0-9a-f]{40}", c[section]["revision"]
        ):
            raise ValueError(
                f"{section} requires a model_id and pinned commit revision"
            )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", c["output"]["run_id"]):
        raise ValueError("unsafe run_id")
    for key in ["epochs", "world_size", "batch_size", "gradient_accumulation_steps"]:
        if type(c["train"][key]) is not int or c["train"][key] < 1:
            raise ValueError(f"train.{key} must be positive")
    if c["train"]["batch_size"] != 1:
        raise ValueError("SCST implementation supports batch_size=1 per rank")
    if c["train"]["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if c["train"]["precision"] not in {"fp16", "fp32"}:
        raise ValueError("precision must be fp16 or fp32")
    if (
        c["model"]["dtype"]
        != {"fp16": "float16", "fp32": "float32"}[c["train"]["precision"]]
    ):
        raise ValueError("model dtype and precision disagree")
    for key in ["learning_rate", "eps", "gradient_clip_norm"]:
        if not math.isfinite(c["train"][key]) or c["train"][key] <= 0:
            raise ValueError(f"invalid {key}")
    if not 0 <= c["train"]["warmup_ratio"] < 1 or c["train"]["weight_decay"] < 0:
        raise ValueError("invalid warmup/weight_decay")
    if len(c["train"]["betas"]) != 2 or any(
        not 0 <= b < 1 for b in c["train"]["betas"]
    ):
        raise ValueError("invalid AdamW betas")
    if c["pissa"]["r"] < 1 or c["pissa"]["alpha"] <= 0 or c["pissa"]["dropout"] != 0:
        raise ValueError("PiSSA requires positive rank/alpha and zero dropout")
    if not re.fullmatch(r"pissa_niter_\d+", c["pissa"]["init_lora_weights"]):
        raise ValueError("use explicit fast PiSSA initialization")
    if set(c["pissa"]["target_modules"]) != {"q_proj", "v_proj"}:
        raise ValueError("target q_proj/v_proj in encoder and decoder")
    if not math.isfinite(c["scst"]["weight"]) or c["scst"]["weight"] < 0:
        raise ValueError("invalid SCST weight")
    if (c["scst"]["temperature"], c["scst"]["top_p"], c["scst"]["top_k"]) != (
        1.0,
        1.0,
        0,
    ):
        raise ValueError("sampling and replay require temperature=1/top_p=1/top_k=0")
    if (
        c["semantic"]["idf"]
        or c["semantic"]["rescale"]
        or not c["semantic"]["word_segment"]
    ):
        raise ValueError("semantic metric must use PyVi, no IDF/rescale")
    if (
        c["semantic"]["num_layers"] < 1
        or c["semantic"]["batch_size"] < 1
        or c["semantic"]["reference_cache_size"] < 0
    ):
        raise ValueError("invalid semantic dimensions")
    a, b = c["split"]["validation_fraction"], c["split"]["test_fraction"]
    if not 0 < a < 1 or not 0 < b < 1 or a + b >= 1:
        raise ValueError("invalid split fractions")
    if (
        c["decode"]["max_new_tokens"] < 1
        or c["decode"]["num_beams"] != 1
        or c["decode"]["do_sample"]
    ):
        raise ValueError("evaluation uses greedy decoding")
    if c["runtime"]["device"] not in {"cuda", "cpu"} or c["runtime"]["cpu_threads"] < 1:
        raise ValueError("invalid runtime")


def config_fingerprint(config):
    c = copy.deepcopy(config)
    c["dataset"].pop("root")
    c["runtime"].pop("cache_dir")
    c.pop("output")
    return fingerprint(c)


def run_dir(config):
    return (
        Path(config["output"]["root"]).expanduser().resolve()
        / config["output"]["run_id"]
    )
