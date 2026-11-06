"""Grouped 8:1:1 agent manifests; references never enter inference inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import wave

from lifemate.agent import prepare_request, select_available_tools
from .io import artifact_hashes, code_fingerprint, index_rows, read_json, read_rows, sha256, write_json, write_rows


def category(row):
    calls = row["label"]["tool_calls"]
    return "out_of_scope" if not calls else "missing_info" if any(
        c["name"] == "sys.ask_missing_info" for c in calls) else "action"


def agent_item(row):
    """Allowlist context too: installed apps contain names/schema access, no state."""
    result = prepare_request(row)
    context = row["context"]
    result["context"] = {k: deepcopy(context[k]) for k in (
        "current_time", "timezone", "location", "user_profile") if k in context}
    result["context"]["installed_apps"] = {
        app: {"tools": list(spec["tools"])} for app, spec in context["installed_apps"].items()}
    result["audio"] = row["audio"]
    return result


def _normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _choose_groups(groups, target, rng):
    # Random, round-robin strata before exact subset sum avoids a contiguous ID split.
    buckets = defaultdict(list)
    for group in groups:
        buckets[group["stratum"]].append(group)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = []
    while any(buckets.values()):
        keys = [k for k, v in buckets.items() if v]
        rng.shuffle(keys)
        ordered.extend(buckets[k].pop() for k in keys)
    # Preference for proportional strata; remaining groups still available for exact fill.
    counts = Counter(g["stratum"] for g in groups for _ in g["ids"])
    total = sum(counts.values())
    used = Counter()
    ordered.sort(key=lambda g: -counts[g["stratum"]])
    preferred, rest = [], []
    for group in ordered:
        key, size = group["stratum"], len(group["ids"])
        if used[key] + size <= round(target * counts[key] / total):
            preferred.append(group)
            used[key] += size
        else:
            rest.append(group)
    reachable = {0: ()}
    for group in preferred + rest:
        size = len(group["ids"])
        for n, selected in list(reachable.items()):
            if n + size <= target and n + size not in reachable:
                reachable[n + size] = (*selected, group["group_id"])
        if target in reachable:
            return set(reachable[target])
    raise ValueError(f"Cannot reach exact split size {target} without breaking groups; no split written")


def prepare(data_root, output, seed=2026):
    root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a fresh split output directory")
    source = root / "train/train.jsonl"
    rows = read_rows(source)
    indexed = index_rows(rows)
    if not rows:
        raise ValueError("Empty training source")
    catalog_path = root / "train/train_tools_catalog.json"
    catalog = read_json(catalog_path)
    parent = {sample_id: sample_id for sample_id in indexed}
    evidence, audio, owners, transcript_owners = [], {}, {}, {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def link(a, b, reason, value):
        if a != b:
            evidence.append({"left": a, "right": b, "reason": reason, "value": value})
            parent[find(b)] = find(a)
    for row in rows:
        sid = row["id"]
        select_available_tools(agent_item(row), catalog)
        turns = [(None, row), *[(i, t) for i, t in enumerate(row.get("history", [])) if t["role"] == "user"]]
        for index, turn in turns:
            relative = turn["audio"]
            if not isinstance(relative, str) or Path(relative).parts[:1] != ("audio",) or any(
                    part.endswith("_environment") for part in Path(relative).parts):
                raise ValueError(f"{sid}: audio reference must be inside audio/, never an environment file")
            path = (root / "train" / relative).resolve()
            if not path.is_relative_to(root / "train") or not path.is_file():
                raise ValueError(f"{sid}: missing/unsafe audio {relative}")
            if relative not in audio:
                with wave.open(str(path), "rb") as stream:
                    audio[relative] = {"sha256": sha256(path), "channels": stream.getnchannels(),
                        "sample_rate": stream.getframerate(), "sample_width": stream.getsampwidth(),
                        "frames": stream.getnframes(), "duration_seconds": stream.getnframes() / stream.getframerate(),
                        "compression": stream.getcomptype(), "references": []}
            entry = audio[relative]
            text = _normalize(turn["audio_text"])
            entry["references"].append({"id": sid, "history_index": index, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
            digest = entry["sha256"]
            if digest in owners:
                link(sid, owners[digest], "identical_audio_bytes_current_or_history", digest)
            owners[digest] = sid
        # Full conversation, including assistant context, is stronger than a generic short utterance.
        conversation = {"current": _normalize(row["audio_text"]), "history": [
            {"role": t["role"], "text": _normalize(t.get("audio_text", t.get("text", ""))),
             "tool_calls": t.get("tool_calls", [])} for t in row.get("history", [])]}
        signature = json.dumps(conversation, ensure_ascii=False, sort_keys=True)
        if signature in transcript_owners:
            link(sid, transcript_owners[signature], "identical_normalized_conversation", hashlib.sha256(signature.encode()).hexdigest())
        transcript_owners[signature] = sid
        # Group the same full conversation with only numeric substitutions.
        numeric_signature = re.sub(r"\d+", "#", signature)
        template_key = "numeric-template:" + numeric_signature
        if template_key in transcript_owners:
            link(sid, transcript_owners[template_key], "identical_conversation_numeric_template",
                 hashlib.sha256(numeric_signature.encode()).hexdigest())
        transcript_owners[template_key] = sid
        for key in ("conversation_id", "scenario_id", "template_id", "speaker_id"):
            if row.get(key) is not None:
                signature_key = f"metadata:{key}:{row[key]}"
                if signature_key in owners:
                    link(sid, owners[signature_key], key, row[key])
                owners[signature_key] = sid
    grouped = defaultdict(list)
    for sid in indexed:
        grouped[find(sid)].append(sid)
    groups = []
    for ids in grouped.values():
        ids.sort()
        group_id = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
        strata = Counter(f"{category(indexed[sid])}:history={bool(indexed[sid].get('history'))}" for sid in ids)
        groups.append({"group_id": group_id, "ids": ids, "stratum": strata.most_common(1)[0][0]})
    n = len(rows)
    targets = {"train": int(n * .8), "validation": int(n * .1)}
    targets["local_test"] = n - sum(targets.values())
    rng = random.Random(seed)
    test_groups = _choose_groups(groups, targets["local_test"], rng)
    dev_groups = _choose_groups([g for g in groups if g["group_id"] not in test_groups], targets["validation"], rng)
    membership = {sid: ("local_test" if g["group_id"] in test_groups else
                        "validation" if g["group_id"] in dev_groups else "train", g["group_id"])
                  for g in groups for sid in g["ids"]}
    # Near templates are evidence for review, not invented speaker/scenario metadata.
    templates = defaultdict(list)
    for row in rows:
        text = _normalize(row["audio_text"])
        templates[re.sub(r"\d+", "#", text)].append(row["id"])
    near = [ids for ids in templates.values() if len(ids) > 1 and len({membership[i][0] for i in ids}) > 1]
    conflicts = [key for key, value in audio.items() if len({r["text_sha256"] for r in value["references"]}) > 1]
    # Also check different audio paths with equal bytes for conflicting references.
    byte_texts = defaultdict(set)
    for value in audio.values():
        byte_texts[value["sha256"]].update(r["text_sha256"] for r in value["references"])
    byte_conflicts = [digest for digest, texts in byte_texts.items() if len(texts) > 1]
    if conflicts or byte_conflicts:
        raise ValueError(f"Audio/transcript reference conflicts: {len(conflicts)} paths, {len(byte_conflicts)} byte hashes")
    summaries = {}
    for split in targets:
        selected = [r for r in rows if membership[r["id"]][0] == split]
        folder = output / split
        write_json(folder / "manifest.json", {"split": split, "physical_split": "train", "ids": [r["id"] for r in selected],
            "groups": {r["id"]: membership[r["id"]][1] for r in selected}})
        write_rows(folder / "inputs.jsonl", (agent_item(r) for r in selected))
        write_rows(folder / "references.jsonl", ({"id": r["id"], "group_id": membership[r["id"]][1],
            "label": r["label"], "oracle_speech": {"id": r["id"], "current_text": r["audio_text"],
            "history_user_texts": [{"history_index": i, "text": t["audio_text"]}
                for i, t in enumerate(r.get("history", [])) if t["role"] == "user"]}} for r in selected))
        summaries[split] = {"samples": len(selected), "categories": dict(Counter(category(r) for r in selected)),
            "with_history": sum(bool(r.get("history")) for r in selected),
            "apps": dict(Counter(app for r in selected for app in r["context"]["installed_apps"])),
            "gold_call_counts": dict(Counter(str(len(r["label"]["tool_calls"])) for r in selected))}
    write_rows(output / "grouping_evidence.jsonl", evidence)
    write_json(output / "audio_audit.json", audio)
    metadata = {"seed": seed, "physical_split": "train", "source_sha256": sha256(source),
        "catalog_sha256": sha256(catalog_path), "version": (root / "VERSION").read_text(encoding="utf-8") if (root / "VERSION").exists() else None,
        "groups": len(groups), "max_group_size": max(len(g["ids"]) for g in groups), "splits": summaries,
        "overlapping_ids": 0, "overlapping_audio_hashes": 0, "reference_conflicts": conflicts,
        "metadata_fields_found": [k for k in ("conversation_id", "scenario_id", "template_id", "speaker_id") if any(k in r for r in rows)],
        "limitations": ["No speaker/region inferred from IDs", "Full conversation numeric templates grouped conservatively; broader semantic paraphrases are not exhaustively detected"],
        "cross_split_numeric_template_candidates": near, "code": code_fingerprint()}
    write_json(output / "metadata.json", metadata)
    write_json(output / "integrity.json", artifact_hashes(output))
    return metadata


def verify(bundle, data_root):
    bundle, root = Path(bundle), Path(data_root)
    expected = read_json(bundle / "integrity.json")
    for relative, digest in expected.items():
        if sha256(bundle / relative) != digest:
            raise ValueError(f"Changed split artifact: {relative}")
    metadata = read_json(bundle / "metadata.json")
    if sha256(root / "train/train.jsonl") != metadata["source_sha256"]:
        raise ValueError("Dataset fingerprint changed")
    if sha256(root / "train/train_tools_catalog.json") != metadata["catalog_sha256"]:
        raise ValueError("Catalog fingerprint changed")
    ids, groups, hashes = set(), set(), set()
    audit = read_json(bundle / "audio_audit.json")
    for split in ("train", "validation", "local_test"):
        manifest = read_json(bundle / split / "manifest.json")
        current_ids = set(manifest["ids"])
        current_groups = set(manifest["groups"].values())
        if len(current_ids) != len(manifest["ids"]) or ids & current_ids or groups & current_groups:
            raise ValueError("ID/group leakage")
        current_hashes = {a["sha256"] for a in audit.values() if any(r["id"] in current_ids for r in a["references"])}
        if hashes & current_hashes:
            raise ValueError("Current/history audio byte leakage")
        ids |= current_ids
        groups |= current_groups
        hashes |= current_hashes
    for relative, entry in audit.items():
        if sha256(root / "train" / relative) != entry["sha256"]:
            raise ValueError(f"Audio fingerprint changed: {relative}")
    return {"verified": True, "samples": len(ids), "audio_paths": len(audit), "overlap": 0}


def freeze(bundle, config_path, output):
    bundle = Path(bundle).resolve()
    config = read_json(config_path)
    # Revisions and weights must be supplied by the team, never guessed by tooling.
    required = ("model_source", "model_revision", "weights_fingerprint", "quantization", "generation", "prompt_sha256")
    if any(key not in config or config[key] is None or config[key] == "" for key in required):
        raise ValueError("Freeze config requires: " + ", ".join(required))
    write_json(output, {"created_utc": datetime.now(timezone.utc).isoformat(), "bundle_integrity_sha256": sha256(bundle / "integrity.json"),
        "config": config, "config_sha256": sha256(config_path), "code": code_fingerprint(),
        "split_artifacts": read_json(bundle / "integrity.json")})


def access_split(bundle, split, *, allow_local_test=False, frozen=None, purpose="evaluate"):
    bundle = Path(bundle)
    if split == "local_test":
        if not allow_local_test or frozen is None:
            raise ValueError("Local test requires --allow-local-test and --frozen; freeze the agent configuration first")
        snapshot = read_json(frozen)
        if snapshot["bundle_integrity_sha256"] != sha256(bundle / "integrity.json"):
            raise ValueError("Frozen split does not match bundle")
        if snapshot["code"]["sha256"] != code_fingerprint()["sha256"]:
            raise ValueError("Agent/evaluator code changed after freeze")
        for relative, digest in snapshot["split_artifacts"].items():
            if sha256(bundle / relative) != digest:
                raise ValueError(f"Frozen artifact changed: {relative}")
        with (bundle / "audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "purpose": purpose,
                "frozen_sha256": sha256(frozen), "warning": "Using test errors to change configuration consumes test independence"}) + "\n")
    if split not in ("train", "validation", "local_test"):
        raise ValueError("Unknown logical split")
    return bundle / split
