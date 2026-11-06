"""Create a local, uploadable validation overlay. Does not publish anything."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


REPO = Path(__file__).resolve().parents[1]


def package(bundle, output):
    bundle, output = Path(bundle), Path(output)
    files = {}
    for path in sorted((REPO / "src/lifemate").rglob("*.py")):
        files[path.relative_to(REPO).as_posix()] = path.read_bytes()
    for relative in ("configs/lifemate/agent-evaluation.json", "notebooks/lifemate/evaluate_llm_kaggle.py",
                     "docs/lifemate/kaggle-without-push.md"):
        files[relative] = (REPO / relative).read_bytes()
    # Only validation is portable. No train/local-test gold, audio, environment or weights.
    for relative in ("metadata.json", "integrity.json", "validation/manifest.json", "validation/inputs.jsonl", "validation/references.jsonl"):
        files["bundle/" + relative] = (bundle / relative).read_bytes()
    integrity = json.loads(files["bundle/integrity.json"])
    manifest = json.loads(files["bundle/validation/manifest.json"])
    for name in ("manifest.json", "inputs.jsonl", "references.jsonl"):
        relative = "validation/" + name
        if hashlib.sha256(files["bundle/" + relative]).hexdigest() != integrity[relative]:
            raise ValueError(f"Changed split artifact: {relative}")
    hashes = {relative: hashlib.sha256(blob).hexdigest() for relative, blob in files.items()}
    code_hash = hashlib.sha256(json.dumps({k: v for k, v in hashes.items() if k.startswith("src/")}, sort_keys=True).encode()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    payload = {"format": 1, "purpose": "Unpushed local source overlay for Kaggle LLM validation",
        "git_base_commit": head, "code_sha256": code_hash, "validation_samples": len(manifest["ids"]),
        "contains_local_uncommitted_code": True, "contains_model_weights": False,
        "contains_local_test": False, "files": hashes}
    files["payload_manifest.json"] = (json.dumps(payload, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, blob in files.items():
            archive.writestr("lifemate_kaggle_eval/" + relative, blob)
    return {"path": str(output), "bytes": output.stat().st_size, "files": len(files), "validation_samples": len(manifest["ids"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=REPO / "output/lifemate/evaluation/agent-splits")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.bundle, args.output)))
