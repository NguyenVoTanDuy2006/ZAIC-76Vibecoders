"""One immutable subset selection shared by inference, validation and scoring."""
from __future__ import annotations

from .io import index_rows, read_json, sha256, write_json


def create_selection(bundle, split, output, *, sample_ids=None, limit=None):
    from pathlib import Path

    manifest_path = Path(bundle) / split / "manifest.json"
    manifest = read_json(manifest_path)
    ids = manifest["ids"]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Parent manifest must contain unique, nonempty IDs")
    if sample_ids is not None:
        selected = list(sample_ids)
        if not selected or any(not isinstance(sid, str) for sid in selected):
            raise ValueError("sample_ids must contain nonempty string IDs")
        if len(selected) != len(set(selected)) or set(selected) - set(ids):
            raise ValueError("Duplicate IDs or IDs outside the selected logical split")
    else:
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        selected = ids[:] if limit is None else ids[:limit]
    if split == "local_test" and selected != ids:
        raise ValueError("Local test must use its full frozen manifest")
    selection = {"format": 1, "split": split, "physical_split": manifest["physical_split"],
                 "parent_manifest_sha256": sha256(manifest_path), "ids": selected,
                 "groups": {sid: manifest["groups"][sid] for sid in selected}}
    write_json(output, selection)
    return selection


def select_rows(rows, manifest_path, selection_path=None):
    """Validate against the parent; never filter predictions to hide missing IDs."""
    manifest = read_json(manifest_path)
    indexed = index_rows(rows)
    if list(indexed) != manifest["ids"]:
        raise ValueError("Manifest/input ID mismatch")
    if selection_path is None:
        return rows
    selection = read_json(selection_path)
    ids = selection.get("ids")
    if (selection.get("format") != 1 or selection.get("split") != manifest["split"]
            or selection.get("physical_split") != manifest["physical_split"]
            or selection.get("parent_manifest_sha256") != sha256(manifest_path)):
        raise ValueError("Selection does not match the parent manifest")
    if (not isinstance(ids, list) or not ids or any(not isinstance(sid, str) for sid in ids)
            or len(ids) != len(set(ids)) or set(ids) - set(indexed)):
        raise ValueError("Selection contains invalid, duplicate or foreign IDs")
    if selection.get("groups") != {sid: manifest["groups"][sid] for sid in ids}:
        raise ValueError("Selection group mapping differs from the parent")
    if manifest["split"] == "local_test" and ids != manifest["ids"]:
        raise ValueError("Local test must use its full frozen manifest")
    return [indexed[sid] for sid in ids]
