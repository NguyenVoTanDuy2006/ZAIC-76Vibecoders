"""Allowlisted reports and resumable run archives; models/credentials stay outside."""

from __future__ import annotations

import os
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .locking import RunLock

FILES = {
    "config.json",
    "provenance.json",
    "dataset.json",
    "history.json",
    "best.json",
    "best_model.json",
    "smoke.json",
}
DIRECTORIES = {
    "manifests",
    "adapters",
    "best_adapter",
    "validation",
    "test",
    "smoke_adapter",
    "smoke_reloaded",
}


def _allowed(relative, resume):
    parts = relative.parts
    if any(p.startswith(".") for p in parts):
        return False
    return (
        len(parts) == 1
        and parts[0] in FILES
        or parts[0] in DIRECTORIES
        or resume
        and (
            parts[0] == "pissa_initial"
            or relative.as_posix() == "checkpoints/latest.pt"
        )
    )


def create_bundles(root, destination):
    with RunLock(root):
        return _create_bundles(root, destination)


def _create_bundles(root, destination):
    import torch

    root, destination = Path(root).resolve(), Path(destination).resolve()
    if not (root / "checkpoints/latest.pt").is_file():
        raise FileNotFoundError("no completed epoch checkpoint to bundle")
    destination.mkdir(parents=True, exist_ok=True)
    # The lease holds the checkpoint and all committed exports stable for both ZIPs.
    state = torch.load(
        root / "checkpoints/latest.pt", map_location="cpu", weights_only=False
    )
    epoch, best = state["epoch"], state["best"]
    for number in range(1, epoch + 1):
        if not (
            root / f"adapters/epoch-{number:02d}/adapter_model.safetensors"
        ).is_file():
            raise ValueError(
                "run is missing a committed adapter; restore the complete bundle"
            )
    metadata = {"history.json": state["history"], "best.json": best}
    best_source = root / best["adapter"]
    if (
        not best_source.is_relative_to(root)
        or best_source != root / f"adapters/epoch-{best['epoch']:02d}"
        or best["epoch"] > epoch
    ):
        raise ValueError("checkpoint best adapter is outside the committed epochs")
    if (root / "config.json").is_file():
        model = json.loads((root / "config.json").read_text())["model"]
        metadata["best_model.json"] = {
            **model,
            "adapter_path": "best_adapter",
            "training_method": "pissa_converted_lora",
            "selected_epoch": best["epoch"],
            "selection": "validation.latest.normalized_bertscore_f1",
        }
    del state
    outputs = {}
    for kind in ["reports", "resume"]:
        target = destination / f"{root.name}-{kind}.zip"
        fd, archive_tmp = tempfile.mkstemp(
            prefix=f".{kind}-", suffix=".zip", dir=destination
        )
        os.close(fd)
        try:
            with zipfile.ZipFile(
                archive_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
            ) as archive:
                for path in sorted(root.rglob("*")):
                    relative = path.relative_to(root)
                    if (
                        relative.as_posix() in metadata
                        or relative.parts[0] == "best_adapter"
                    ):
                        continue
                    if (
                        relative.parts[0] in {"adapters", "validation"}
                        and len(relative.parts) > 1
                    ):
                        name = relative.parts[1]
                        if name.startswith("epoch-") and (
                            not name[6:].isdigit() or int(name[6:]) > epoch
                        ):
                            continue
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and _allowed(relative, kind == "resume")
                    ):
                        archive.write(path, relative.as_posix())
                for name, value in metadata.items():
                    archive.writestr(
                        name,
                        json.dumps(
                            value, ensure_ascii=False, allow_nan=False, indent=2
                        ),
                    )
                for path in sorted(best_source.rglob("*")):
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and not any(
                            p.startswith(".")
                            for p in path.relative_to(best_source).parts
                        )
                    ):
                        archive.write(
                            path,
                            ("best_adapter" / path.relative_to(best_source)).as_posix(),
                        )
            os.replace(archive_tmp, target)
        finally:
            Path(archive_tmp).unlink(missing_ok=True)
        outputs[kind] = str(target)
    return outputs


def restore_run_bundle(bundle, destination):
    with RunLock(destination):
        return _restore_run_bundle(bundle, destination)


def _restore_run_bundle(bundle, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("restore into a new run directory")
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        seen = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or not path.parts
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.filename in seen
                or not _allowed(path, True)
            ):
                raise ValueError(f"invalid run bundle member: {member.filename}")
            seen.add(member.filename)
        if "checkpoints/latest.pt" not in seen:
            raise ValueError("bundle has no native checkpoint")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination.parent))
        try:
            archive.extractall(staging)
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def create_source_bundle(destination):
    source = Path(__file__).resolve().parents[1]
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if (
                path.is_file()
                and not any(
                    p.startswith(".") or p == "__pycache__" for p in relative.parts
                )
                and path.suffix in {".py", ".json", ".md", ".txt"}
            ):
                archive.write(path, relative.as_posix())
    return str(destination)
