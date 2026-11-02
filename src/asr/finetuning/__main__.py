from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
from pathlib import Path

from .artifacts import fingerprint, write_json
from .config import config_fingerprint, load_config, run_dir
from .locking import RunLock


def required_pins():
    lines = (
        (Path(__file__).resolve().parent / "requirements.txt")
        .read_text()
        .splitlines()
    )
    return dict(
        line.split("==", 1)
        for line in lines
        if "==" in line and not line.startswith("#")
    )


def validate_runtime(info):
    from packaging.version import Version

    if Version(info["python"]).release[:2] != (3, 12):
        raise ValueError("Python 3.12 is required for this training stack")
    for name, expected in required_pins().items():
        actual = info["packages"].get(name)
        if actual is None or Version(actual).base_version != expected:
            raise ValueError(f"runtime requires {name}=={expected}, found {actual}")
        if name == "torch" and Version(actual).local not in {None, "cpu", "cu124"}:
            raise ValueError("runtime requires Torch 2.6 CPU or CUDA 12.4 build")


def package_signature(packages):
    from packaging.version import Version

    result = dict(packages)
    # PyPI 2.6.0 and wheel-index 2.6.0+cu124 are the supported CUDA 12.4 build.
    value = Version(result["torch"])
    result["torch"] = value.base_version + (
        "+cpu" if value.local == "cpu" else "+cu124"
    )
    return result


def provenance(config):
    packages = {}
    for name in required_pins():
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = Path(__file__).resolve().parent
    asr_root = source.parent
    source_files = sorted((*source.glob("*.py"), *(asr_root / "common").glob("*.py")))
    return {
        "python": platform.python_version(),
        "packages": packages,
        "config_fingerprint": config_fingerprint(config),
        "source_fingerprint": fingerprint(
            {p.relative_to(asr_root).as_posix(): p.read_text() for p in source_files}
        ),
        "scorer_fingerprint": fingerprint(config["semantic"]),
        "semantics": "NFC/lowercase/punctuation-to-space/whitespace + PyVi; digits unchanged; no IDF/rescale",
        "resume": "completed epochs, identical world size; restore complete resume bundle",
    }


def write_provenance(root, config):
    old_path = root / "provenance.json"
    info = provenance(config)
    validate_runtime(info)
    if old_path.exists():
        old = json.loads(old_path.read_text())
        for key in ["source_fingerprint", "config_fingerprint", "scorer_fingerprint"]:
            if old[key] != info[key]:
                raise ValueError(
                    f"{key} differs from existing run; use the original source/config or a new run_id"
                )
        if package_signature(old["packages"]) != package_signature(info["packages"]):
            raise ValueError("runtime packages differ from the original run")
        if old["python"].split(".")[:2] != info["python"].split(".")[:2]:
            raise ValueError("Python runtime differs from the original run")
        # Keep the original runtime record, and record the compatible current patch.
        old["last_runtime"] = {"python": info["python"], "packages": info["packages"]}
        info = old
    write_json(old_path, info)
    write_json(root / "config.json", config)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Original Qwen ASR PiSSA + CE/SCST")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "smoke",
            "train",
            "evaluate",
            "bundle",
            "bundle-source",
            "restore",
        ],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "bundle-source":
        from .bundles import create_source_bundle

        print(create_source_bundle(args.destination or Path("zaic-pissa-source.zip")))
        return
    config = load_config(args.config)
    if args.command == "smoke":
        config = copy.deepcopy(config)
        config["output"]["run_id"] += "-smoke"
    root = run_dir(config)
    if args.command == "restore":
        if args.bundle is None:
            parser.error("restore requires --bundle")
        from .bundles import restore_run_bundle

        restore_run_bundle(args.bundle, root)
        print(root)
        return
    if args.command == "bundle":
        from .bundles import create_bundles

        print(
            json.dumps(
                create_bundles(root, args.destination or root.parent / "bundles"),
                indent=2,
            )
        )
        return
    if args.command == "prepare":
        from .data import prepare_dataset

        with RunLock(root):
            write_provenance(root, config)
            _, report = prepare_dataset(config, root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    import torch
    from .runtime import DistributedRuntime
    from .trainer import Trainer

    runtime = DistributedRuntime(config["runtime"]["device"]).initialize(config)
    trainer = Trainer(config, runtime)
    try:

        def metadata():
            if runtime.main:
                root.mkdir(parents=True, exist_ok=True)
                info = write_provenance(root, config)
                info["devices"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
                write_json(root / "provenance.json", info)
                write_json(root / "config.json", config)

        with trainer.exclusive_run():
            runtime.call(metadata, "write run provenance")
            if args.command == "train":
                result = trainer.train(resume=args.resume)
            elif args.command == "smoke":
                result = trainer.smoke()
            else:
                result = trainer.evaluate_test()
        if runtime.main:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        trainer.release()
        runtime.close()


if __name__ == "__main__":
    main()
