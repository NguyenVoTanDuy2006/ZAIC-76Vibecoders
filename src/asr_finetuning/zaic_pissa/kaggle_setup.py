"""Standard-library bootstrap helpers; imported before the isolated ML environment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def checked(command, **kwargs):
    return subprocess.run([str(v) for v in command], check=True, **kwargs)


def ensure_pip(python):
    probe = (
        lambda: subprocess.run(
            [str(python), "-m", "pip", "--version"], capture_output=True
        ).returncode
        == 0
    )
    if probe():
        return
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=False)
    if not probe():
        # Host pip can manage a Python environment even if that environment has no pip.
        checked(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                python,
                "install",
                "--upgrade",
                "pip",
            ]
        )
    if not probe():
        raise RuntimeError(f"could not bootstrap pip for {python}")


def python312(temp_root):
    candidates = [
        sys.executable,
        shutil.which("python3.12"),
        "/usr/bin/python3.12",
        "/opt/conda/bin/python3.12",
    ]
    for candidate in filter(None, candidates):
        if not Path(candidate).is_file():
            continue
        result = subprocess.run(
            [str(candidate), "-c", "import sys; print(sys.version_info[:2]==(3,12))"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == "True":
            return Path(candidate)
    bootstrap = Path(temp_root) / "bootstrap"
    checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            bootstrap,
            "pbs-installer[all]==2026.10.25",
        ]
    )
    # Version-independent bootstrap fallback is intentionally confined to Python provisioning.
    env = dict(os.environ, PYTHONPATH=str(bootstrap))
    destination = Path(temp_root) / "python312"
    checked(
        [
            sys.executable,
            "-c",
            'from pbs_installer import install; import sys; install("3.12", sys.argv[1])',
            destination,
        ],
        env=env,
    )
    python = destination / "bin/python3.12"
    if not python.is_file():
        raise FileNotFoundError("Python 3.12 provisioning failed")
    return python


def make_environment(source, temp_root):
    temp_root = Path(temp_root)
    py = temp_root / "venv/bin/python"
    if not py.is_file():
        checked([python312(temp_root), "-m", "venv", "--without-pip", py.parent.parent])
    ensure_pip(py)  # Also repairs a pre-existing incomplete environment.
    checked([py, "-m", "pip", "install", "--upgrade", "pip"])
    checked(
        [
            py,
            "-m",
            "pip",
            "install",
            "torch==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ]
    )
    checked([py, "-m", "pip", "install", "-r", Path(source) / "requirements.txt"])
    checked([py, "-m", "pip", "check"])
    return py


def find_dataset(input_root=Path("/kaggle/input"), override=None):
    if override:
        root = Path(override)
        if not (root / "train.jsonl").is_file() or not (root / "audio").is_dir():
            raise ValueError("DATA_ROOT must contain train.jsonl and audio/")
        return root
    candidates = sorted(
        {
            p.parent
            for p in Path(input_root).rglob("train.jsonl")
            if (p.parent / "audio").is_dir()
        }
    )
    if len(candidates) != 1:
        raise ValueError(
            f"found {len(candidates)} datasets (multiple or missing inputs); set DATA_ROOT explicitly: {candidates}"
        )
    return candidates[0]
