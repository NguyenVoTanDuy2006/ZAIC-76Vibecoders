from __future__ import annotations

import importlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError


ASR_RUNTIME_VERSIONS = {"transformers": "4.57.6", "qwen-asr": "0.0.6", "bert-score": "0.3.13", "pyvi": "0.1.1"}


def _installed_versions() -> dict[str, str | None]:
    from importlib.metadata import version

    installed = {}
    for package in ASR_RUNTIME_VERSIONS:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return installed


def ensure_asr_runtime() -> dict[str, str]:
    """Install pinned ASR and semantic metric packages if needed."""
    installed = _installed_versions()
    packages_to_install = [
        f"{package}=={expected}"
        for package, expected in ASR_RUNTIME_VERSIONS.items()
        if installed[package] != expected
    ]
    if not packages_to_install:
        return installed  # type: ignore[return-value]

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages_to_install],
        check=True,
    )
    importlib.invalidate_caches()

    if any(name in sys.modules for name in ("transformers", "qwen_asr", "bert_score", "pyvi")):
        raise RuntimeError(
            "The pinned ASR dependencies were installed, but this Kaggle kernel has already loaded "
            "an older ASR package. Restart the Kaggle kernel, then rerun the notebook from the first cell."
        )

    installed = _installed_versions()
    if installed != ASR_RUNTIME_VERSIONS:
        raise RuntimeError(
            f"ASR dependency installation did not produce the required versions: {installed}; "
            f"expected {ASR_RUNTIME_VERSIONS}."
        )
    return installed  # type: ignore[return-value]
