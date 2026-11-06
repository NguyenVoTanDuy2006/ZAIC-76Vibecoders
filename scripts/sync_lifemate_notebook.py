"""Regenerate the self-contained Kaggle test notebook from current local source.

Compatibility entry point: python scripts/sync_lifemate_notebook.py
No dataset, model, network access or Git mutation is involved.
"""
from pathlib import Path
import runpy


def sync():
    builder = Path(__file__).resolve().with_name("build_lifemate_test_notebook.py")
    return runpy.run_path(str(builder), run_name="__main__")


if __name__ == "__main__":
    sync()
