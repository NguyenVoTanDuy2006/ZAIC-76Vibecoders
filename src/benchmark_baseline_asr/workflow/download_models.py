from __future__ import annotations

import argparse
import os

from benchmark_baseline_asr.application.download_service import download_models as download_selected_models
from benchmark_baseline_asr.infrastructure.secrets.kaggle_secrets import get_huggingface_token
from benchmark_baseline_asr.infrastructure.storage.artifacts import read_json, write_json
from benchmark_baseline_asr.workflow.bootstrap import create_run


def download(config_path: str, environment: str | None = None, run_id: str | None = None, *, token: str | None = None):
    config, run_dir = create_run(config_path, environment, run_id)
    token_label = config["_loaded"]["secrets"].get("token_label", "HF_TOKEN")
    if token is not None:
        secret = token
    elif config["_loaded"]["secrets"].get("provider", "kaggle") == "kaggle":
        secret = get_huggingface_token(token_label)
    else:
        secret = os.environ.get(config["_loaded"]["secrets"].get("token_env", "HF_TOKEN"))
    resolved = download_selected_models(config["models"], config["_loaded"]["download"], secret)
    write_json(run_dir / "models_resolved.json", resolved)
    return run_dir, resolved


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run_dir, resolved = download(args.config, args.environment, args.run_id)
    print(f"run_dir={run_dir}\nmodels={len(resolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
