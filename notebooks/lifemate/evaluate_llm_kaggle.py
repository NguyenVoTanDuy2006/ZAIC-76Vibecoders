# %% [markdown]
# # Test Qwen + ToolEnv + evaluation hiện tại, không cần push GitHub
# Upload lifemate-kaggle-eval.zip thành Kaggle Dataset private, rồi Add Input.
# Add Input dữ liệu BTC gốc (train.jsonl, catalog, train_environment, zaic_tools).
# Bật GPU. Bật Internet nếu cài dependencies/tải Qwen từ Hugging Face.
# Copy các cell vào notebook mới, hoặc chạy script này bằng %run.
# Mặc định: transcript chuẩn -> Qwen THẬT -> ToolEnv -> evaluate, toàn validation.
# Không chạy ASR, không lấy gold calls thay cho quyết định của model.

# %%
from pathlib import Path
import hashlib
import importlib
import json
import os
import subprocess
import sys
import uuid

# Thường có thể giữ None: launcher tự tìm payload và train/train.jsonl.
PAYLOAD_ROOT = None
DATA_ROOT = None

# "oracle_text": chỉ đo agent với transcript chuẩn trên validation.
# "asr_text": đo agent từ file prediction ASR có sẵn của đội.
MODE = "oracle_text"
SPEECH_FILE = None  # MODE="asr_text": /kaggle/input/.../asr_predictions.jsonl

# Để None nếu muốn tải model public vào /kaggle/working; không tải trên máy local.
# Hoặc gán thư mục Qwen đã Add Input, có config.json/tokenizer/*.safetensors.
MODEL_PATH = None
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = None  # None: resolve commit hiện tại một lần và ghi lại SHA.
INSTALL_DEPENDENCIES = True
WORK_ROOT = Path("/kaggle/working/lifemate-llm-evaluation")


# %%
def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def find_parent(filename, explicit=None, *, directory=None, search_root="/kaggle/input"):
    if explicit is not None:
        parent = Path(explicit).resolve()
        candidate = parent / directory / filename if directory else parent / filename
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return parent
    found = set()
    for folder, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d != "audio" and not d.endswith("_environment") and d != ".git"]
        if filename in files and (directory is None or Path(folder).name == directory):
            found.add(Path(folder).parent if directory else Path(folder))
    if len(found) != 1:
        raise ValueError(f"Found {len(found)} roots for {filename}; set PAYLOAD_ROOT/DATA_ROOT explicitly")
    return found.pop()


def check_payload(root):
    manifest = json.loads((root / "payload_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Changed/missing payload file: {relative}")
    return manifest


def select_source(root):
    # An existing GitHub checkout may remain in the notebook. This path wins.
    previous = sys.modules.get("lifemate")
    if previous is not None and not Path(previous.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Kernel already imported another lifemate. Restart session before using this payload.")
    sys.path.insert(0, str(root / "src"))
    importlib.invalidate_caches()
    module = importlib.import_module("lifemate.evaluation.runner")
    if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Imported evaluation from the wrong source")
    print("Evaluation source:", module.__file__)


# %%
def prepare_model(work):
    import torch
    import transformers
    from transformers import GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Kaggle before running the Qwen FP16 backend")
    if not hasattr(GenerationConfig(), "prefill_chunk_size"):
        raise RuntimeError("Transformers lacks prefill_chunk_size; install the pinned dependencies and restart session")
    print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__, "| transformers:", transformers.__version__)
    if MODEL_PATH is not None:
        model = Path(MODEL_PATH)
        if not (model / "config.json").is_file():
            raise FileNotFoundError(f"MODEL_PATH must contain config.json: {model}")
        return model, MODEL_REVISION
    from huggingface_hub import HfApi, snapshot_download

    # Public Qwen: no token needed. Existing HF_TOKEN is respected by the library.
    commit = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION or "main").sha
    model = work / "model"
    snapshot_download(repo_id=MODEL_ID, revision=commit, local_dir=model,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"])
    return model, commit


# %%
def main():
    if MODE not in ("oracle_text", "asr_text"):
        raise ValueError("MODE must be oracle_text or asr_text")
    payload = find_parent("payload_manifest.json", PAYLOAD_ROOT)
    package = check_payload(payload)
    data_root = find_parent("train.jsonl", DATA_ROOT, directory="train")
    bundle = payload / "bundle"
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    # Detect another data release before downloading/loading a large model.
    for relative, key in (("train/train.jsonl", "source_sha256"), ("train/train_tools_catalog.json", "catalog_sha256")):
        if digest(data_root / relative) != metadata[key]:
            raise ValueError(f"BTC data differs from the prepared validation split: {relative}")
    if MODE == "asr_text" and (SPEECH_FILE is None or not Path(SPEECH_FILE).is_file()):
        raise FileNotFoundError("Set SPEECH_FILE to the team's existing ASR predictions")
    select_source(payload)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if INSTALL_DEPENDENCIES:
        # Dedicated LLM notebook; preserve Kaggle's torch/CUDA installation.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
            "transformers==5.17.0", "accelerate>=1.12,<2", "huggingface-hub>=1.5,<2"])
        if "transformers" in sys.modules:
            raise RuntimeError("Dependencies updated after Transformers was imported. Restart the session, then rerun.")
    model, revision = prepare_model(WORK_ROOT)
    settings = json.loads((payload / "configs/lifemate/agent-evaluation.json").read_text(encoding="utf-8"))
    settings.update(model_source=str(model), model_revision=revision, device="cuda:0")
    run_root = WORK_ROOT / (MODE + "_" + uuid.uuid4().hex[:10])
    run_root.mkdir()
    config = run_root / "config.json"
    config.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    from lifemate.evaluation.runner import run
    from lifemate.evaluation.__main__ import main as evaluation_cli

    result = run(bundle, "validation", data_root, SPEECH_FILE, run_root / "agent",
                 config_path=config, oracle=MODE == "oracle_text")
    command = ["evaluate", "--data-root", str(data_root), "--bundle", str(bundle),
        "--split", "validation", "--predictions", result["predictions"],
        "--traces", result["traces"], "--output", str(run_root / "evaluation")]
    if evaluation_cli(command) != 0:
        raise RuntimeError(f"Evaluation failed. Preserve traces at {run_root}")
    metrics = json.loads((run_root / "evaluation/metrics.json").read_text(encoding="utf-8"))
    summary = {"mode": MODE, "model_id": MODEL_ID, "model_revision": revision,
        "is_real_llm_inference": True, "is_asr_inference": False,
        "validation_samples": result["samples"], "successful_agent_runs": result["successful"],
        "local_estimated_score": metrics["score"], "payload_code_sha256": package["code_sha256"],
        "output": str(run_root)}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Read errors:", run_root / "evaluation/samples.jsonl")
    return summary


# %%
if __name__ == "__main__":
    report = main()
