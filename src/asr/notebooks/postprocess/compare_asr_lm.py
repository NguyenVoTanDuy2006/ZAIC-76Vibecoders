# %% [markdown]
# # ZAIC PiSSA → Qwen3-4B ASR correction comparison
# Kaggle GPU launcher. Attach the previous PiSSA output, ZAIC train audio,
# pinned Qwen3-ASR base snapshot, and Qwen3-4B-Instruct-2507 snapshot as Inputs.
# Source defaults to current `asr` branch; set SOURCE_ROOT for attached source.

# %%
from pathlib import Path
import subprocess
import sys

SOURCE_ROOT = None  # /kaggle/input/zaic-source/repository
SOURCE_REPOSITORY = "https://github.com/NguyenVoTanDuy2006/ZAIC-76Vibecoders.git"
SOURCE_REF = "asr"
SOURCE_WORK = Path("/kaggle/working/zaic-asr-lm-source")

RUN_INPUT = Path("/kaggle/input/zaic-qwen-pissa/zaic-pissa-scst-v1")
PREDICTIONS = RUN_INPUT / "validation/epoch-03/predictions.jsonl"
TRAIN_MANIFEST = RUN_INPUT / "manifests/train.jsonl"
ASR_CONFIG = RUN_INPUT / "config.json"
ADAPTER_PATH = RUN_INPUT / "best_adapter"
TRAIN_AUDIO_ROOT = Path("/kaggle/input/zaic-data/train+public_test/train")
ASR_MODEL_PATH = Path("/kaggle/input/qwen3-asr-1-7b-hf")
LM_MODEL_PATH = Path("/kaggle/input/qwen3-4b-instruct-2507")
OUTPUT_DIR = Path("/kaggle/working/zaic-asr-lm/validation-epoch-03")


# %%
def checked(*args):
    subprocess.run([str(part) for part in args], check=True)


def resolve_source():
    if SOURCE_ROOT:
        root = Path(SOURCE_ROOT).resolve()
    else:
        if not (SOURCE_WORK / ".git").is_dir():
            checked("git", "clone", "--depth", "1", "--filter=blob:none",
                    "--sparse", "--branch", SOURCE_REF, SOURCE_REPOSITORY, SOURCE_WORK)
            checked("git", "-C", SOURCE_WORK, "sparse-checkout", "set", "src/asr")
        root = SOURCE_WORK
    if not (root / "src/asr/postprocess/experiment.py").is_file():
        raise FileNotFoundError(f"Source lacks ASR postprocess package: {root}")
    return root


# %%
def main():
    root = resolve_source()
    # Native Transformers 5 environment. Do not install the qwen-asr SDK here.
    checked(sys.executable, "-m", "pip", "install", "-q", "-r",
            root / "src/asr/finetuning/requirements.txt")
    sys.path.insert(0, str(root / "src"))
    from asr.postprocess.experiment import run_experiment

    summary = run_experiment(
        predictions_path=PREDICTIONS, train_manifest=TRAIN_MANIFEST,
        train_audio_root=TRAIN_AUDIO_ROOT, asr_config=ASR_CONFIG,
        asr_model_path=ASR_MODEL_PATH, adapter_path=ADAPTER_PATH,
        lm_model_path=LM_MODEL_PATH, output_dir=OUTPUT_DIR,
    )
    print("Results:", OUTPUT_DIR / "comparison.csv")
    for name, groups in summary.items():
        metrics = groups["all"]
        print(f"{name}: normalized WER={metrics['normalized_wer']:.4%}; "
              f"normalized CER={metrics['normalized_cer']:.4%}; "
              f"raw CER={metrics['raw_cer']:.4%}")


if __name__ == "__main__":
    main()
