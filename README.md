# Zalo AI Challenge utilities

## Baseline ASR benchmark

The modular ASR benchmark is under `src/benchmark_baseline_asr`. It compares PhoWhisper-medium/large and Qwen3-ASR-0.6B/1.7B on the training split, reporting raw and normalized CER/WER plus inference timing. Kaggle workers run one model at a time, with one data shard on each T4 GPU.

Install the project and the ASR extra in an environment that already has a compatible PyTorch/CUDA build:

```bash
pip install '.[asr]'
```

Prepare the manifest and the two audio shards, then use the same run ID for the separate workflow commands:

```bash
asr-benchmark prepare-dataset --environment local --run-id example
asr-benchmark download-models --environment local --run-id example
asr-benchmark transcribe --environment local --run-id example
asr-benchmark evaluate --environment local --run-id example
asr-benchmark compare --environment local --run-id example
```

Or run all five workflows in order with one command:

```bash
asr-benchmark benchmark --environment local --run-id example
```

For Kaggle, select `--environment kaggle` and set the dataset root in `src/benchmark_baseline_asr/configs/environments/kaggle.json` to the mounted folder containing `train.jsonl` and `audio/`. Add a Kaggle Secret labelled `HF_TOKEN` before downloading models. The notebook is intentionally outside this source package.

The `source_manager` utility inside the benchmark package downloads a sparse checkout from GitHub and removes only its marked source directory:

```bash
PYTHONPATH=src python -m benchmark_baseline_asr.source_manager.cli download
PYTHONPATH=src python -m benchmark_baseline_asr.source_manager.cli cleanup
```

The cleanup command is separate from benchmark execution so output can be inspected before removing the downloaded source.

Unit tests for the benchmark and source manager are in `src/benchmark_baseline_asr/tests/` and can be run from the repository root with `PYTHONPATH=src python -m unittest discover -s src/benchmark_baseline_asr/tests/unit`.
