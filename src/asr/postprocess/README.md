# ASR transcript correction experiment

The Kaggle launcher is `src/asr/notebooks/postprocess/compare_asr_lm.py`. Edit its
input paths to attached Kaggle datasets before running. It needs:

- `validation/epoch-03/predictions.jsonl`, `manifests/train.jsonl`, and
  `config.json` from `output/zaic-qwen-pissa/zaic-pissa-scst-v1`;
- the ZAIC train audio directory, the pinned Qwen3-ASR-1.7B base snapshot, and
  that run's `best_adapter` directory;
- a local snapshot of `Qwen/Qwen3-4B-Instruct-2507` with `config.json` and tokenizer.

The launcher uses the current `asr` branch unless `SOURCE_ROOT` points to an
attached source checkout. Run source containing this package; a branch that has
not received these files cannot launch the experiment. The native Transformers 5
requirements are installed with `pip`. ASR is loaded only to produce three
few-shot examples from train audio, then unloaded before the 4B LM is loaded.

The run writes under `/kaggle/working/zaic-asr-lm/validation-epoch-03`:

| Path | Purpose |
| --- | --- |
| `predictions/*.jsonl` | One ASR baseline and six correction variants |
| `comparison.csv`, `comparison.json`, `selection.json` | CER/WER and the normalized-WER winner |
| `audit.jsonl`, `check_counts.json` | Candidate text and hard/soft decisions |
| `generation/`, `judge/`, `train_support_predictions.jsonl` | Resumable per-clip caches |
| `run_metadata.json`, `fewshot_examples.json` | Input identity and train-only prompt examples |

The primary comparison metric is normalized WER. All LM calls count tokens after
the Qwen chat template and reserve output tokens within a 16,384-token context.
The current ASR text is never truncated; old user-history ASR is dropped first
if needed. The model never receives validation `reference` text. Results from
this experiment do not replace the agent's ASR input file.
