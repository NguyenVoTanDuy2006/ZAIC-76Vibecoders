# Notebook conventions

- If the user has not made the execution platform clear, ask whether the notebook is for local execution or Kaggle before implementing it.
- Kaggle notebooks use Python percent scripts (`.py`, `# %%` cells), with `pip` for installation. Read attached data/adapters from `/kaggle/input`, save results under `/kaggle/working`, and use Kaggle Secrets for credentials.
- Local notebooks use `.ipynb`.

## LifeMate ASR handoff

- Names follow purpose: `src/asr_benchmark` is the importable benchmark package, `src/asr_finetuning` contains the isolated `zaic_pissa` trainer, and `src/lifemate` is integration/agent code. Distribution name is `lifemate`; console commands are `asr-benchmark` and `lifemate-agent`. Do not recreate the deleted placeholder package.
- Current-source Kaggle launchers target branch `agent`. The adapter-benchmark launcher remains pinned to its historical SHA and detects that checkout's original package layout; keep that compatibility when changing imports.

- LifeMate code is under `src/lifemate/`: `agent.py` owns the single agent loop, `asr/predict.py` reuses existing benchmark ASR adapters, and `asr/predictions.py` is the standard-library-only speech contract/file reader.
- The default config is the existing Qwen3-ASR-1.7B PiSSA config. Model/adapter paths are explicit; inference defaults to offline/cache only. Preserve the separate baseline SDK and native Transformers 5 environments.
- Predict current audio and all user-history audio. Preserve original history indices. Never use `audio_text`, `label`, or `reference` to replace missing inference, and never read `*_environment` files directly.
- Launchers are `notebooks/lifemate/predict_asr.py` and `notebooks/lifemate/run_agent.ipynb`. The notebook imports the agent module instead of embedding a second copy. Shared `lifemate.paths` must make ASR output equal agent input from both repo-root and nested working directories.
- Leave `load_toolenv()` in `src/lifemate/agent.py` as a visible team-lead hook. Exactly one ToolEnv per sample; canonical output comes from its log/hash. Keep `main()` simple and run full splits by default.
- Focused tests: set `PYTHONPATH=src`, then `python -m unittest discover -s tests/lifemate -p "test_*.py" -v`. Team flow documentation is `docs/lifemate/README.md`; root README contains the current structure and run order.
- Test scratch is `.test-scratch/` under the repo and is removed after each case. Keep predictions/weights/credentials out of commits. CPU fake-backend tests are not GPU inference or competition quality results.
