# ZAIC Qwen PiSSA + SCST Implementation Plan

> Execute inline with superpowers:executing-plans. This document records the user-approved specification and implementation checklist.

**Goal:** Modular fine-tuning of original Qwen3-ASR-1.7B-hf using PiSSA and `CE + 0.1 * SCST`, on Kaggle T4 x2, for three mixed epochs.

**Architecture:** Self-contained package `zaic_pissa` inside the requested `src/asr_finetuning`. Separate configuration, data, model/PiSSA lifecycle, reward, objectives, distributed runtime, training, evaluation and checkpoints. Thin CLI and Kaggle percent script. Port relevant Research code without importing Research at runtime.

## Approved specification

- Original base: `Qwen/Qwen3-ASR-1.7B-hf`, revision `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`.
- All 5,036 labelled user clips, including history, about 6.83 hours. Each clip is an independent waveform/transcript pair; assistant/tool history is not a target.
- Conversation split 80/10/10. Sort IDs, shuffle with Python Random(1234); first ceil(N*.1) test, next ceil(N*.1) validation, remainder train. Current counts: train 1,235 conversations/4,043 clips; validation 155/470; test 155/523.
- Preserve original audio and transcript. Verify WAV mono PCM16 16kHz, paths, nonempty labels and PCM hash disjointness across splits. Save portable manifest fingerprint. No speaker-disjoint claim.
- PiSSA rank 8, alpha 16, dropout 0, bias none, pissa_niter_4, q_proj/v_proj in encoder and decoder; frozen original base. Same initialization on both DDP ranks.
- Exactly 3 epochs; CE and SCST on every batch; no early stopping. AdamW LR 5e-5, weight decay .01, betas (.9,.999), eps 1e-8; linear decay and 3% warmup. Seed 1234, batch 1/GPU, accumulation 8, FP16, clipping 1, non-reentrant gradient checkpointing, 2 loader workers/rank.
- Reward: frozen `vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6`, layer 12, BERTScore F1, PyVi, no IDF or baseline rescale. Normalize like existing benchmark (NFC, lowercase, punctuation to spaces, collapse whitespace; keep number spelling).
- Each clip: current-model greedy baseline and one sampled hypothesis. Advantage = sampled F1 - greedy F1, detached, unscaled and unclipped. SCST loss = -advantage * sum of sampled-token log probabilities. CE is a supervised token mean. Total loss = CE + .1 SCST.
- Vietnamese, 768 generated tokens. Sampling temperature 1/top_p 1/top_k 0. Suppress structural special tokens except EOS both in rollout and replay probability. Use actual generated IDs, include first EOS, exclude prompt/pad/after-EOS. No gradient detachment through audio encoder.
- Empty prediction reward 0; overlong text is fully chunked, embeddings concatenated for whole-pair matching, flagged chunked. Scorer/nonfinite errors abort, not fake rewards.
- Every epoch: validation latest and all-user metrics. Best is highest mean normalized F1 on 155 validation latest clips; ties within 1e-6 use WER, CER, then earlier epoch. Always report raw/normalized WER/CER as well.
- Base validation before attaching PiSSA. After three epochs, original base vs freshly reloaded best export on the held-out test under identical inference settings. Test never selects an epoch.
- Export every epoch as standard rank-16 LoRA delta loadable on original base, plus processor and method metadata. Preserve training model/config/flags/RNG during export.
- Keep 3 epoch adapters, best pointer/export, and only latest full training checkpoint. Checkpoint: native factors, initial factors, residual weights, optimizer, scheduler, scaler, RNG per rank, config/data/scorer fingerprints, history and best metadata. Atomic replacement. Resume only at completed epoch boundaries, same world size, with the run artifact bundle restored.
- Kaggle `.py` cells, pip-only isolated Python 3.12 environment. Torch 2.6.0 CUDA12.4, Transformers5.17.0, PEFT0.21.1, bert-score0.3.13, pyvi0.1.1. Inputs/Secrets from Kaggle; source/venv/base cache in /kaggle/temp; results/checkpoints in /kaggle/working. Separate small report/adapter ZIP and resumable bundle.

## Review focus

1. Policy sampling/replay equality, actual generated IDs and EOS masks; wrong masks silently optimize another objective.
2. Frozen semantic encoder and nonzero PiSSA gradients in both encoder/decoder; no scoring errors disguised as zero reward.
3. DDP identical collective/backward schedules, including zero advantages, failures, partial accumulation and uneven evaluation shards.
4. PiSSA export/resume residual correctness and preservation of the live training adapter.
5. Resume after moving Kaggle mounts, durable epoch commit/retention and no test leakage or duplicate metrics.

## Implementation checklist

- [x] Config, data validation, deterministic manifests/splits; CPU fixture tests and real-data prepare.
- [x] SCST policy, masks, reward encoder/chunking; gradient-sign and frozen encoder tests.
- [x] PiSSA/model lifecycle and atomic resumable checkpoints; export/reload/resume tests.
- [x] DDP trainer/evaluation, finite-gradient updates, selection and fair baseline.
- [x] CLI, Kaggle launcher, bundles, documentation and tests.
- [x] Entire existing unit suite, new unit/integration tests, fresh review and fixes.

## Acceptance

Unit tests cover input/split, assistant labels, objective signs/scaling/masks, semantic empty/long text, selection, checkpoint validation and atomic failures. Small random Qwen integration covers real PiSSA forward/backward, generation/replay, export and resumed next update. Two-process CPU/Gloo tests cover distributed schedules and metrics where feasible. Real two-T4 GPU smoke is provided in Kaggle and is required before the three-epoch run; this local host has no CUDA GPU. Software correctness is not a promise that test BERTScore improves.
