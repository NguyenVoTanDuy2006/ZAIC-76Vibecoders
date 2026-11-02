from __future__ import annotations

import time

from .artifacts import write_json, write_jsonl
from .data import AudioDataset
from .metrics import summarize_predictions


def evaluate_model(
    model, processor, policy, scorer, records, config, runtime, destination=None
):
    dtype = next(model.parameters()).dtype
    dataset = AudioDataset(
        records[runtime.rank :: runtime.world_size], config["dataset"]["root"]
    )

    def infer():
        rows = []
        for i in range(len(dataset)):
            item = dataset[i]
            record = item["record"]
            prompt = policy.prompt(item, runtime.device, dtype)
            runtime.synchronize()
            start = time.perf_counter()
            tokens, text = policy.generate(model, prompt)
            runtime.synchronize()
            latency = time.perf_counter() - start
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "conversation_id": record["conversation_id"],
                    "is_latest": record["is_latest"],
                    "reference": record["reference"],
                    "prediction": text,
                    "duration_seconds": record["duration_seconds"],
                    "latency_seconds": latency,
                    "truncated": not any(
                        int(t) in policy.eos_ids for t in tokens[0].tolist()
                    ),
                }
            )
        for normalized, prefix in [(False, "raw"), (True, "normalized")]:
            scores = scorer.score_pairs(
                [(r["reference"], r["prediction"]) for r in rows], normalized=normalized
            )
            for row, score in zip(rows, scores):
                row[f"{prefix}_bertscore_f1"] = score.f1
                row[f"{prefix}_semantic_chunked"] = score.chunked
        return rows

    local = runtime.call(infer, "generation/semantic evaluation")
    rows = sorted(
        [r for shard in runtime.gather(local) for r in shard],
        key=lambda r: r["sample_id"],
    )
    summary = summarize_predictions(rows, [r["sample_id"] for r in records])

    def save():
        if destination is not None and runtime.main:
            write_jsonl(destination / "predictions.jsonl", rows)
            write_json(destination / "summary.json", summary)

    runtime.call(save, "write evaluation artifacts")
    return summary
