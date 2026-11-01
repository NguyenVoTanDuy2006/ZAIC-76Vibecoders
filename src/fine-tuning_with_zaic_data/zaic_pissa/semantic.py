from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metrics import normalize


@dataclass(frozen=True)
class SemanticScore:
    f1: float
    chunked: bool = False


class PhoBERTReward:
    """Official short-text BERTScore; full-content matching for long inputs.

    Overlong text uses separately contextualized windows, not silent truncation.
    This extension is marked chunked; its score is an approximation to a
    hypothetical longer-context PhoBERT, not that unavailable model's score.
    """

    def __init__(self, config, device, model_path=None, *, scorer=None, segmenter=None):
        if scorer is None:
            from bert_score import BERTScorer

            scorer = BERTScorer(
                model_type=str(model_path),
                num_layers=config["num_layers"],
                device=str(device),
                batch_size=config["batch_size"],
                idf=False,
                rescale_with_baseline=False,
                use_fast_tokenizer=False,
            )
        if segmenter is None:
            from pyvi import ViTokenizer

            segmenter = ViTokenizer.tokenize
        self.config, self.device, self.scorer, self.segmenter = (
            config,
            torch.device(device),
            scorer,
            segmenter,
        )
        self.model = scorer._model.to(device).float().eval().requires_grad_(False)
        self.tokenizer = scorer._tokenizer
        limit = self.tokenizer.model_max_length
        self.limit = min(
            limit if isinstance(limit, int) and limit < 100000 else 256, 256
        )
        self.cache = OrderedDict()

    def _prepare(self, text, normalized):
        if normalized:
            text = normalize(text)
        return self.segmenter(text) if text.strip() else ""

    def _long(self, text):
        return (
            len(
                self.tokenizer.encode(
                    text, add_special_tokens=True, truncation=False, verbose=False
                )
            )
            > self.limit
        )

    def _embeddings(self, text, cache_reference=False):
        if cache_reference and text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text].to(self.device)
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        size = self.limit - self.tokenizer.num_special_tokens_to_add(pair=False)
        if size < 1 or not tokens:
            raise ValueError("invalid semantic token window/content")
        pieces = []
        for start in range(0, len(tokens), size):
            item = self.tokenizer.prepare_for_model(
                tokens[start : start + size],
                add_special_tokens=True,
                return_special_tokens_mask=True,
                truncation=False,
            )
            inputs = {
                k: torch.tensor([v], device=self.device)
                for k, v in item.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}
            }
            output = self.model(**inputs)
            hidden = (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output[0]
            )
            keep = ~torch.tensor(
                item["special_tokens_mask"], device=self.device, dtype=torch.bool
            )
            pieces.append(hidden[0, keep])
        result = F.normalize(torch.cat(pieces, dim=0).float(), p=2, dim=-1)
        if cache_reference and self.config["reference_cache_size"]:
            self.cache[text] = result.detach().cpu()
            while len(self.cache) > self.config["reference_cache_size"]:
                self.cache.popitem(last=False)
        return result

    def score_pairs(self, pairs, normalized=True):
        pairs = [
            (self._prepare(a, normalized), self._prepare(b, normalized))
            for a, b in pairs
        ]
        scores = [None] * len(pairs)
        short = []
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=False),
        ):
            for i, (ref, hyp) in enumerate(pairs):
                if not ref.strip() or not hyp.strip():
                    scores[i] = SemanticScore(
                        1.0 if not ref.strip() and not hyp.strip() else 0.0
                    )
                elif self._long(ref) or self._long(hyp):
                    a, b = self._embeddings(ref, True), self._embeddings(hyp)
                    sim = b @ a.T
                    precision, recall = (
                        sim.max(dim=1).values.mean(),
                        sim.max(dim=0).values.mean(),
                    )
                    denominator = precision + recall
                    value = (
                        float((2 * precision * recall / denominator).cpu())
                        if float(denominator) != 0
                        else 0.0
                    )
                    scores[i] = SemanticScore(value, True)
                else:
                    short.append(i)
            if short:
                _, _, values = self.scorer.score(
                    [pairs[i][1] for i in short],
                    [pairs[i][0] for i in short],
                    batch_size=self.config["batch_size"],
                )
                for i, value in zip(short, values.detach().cpu().tolist()):
                    scores[i] = SemanticScore(float(value))
        if any(s is None or not math.isfinite(s.f1) for s in scores):
            raise ValueError("semantic scorer returned incomplete/nonfinite results")
        return scores
